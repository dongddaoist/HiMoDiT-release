"""
SMILES -> hierarchical layout labels.
=====================================

The inverse of `himodit.chem.decoder`. Takes a SMILES string and emits
the label dict that all four training stages consume, or a rejection
reason explaining which structural rule the molecule violated.

Pipeline per molecule
---------------------
  1. Parse and sanitize with RDKit.
  2. Detect terminal fragments (maximal SMARTS matches with exactly one
     bond crossing out of the matched atom set). Everything not in a
     terminal is scaffold.
  3. Collect scaffold rings via SSSR.
  4. Classify every ring pair as fused, linked, spiro, or unrelated, and
     reject topologies the layout vocabulary cannot express.
  5. Order rings canonically (BFS from the ring with the lowest minimum
     atom index) and extract each ring's canonical traversal.
  6. Walk out from ring atoms to build branch trees for side chains.
  7. Assemble R, F, L, spiro_atom_positions, B_*, and atom_ids in the
     canonical atom order the decoder reconstructs.

Two encoders live here.

`extract_layout` is the current one: branch-tree side chains and spiro
junctions at sp3-quaternary C / N+ / Si centres. Measured retention on
ZINC250K is 93.6%.

`extract_layout_baseline` is the earlier encoder, which represented side
chains as linear chains only and rejected all spiro. Measured retention
83.4%. It is kept so the improvement can be reproduced from a clean
checkout, and because it performs a decode round-trip check that the
current encoder does not.

Known limitation
----------------
`extract_layout` grows branch trees from ring atoms only, so a
substituent attached to a *linker* atom is collected by neither the ring
walk nor the linker path, and is silently dropped. This affects 6.45% of
accepted ZINC250K labels (measured; see docs/limitations.md). Pass
`strict=True` to reject those molecules instead, at a cost of roughly
6 percentage points of retention.
"""
from __future__ import annotations

import os
import sys
import pickle
from collections import deque, defaultdict
from typing import Tuple, List, Optional, Dict, Set, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import networkx as nx

# Import the decoder constants and utilities from Batch 1.1
from himodit.chem.decoder import (
    R_MAX, M_MAX, P_MAX, L_MAX, P_LEN_MAX,
    RING_PAD, RING_6_AROM, RING_6_ALIPH, RING_5_AROM, RING_5_ALIPH,
    RING_3_AROM, RING_3_ALIPH, RING_4_AROM, RING_4_ALIPH,
    RING_7_AROM, RING_7_ALIPH,
    F_NONE, F_FUSED, F_LINKED,
    BOND_NONE, BOND_SINGLE, BOND_AROMATIC,
    AROMATIC_ATOM_IDS,
    decode_scaffold_baseline,
    list_valid_pendant_positions,
)

# Import terminal detection (the earlier version algorithm + the earlier encoder extended SMARTS)
# K=22 terminal vocabulary with charge-form coverage.
from himodit.chem.terminal_smarts import CURATED_TERMINALS
from himodit.chem.terminal_detection import (
    detect_terminals_in_molecule, compile_patterns,
)

# Pre-compile SMARTS once at module load
_COMPILED_TERMINAL_PATTERNS = compile_patterns(CURATED_TERMINALS)


# ─── the earlier version atom vocab (matches config.py) ────────────────────────────
# {0:<PAD>, 1:c, 2:O, 3:C, 4:N, 5:n, 6:S, 7:F, 8:s, 9:o}
# Includes charged species; see docs/label_schema.md.
ATOM_VOCAB = ["<PAD>", "c", "O", "C", "N", "n", "S", "F", "s", "o", "O-", "N+", "n+", "N-", "n-", "P+"]
ATOM_SYMBOL_TO_ID = {sym: i for i, sym in enumerate(ATOM_VOCAB)}


def _rdkit_atom_to_vocab_id(atom: Chem.Atom) -> Optional[int]:
    """Charge-aware vocab mapping.

    Neutral atoms behave as before (case-folded for aromatic).
    Charged atoms map to one of: O-, N+, n+, N-, n-, P+.
    Anything else (e.g. [C-], [S+], [o+]) returns None and the molecule
    is rejected with `atom_*_not_in_vocab_*`.
    """
    sym = atom.GetSymbol()
    is_arom = atom.GetIsAromatic()
    chg = atom.GetFormalCharge()
    if chg == 0:
        if is_arom:
            return ATOM_SYMBOL_TO_ID.get(sym.lower())
        return ATOM_SYMBOL_TO_ID.get(sym)
    if is_arom:
        if sym == "N" and chg == +1: return ATOM_SYMBOL_TO_ID.get("n+")
        if sym == "N" and chg == -1: return ATOM_SYMBOL_TO_ID.get("n-")
        return None
    if sym == "O" and chg == -1: return ATOM_SYMBOL_TO_ID.get("O-")
    if sym == "N" and chg == +1: return ATOM_SYMBOL_TO_ID.get("N+")
    if sym == "N" and chg == -1: return ATOM_SYMBOL_TO_ID.get("N-")
    if sym == "P" and chg == +1: return ATOM_SYMBOL_TO_ID.get("P+")
    return None


def _classify_ring_pair(
    m: Chem.Mol, ring_a_idx: int, ring_b_idx: int, all_rings: List[Tuple[int, ...]]
) -> Tuple[str, Dict]:
    """Determine relation between two scaffold rings.

    Returns ('fused' | 'linked' | 'spiro_invalid' | 'peri_invalid' |
             'no_clean_endpoint' | 'none', info dict).

    fused: share exactly 2 atoms with a bond between them
    linked: connected by a path of NON-RING atoms only (length >= 0)
    spiro_invalid: share 1 atom (spiro fusion, not supported)
    peri_invalid: share >2 atoms (peri fusion, not supported)
    no_clean_endpoint: rings can't be linked because no ring atom is
                       exclusively in one ring (only inside complex
                       fused systems)
    none: not directly related
    """
    ring_a = all_rings[ring_a_idx]
    ring_b = all_rings[ring_b_idx]
    set_a = set(ring_a)
    set_b = set(ring_b)
    shared = set_a & set_b

    if len(shared) == 2:
        a1, a2 = list(shared)
        bond = m.GetBondBetweenAtoms(a1, a2)
        if bond is not None:
            return ("fused", {"shared_atoms": sorted(shared)})
        return ("spiro_invalid", {"shared_atoms": sorted(shared)})
    elif len(shared) == 1:
        return ("spiro_invalid", {"shared_atoms": sorted(shared)})
    elif len(shared) > 2:
        return ("peri_invalid", {"shared_atoms": sorted(shared)})

    # No shared atoms; look for linker path through non-ring atoms.
    other_ring_atoms = set()
    for k, r in enumerate(all_rings):
        if k != ring_a_idx and k != ring_b_idx:
            other_ring_atoms.update(r)

    a_exclusive = set_a - other_ring_atoms
    b_exclusive = set_b - other_ring_atoms
    if not a_exclusive or not b_exclusive:
        return ("no_clean_endpoint", {})

    # BFS: start in ring_a's exclusive atoms; the FIRST atom in the path
    # is the ring_a-side endpoint. Track the path so we can recover both
    # endpoints. linker_length = number of NON-ring atoms in the path
    # (not including the two ring endpoints).
    for start in a_exclusive:
        visited = {start}
        # queue: (current_atom, ring_a_endpoint, linker_atoms_so_far)
        queue = deque([(start, start, [])])
        while queue:
            cur, ring_a_endpoint, linker_atoms = queue.popleft()
            for neighbor in m.GetAtomWithIdx(cur).GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx in visited:
                    continue
                if n_idx in b_exclusive:
                    # Found ring_b. Linker = the non-ring atoms walked through.
                    return (
                        "linked",
                        {
                            "linker_length": len(linker_atoms),
                            "attach_a": ring_a_endpoint,
                            "attach_b": n_idx,
                            "linker_atoms": list(linker_atoms),
                        },
                    )
                if n_idx in set_a or n_idx in set_b or n_idx in other_ring_atoms:
                    continue
                visited.add(n_idx)
                queue.append((n_idx, ring_a_endpoint, linker_atoms + [n_idx]))
    return ("none", {})


def _are_edges_opposite(
    ring_atoms_in_order: List[int],
    edge_a: Set[int],
    edge_b: Set[int],
) -> bool:
    """Two edges (each a 2-atom set) on a ring with atoms in cyclic order:
    are they 'opposite' (linear fusion)?

    For 6-ring: opposite means the edges are exactly 3 positions apart.
    For 5-ring: opposite means 2 positions apart (not adjacent).
    """
    n = len(ring_atoms_in_order)
    pos_to_idx = {a: i for i, a in enumerate(ring_atoms_in_order)}

    e1 = sorted(pos_to_idx[a] for a in edge_a if a in pos_to_idx)
    e2 = sorted(pos_to_idx[a] for a in edge_b if a in pos_to_idx)
    if len(e1) != 2 or len(e2) != 2:
        return False

    def edge_pos(positions):
        a, b = positions
        if abs(a - b) == 1:
            return min(a, b)
        if abs(a - b) == n - 1:
            return n - 1
        return None

    p1 = edge_pos(e1)
    p2 = edge_pos(e2)
    if p1 is None or p2 is None:
        return False
    d = abs(p1 - p2)
    d = min(d, n - d)
    if n == 6:
        return d == 3
    elif n == 5:
        return d == 2
    return False


def _check_topology(
    m: Chem.Mol, rings: List[Tuple[int, ...]]
) -> Tuple[str, Any]:
    """Validate that the scaffold ring system fits the earlier encoder's vocab.

    Returns ('ok', {'graph': nx.Graph, 'relations': dict}) or
            ('rejected', reason_string).
    """
    n = len(rings)
    for i, r in enumerate(rings):
        if len(r) not in (3, 4, 5, 6, 7):
            return ("rejected", f"ring_{i}_size_{len(r)}")
    if n > R_MAX:
        return ("rejected", f"too_many_rings_{n}_max_{R_MAX}")

    relations = {}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            rel, info = _classify_ring_pair(m, i, j, rings)
            relations[(i, j)] = (rel, info)
            if rel in ("spiro_invalid", "peri_invalid", "no_clean_endpoint"):
                return ("rejected", f"rings_{i}_{j}_{rel}")
            if rel in ("fused", "linked"):
                G.add_edge(i, j, relation=rel, info=info)

    if n > 1:
        if not nx.is_connected(G):
            return ("rejected", "ring_graph_disconnected")
        if not nx.is_tree(G):
            return ("rejected", "ring_graph_has_cycle_peri_fusion")

    # Linear-fusion check
    for ring_idx in range(n):
        fused_neighbors = []
        for nbr in G.neighbors(ring_idx):
            if G.edges[ring_idx, nbr]["relation"] == "fused":
                shared = set(G.edges[ring_idx, nbr]["info"]["shared_atoms"])
                fused_neighbors.append((nbr, shared))
        if len(fused_neighbors) < 2:
            continue
        if len(fused_neighbors) > 2:
            return ("rejected", f"ring_{ring_idx}_too_many_fused_neighbors")
        e1 = fused_neighbors[0][1]
        e2 = fused_neighbors[1][1]
        if e1 & e2:
            return ("rejected", f"ring_{ring_idx}_angular_fusion_shared_atom")
        if not _are_edges_opposite(list(rings[ring_idx]), e1, e2):
            return ("rejected", f"ring_{ring_idx}_angular_fusion")

    return ("ok", {"relations": relations, "graph": G})


def _canonical_ring_order(
    rings: List[Tuple[int, ...]], G: nx.Graph
) -> List[int]:
    """Order rings canonically for layout (deterministic).

    Strategy: pick the ring with the smallest atom-index minimum as the
    anchor (ring 0). Then BFS from there; ties broken by ring index.
    For chains of fused/linked rings, this gives a stable ordering.
    """
    n = len(rings)
    if n == 0:
        return []
    if n == 1:
        return [0]
    # Pick anchor: ring with smallest atom-index minimum
    anchor = min(range(n), key=lambda i: (min(rings[i]), i))
    # BFS
    order = [anchor]
    visited = {anchor}
    queue = deque([anchor])
    while queue:
        cur = queue.popleft()
        # Sort neighbors by ring index (deterministic)
        nbrs = sorted(G.neighbors(cur))
        for nbr in nbrs:
            if nbr not in visited:
                visited.add(nbr)
                order.append(nbr)
                queue.append(nbr)
    return order


def _canonical_traversal_first_ring(
    m: Chem.Mol, ring_atoms: Tuple[int, ...], next_ring_relation: Optional[Dict],
) -> List[int]:
    """Compute a canonical cyclic traversal of the FIRST ring.

    The decoder expects the first ring's atoms in a specific cyclic order.
    The far edge (where the next ring will fuse/link) must end up at
    positions (sz//2 - 1 + sz%2, sz//2) of the traversal.

    Strategy:
      - If next_ring_relation is None (no other rings): use atom order
        starting from the smallest atom index, going in the direction
        that minimizes the next atom index.
      - If next_ring_relation is fused: identify the 2 shared atoms;
        rotate/reverse the traversal so they end up at the canonical
        far-edge positions (i.e., positions sz-1 and sz-2 are shared,
        with shared_b at sz-1 and shared_a at sz-2... wait this is a bit
        different from how the decoder builds).

    Actually let me think about this more carefully. Looking at the
    decoder's build_ring_atom_indices for ring 0:
      ring_atoms[0] = list(range(sz0))  # i.e., [0, 1, 2, 3, 4, 5] for size 6
      far_edges[0] = _far_edge_position(sz0)  # = (2, 3) for size 6
    So ring 0's atoms are at positions 0..5 in traversal, and the far edge
    is between positions 2 and 3 — meaning atoms at *indices* 2 and 3 of
    the traversal are the "far edge atoms."

    For naphthalene's ring 0 in the decoder, the fusion atoms are 2 and 3
    (i.e., the 3rd and 4th atoms in the traversal). So when we extract
    naphthalene from RDKit, the canonical traversal of ring 0 must put
    the fusion-shared atoms at positions 2 and 3.

    For ring 0 with NO fused/linked next ring: any traversal works
    (we'll later attach pendants based on whatever traversal we picked).

    For ring 0 with a fused next ring: we need to find a traversal where
    the shared atoms are at positions 2 and 3.

    For ring 0 with a linked next ring: the linker attachment atom on
    ring 0 must be at position 2 (the first far-edge position).
    """
    sz = len(ring_atoms)
    far_a, far_b = (sz // 2) - 1 + (sz % 2), sz // 2
    # far_a = 2, far_b = 3 for sz=6
    # far_a = 2, far_b = 3 for sz=5

    # Compute all valid cyclic traversals: for each starting atom and
    # each direction (forward, reverse), produce the cyclic order.
    # The ring is given as a tuple from RDKit which already has a cyclic
    # adjacency. Build adjacency from the bonds.
    adj = {a: [] for a in ring_atoms}
    ring_set = set(ring_atoms)
    for a in ring_atoms:
        atom = m.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            n_idx = nbr.GetIdx()
            if n_idx in ring_set:
                adj[a].append(n_idx)

    # For each starting atom, try both directions
    candidates = []
    for start in ring_atoms:
        if len(adj[start]) < 2:
            continue
        for first_step in adj[start]:
            traversal = [start, first_step]
            current = first_step
            prev = start
            while len(traversal) < sz:
                next_options = [x for x in adj[current] if x != prev]
                if len(next_options) != 1:
                    break
                next_atom = next_options[0]
                traversal.append(next_atom)
                prev = current
                current = next_atom
            if len(traversal) == sz and set(traversal) == ring_set:
                candidates.append(tuple(traversal))

    # Filter candidates by next_ring_relation requirement
    if next_ring_relation is None:
        # Pick the traversal that starts with the smallest atom index
        # and goes in the direction with the smallest second-atom
        candidates.sort()
        return list(candidates[0]) if candidates else list(ring_atoms)

    rel = next_ring_relation["relation"]
    info = next_ring_relation["info"]

    valid = []
    if rel == "fused":
        shared = set(info["shared_atoms"])
        # Need shared atoms at traversal positions far_a and far_b
        for tv in candidates:
            if {tv[far_a], tv[far_b]} == shared:
                valid.append(tv)
    elif rel == "linked":
        # The linker attaches to ring 0 at attach_a; attach_a should be
        # at traversal position far_a
        attach_a = info["attach_a"]
        for tv in candidates:
            if tv[far_a] == attach_a:
                valid.append(tv)

    if not valid:
        # Fallback: pick any
        return list(candidates[0]) if candidates else list(ring_atoms)

    # Tiebreak: smallest first atom, then second
    valid.sort()
    return list(valid[0])


def _canonical_traversal_subsequent_ring(
    m: Chem.Mol,
    ring_atoms: Tuple[int, ...],
    relation_to_anchor: Dict,
    next_ring_relation: Optional[Dict],
) -> List[int]:
    """Compute canonical traversal for ring k>0 given its relation to the anchor.

    For fused: ring k's first atom (position 0) is shared atom 'shared_b';
    last atom (position sz-1) is shared atom 'shared_a'. Decoder uses:
      ring_atoms[k] = [shared_b] + new_atoms + [shared_a]
    So traversal[0] = shared_b, traversal[sz-1] = shared_a, and the
    intermediate atoms are the non-shared atoms in cyclic order.

    For linked: ring k's first atom (position 0) is attach_b (the atom
    connected to the linker).
    """
    sz = len(ring_atoms)
    rel = relation_to_anchor["relation"]
    info = relation_to_anchor["info"]
    ring_set = set(ring_atoms)
    adj = {a: [] for a in ring_atoms}
    for a in ring_atoms:
        for nbr in m.GetAtomWithIdx(a).GetNeighbors():
            n_idx = nbr.GetIdx()
            if n_idx in ring_set:
                adj[a].append(n_idx)

    # Generate all cyclic traversals
    candidates = []
    for start in ring_atoms:
        if len(adj[start]) < 2:
            continue
        for first_step in adj[start]:
            traversal = [start, first_step]
            current = first_step
            prev = start
            while len(traversal) < sz:
                next_options = [x for x in adj[current] if x != prev]
                if len(next_options) != 1:
                    break
                next_atom = next_options[0]
                traversal.append(next_atom)
                prev = current
                current = next_atom
            if len(traversal) == sz and set(traversal) == ring_set:
                candidates.append(tuple(traversal))

    valid = []
    if rel == "fused":
        shared = set(info["shared_atoms"])
        # traversal[0] in shared, traversal[-1] in shared, and
        # traversal[0] != traversal[-1]
        for tv in candidates:
            if (tv[0] in shared and tv[-1] in shared and tv[0] != tv[-1]
                    and m.GetBondBetweenAtoms(tv[0], tv[-1]) is not None):
                valid.append(tv)
    elif rel == "linked":
        attach_b = info["attach_b"]
        for tv in candidates:
            if tv[0] == attach_b:
                valid.append(tv)

    # If next_ring_relation is given, further filter so that ring k's
    # OWN far edge matches that relation
    if next_ring_relation is not None and valid:
        far_a, far_b = (sz // 2) - 1 + (sz % 2), sz // 2
        next_rel = next_ring_relation["relation"]
        next_info = next_ring_relation["info"]
        constrained = []
        if next_rel == "fused":
            next_shared = set(next_info["shared_atoms"])
            for tv in valid:
                if {tv[far_a], tv[far_b]} == next_shared:
                    constrained.append(tv)
        elif next_rel == "linked":
            attach_a = next_info["attach_a"]
            for tv in valid:
                if tv[far_a] == attach_a:
                    constrained.append(tv)
        if constrained:
            valid = constrained

    if not valid:
        # Fallback: pick any (caller may reject the molecule)
        return list(candidates[0]) if candidates else list(ring_atoms)

    valid.sort()
    return list(valid[0])


def _ring_type_id(ring_size_n: int, is_aromatic: bool) -> int:
    """Map (size, aromatic) to a ring-type vocabulary ID."""
    if   ring_size_n == 6 and is_aromatic:     return RING_6_AROM
    elif ring_size_n == 6 and not is_aromatic: return RING_6_ALIPH
    elif ring_size_n == 5 and is_aromatic:     return RING_5_AROM
    elif ring_size_n == 5 and not is_aromatic: return RING_5_ALIPH
    elif ring_size_n == 3 and is_aromatic:     return RING_3_AROM
    elif ring_size_n == 3 and not is_aromatic: return RING_3_ALIPH
    elif ring_size_n == 4 and is_aromatic:     return RING_4_AROM
    elif ring_size_n == 4 and not is_aromatic: return RING_4_ALIPH
    elif ring_size_n == 7 and is_aromatic:     return RING_7_AROM
    elif ring_size_n == 7 and not is_aromatic: return RING_7_ALIPH
    raise ValueError(f"Unsupported ring: size={ring_size_n}, arom={is_aromatic}")


def extract_layout_baseline(
    smi: str,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Extract (R, F, L, P_len, P_pos, atom_ids) from a SMILES.

    Performs terminal stripping using the earlier SMARTS list (extended in the earlier encoder
    with =O/=NH/=S double-bond terminals). Atoms that belong to detected
    terminals are excluded from the scaffold, so they don't appear in the
    final layout — Stage 2 will graft them back at sample time.

    Returns (label_dict, None) on success, or (None, reject_reason).
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, "smiles_parse_failed"

    # Detect terminals (parent matches only after subset removal).
    # Each detection is a dict with 'name', 'atom_indices', 'host_atom_idx',
    # 'attach_bond_class', etc.
    terminals = detect_terminals_in_molecule(
        m, CURATED_TERMINALS, _COMPILED_TERMINAL_PATTERNS,
    )

    # Build the set of atoms claimed by any terminal. These are excluded
    # from the scaffold.
    terminal_atom_set: Set[int] = set()
    for t in terminals:
        terminal_atom_set.update(t["atom_indices"])

    # Find rings of the FULL molecule, then filter to only those rings
    # whose atoms are entirely in the scaffold. (A ring atom can never be
    # in a terminal because terminals require exactly 1 crossing bond,
    # while a ring atom in a ring of size >= 3 has at least 2 ring-internal
    # bonds. So this filter is mostly a sanity check.)
    all_rings = list(m.GetRingInfo().AtomRings())
    rings = [r for r in all_rings if not any(a in terminal_atom_set for a in r)]

    if len(rings) == 0:
        if len(all_rings) == 0:
            return None, "no_rings"
        else:
            return None, "all_rings_in_terminals"

    # Topology check (operates on the scaffold rings only)
    status, info = _check_topology(m, rings)
    if status == "rejected":
        return None, info

    G: nx.Graph = info["graph"]
    relations = info["relations"]

    # Canonical ring order
    ring_order = _canonical_ring_order(rings, G)
    n_rings = len(ring_order)

    # Build R, F, L tensors
    R = np.zeros(R_MAX, dtype=np.int64)
    F = np.zeros((R_MAX, R_MAX), dtype=np.int64)
    L = np.zeros((R_MAX, R_MAX), dtype=np.int64)

    for new_k, orig_k in enumerate(ring_order):
        # Determine aromaticity
        ring = rings[orig_k]
        sz = len(ring)
        is_arom = all(m.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
        try:
            R[new_k] = _ring_type_id(sz, is_arom)
        except ValueError:
            return None, f"ring_{new_k}_unsupported"

    # Determine fusion / linker relationships using new ordering
    for new_k in range(1, n_rings):
        orig_k = ring_order[new_k]
        # Find the anchor (parent in BFS tree)
        anchor_new = None
        for new_p in range(new_k):
            orig_p = ring_order[new_p]
            edge_key = (min(orig_p, orig_k), max(orig_p, orig_k))
            rel, rel_info = relations[edge_key]
            if rel in ("fused", "linked") and G.has_edge(orig_p, orig_k):
                # This is a candidate parent
                anchor_new = new_p
                relation_str = rel
                relation_info = rel_info
                break
        if anchor_new is None:
            return None, f"ring_{new_k}_no_parent"

        F[anchor_new, new_k] = (
            F_FUSED if relation_str == "fused" else F_LINKED
        )
        F[new_k, anchor_new] = F[anchor_new, new_k]
        if relation_str == "linked":
            link_len = relation_info["linker_length"]
            if link_len > L_MAX:
                return None, f"linker_{anchor_new}_{new_k}_too_long_{link_len}"
            L[anchor_new, new_k] = link_len
            L[new_k, anchor_new] = link_len

    # Compute canonical traversals for each ring in the new ordering
    ring_traversals: Dict[int, List[int]] = {}

    # First ring: special case (no anchor)
    first_ring_idx = 0
    orig_first = ring_order[first_ring_idx]
    # If a second ring exists, its relation to ring 0 dictates the far-edge constraint
    next_rel = None
    if n_rings > 1:
        # Find the ring that is anchored at ring 0 in the new ordering
        for new_k in range(1, n_rings):
            if F[0, new_k] != F_NONE:
                orig_k = ring_order[new_k]
                edge_key = (min(orig_first, orig_k), max(orig_first, orig_k))
                rel, rel_info = relations[edge_key]
                next_rel = {"relation": rel, "info": rel_info}
                break
    try:
        tv0 = _canonical_traversal_first_ring(
            m, rings[orig_first], next_rel
        )
    except Exception as e:
        return None, f"ring_0_traversal_failed_{e}"
    ring_traversals[0] = tv0

    # Subsequent rings: use relation_to_anchor
    for new_k in range(1, n_rings):
        # Find anchor
        anchor_new = None
        for new_p in range(new_k):
            if F[new_p, new_k] != F_NONE:
                anchor_new = new_p
                break
        if anchor_new is None:
            return None, f"ring_{new_k}_no_parent_traversal"
        orig_k = ring_order[new_k]
        orig_anchor = ring_order[anchor_new]
        edge_key = (min(orig_anchor, orig_k), max(orig_anchor, orig_k))
        rel, rel_info = relations[edge_key]

        # Translate attach atoms / shared atoms into the ANCHOR's
        # traversal frame so the decoder can match atom indices.
        # For now we just use the original atom indices; the decoder
        # validates independently.

        # Determine if this ring has a downstream child to constrain its far edge
        next_rel_for_this = None
        for new_kk in range(new_k + 1, n_rings):
            if F[new_k, new_kk] != F_NONE:
                orig_kk = ring_order[new_kk]
                edge_key2 = (min(orig_k, orig_kk), max(orig_k, orig_kk))
                rel2, info2 = relations[edge_key2]
                next_rel_for_this = {"relation": rel2, "info": info2}
                break

        try:
            tv = _canonical_traversal_subsequent_ring(
                m,
                rings[orig_k],
                {"relation": rel, "info": rel_info},
                next_rel_for_this,
            )
        except Exception as e:
            return None, f"ring_{new_k}_traversal_failed_{e}"
        ring_traversals[new_k] = tv

    # Compute scaffold atom set (atoms in any ring of the ring_traversals)
    scaffold_ring_atoms = set()
    for tv in ring_traversals.values():
        scaffold_ring_atoms.update(tv)

    # Compute linker atoms (non-ring atoms on linker paths).
    # The classifier already provides these in info['linker_atoms'].
    linker_atoms_per_pair: Dict[Tuple[int, int], List[int]] = {}
    for new_k in range(1, n_rings):
        anchor_new = None
        for new_p in range(new_k):
            if F[new_p, new_k] == F_LINKED:
                anchor_new = new_p
                break
        if anchor_new is None:
            continue
        orig_k = ring_order[new_k]
        orig_anchor = ring_order[anchor_new]
        edge_key = (min(orig_anchor, orig_k), max(orig_anchor, orig_k))
        _, rel_info = relations[edge_key]
        # The classifier returns linker_atoms as a list; we may need to
        # reverse it depending on which direction the BFS went (we stored
        # path FROM ring_a_endpoint TO ring_b_endpoint).
        # We need atoms ordered from anchor (ring_anchor_new) toward the
        # new ring (new_k).
        # If orig_anchor < orig_k, the classifier's "ring_a" was orig_anchor
        # → linker_atoms is already in the right order. Otherwise reversed.
        link_atoms_raw = list(rel_info.get("linker_atoms", []))
        # In _classify_ring_pair we always passed (ring_a_idx, ring_b_idx)
        # with i < j. So linker_atoms goes from ring i (smaller index) to ring j.
        # Here we want it from anchor (orig_anchor) to child (orig_k).
        if orig_anchor > orig_k:
            link_atoms_raw = list(reversed(link_atoms_raw))
        linker_atoms_per_pair[(anchor_new, new_k)] = link_atoms_raw

    # Identify pendant atoms: non-ring scaffold atoms that are NOT linker
    # atoms and NOT terminal atoms.
    all_linker_atoms: Set[int] = set()
    for atoms in linker_atoms_per_pair.values():
        all_linker_atoms.update(atoms)

    # Pendants: walk outward from each ring atom; collect non-ring,
    # non-linker, non-terminal, non-already-seen atoms.
    P_len = np.zeros((R_MAX, P_MAX), dtype=np.int64)
    P_pos = np.zeros((R_MAX, P_MAX), dtype=np.int64)

    pendant_atom_ids: Dict[Tuple[int, int], List[int]] = {}

    for new_k in range(n_rings):
        tv = ring_traversals[new_k]
        for trav_pos, ring_atom in enumerate(tv):
            atom = m.GetAtomWithIdx(ring_atom)
            for nbr in atom.GetNeighbors():
                n_idx = nbr.GetIdx()
                if n_idx in scaffold_ring_atoms:
                    continue
                if n_idx in all_linker_atoms:
                    continue
                if n_idx in terminal_atom_set:
                    # This neighbor belongs to a detected terminal — skip
                    # the entire walk; the terminal is Stage 2's job.
                    continue
                # n_idx is a pendant chain start. Walk the pendant chain
                # outward, but excluding terminal atoms.
                pendant_chain = [n_idx]
                visited = {ring_atom, n_idx}
                cur = n_idx
                while True:
                    nbrs_cur = [
                        x.GetIdx() for x in m.GetAtomWithIdx(cur).GetNeighbors()
                        if x.GetIdx() not in visited
                        and x.GetIdx() not in scaffold_ring_atoms
                        and x.GetIdx() not in all_linker_atoms
                        and x.GetIdx() not in terminal_atom_set
                    ]
                    if len(nbrs_cur) == 0:
                        break
                    if len(nbrs_cur) > 1:
                        return None, f"pendant_branched_at_atom_{cur}"
                    nxt = nbrs_cur[0]
                    pendant_chain.append(nxt)
                    visited.add(nxt)
                    cur = nxt

                if len(pendant_chain) > P_LEN_MAX:
                    return None, f"pendant_too_long_{len(pendant_chain)}_max_{P_LEN_MAX}"

                slot = None
                for p in range(P_MAX):
                    if P_len[new_k, p] == 0:
                        slot = p
                        break
                if slot is None:
                    return None, f"ring_{new_k}_more_than_{P_MAX}_pendants"

                P_len[new_k, slot] = len(pendant_chain)
                P_pos[new_k, slot] = trav_pos
                pendant_atom_ids[(new_k, slot)] = pendant_chain

    # Validate that pendants don't collide with structural positions
    for new_k in range(n_rings):
        valid_pos = list_valid_pendant_positions(R, F, L, ring_idx=new_k)
        for p in range(P_MAX):
            if P_len[new_k, p] > 0:
                if int(P_pos[new_k, p]) not in valid_pos:
                    return None, (
                        f"pendant_ring_{new_k}_slot_{p}_pos_{P_pos[new_k, p]}"
                        f"_invalid_valid={valid_pos}"
                    )

    # Build atom_ids in the canonical numbering produced by the decoder
    # The decoder's numbering is determined by:
    #   ring_atoms[0] = list(range(sz0))
    #   ring_atoms[k>0]: shared atoms for fused, linker atoms then ring atoms for linked
    #   then pendants in order of (ring, slot)
    # We need to produce atom_ids[i] for i in 0..M_total-1.
    # We've collected all the atoms; now translate from RDKit's atom indices
    # to the canonical decoder positions.

    # Build mapping: canonical_position → RDKit atom_idx
    canonical_to_rdkit: Dict[int, int] = {}
    next_canonical = 0

    # Ring 0
    tv0 = ring_traversals[0]
    for i, ra in enumerate(tv0):
        canonical_to_rdkit[next_canonical] = ra
        next_canonical += 1

    # Subsequent rings + linkers
    for new_k in range(1, n_rings):
        anchor_new = None
        for new_p in range(new_k):
            if F[new_p, new_k] != F_NONE:
                anchor_new = new_p
                break
        relation = int(F[anchor_new, new_k])
        tv_k = ring_traversals[new_k]
        if relation == F_FUSED:
            # ring k traversal: [shared_b, n0, ..., n_{sz-3}, shared_a]
            # shared_b is at position 0, shared_a at position sz-1
            # New atoms are positions 1..sz-2
            for i in range(1, len(tv_k) - 1):
                canonical_to_rdkit[next_canonical] = tv_k[i]
                next_canonical += 1
        elif relation == F_LINKED:
            # Linker atoms first
            link_atoms = linker_atoms_per_pair.get((anchor_new, new_k), [])
            for la in link_atoms:
                canonical_to_rdkit[next_canonical] = la
                next_canonical += 1
            # Then full ring k atoms in traversal order
            for ra in tv_k:
                canonical_to_rdkit[next_canonical] = ra
                next_canonical += 1

    # Pendants in order (ring, slot)
    for new_k in range(n_rings):
        for slot in range(P_MAX):
            if P_len[new_k, slot] == 0:
                continue
            chain = pendant_atom_ids[(new_k, slot)]
            for ca in chain:
                canonical_to_rdkit[next_canonical] = ca
                next_canonical += 1

    M_total = next_canonical

    # Build atom_ids using the earlier version vocab
    atom_ids = np.zeros(M_total, dtype=np.int64)
    for canon_pos, rdkit_idx in canonical_to_rdkit.items():
        atom = m.GetAtomWithIdx(rdkit_idx)
        vid = _rdkit_atom_to_vocab_id(atom)
        if vid is None:
            return None, f"atom_{rdkit_idx}_not_in_vocab_{atom.GetSymbol()}"
        atom_ids[canon_pos] = vid

    # Round-trip validation
    try:
        aip, bc, am = decode_scaffold_baseline(R, F, L, P_len, P_pos, atom_ids)
    except ValueError as e:
        return None, f"decoder_roundtrip_failed_{e}"

    # Build a canonical SMILES of the original molecule for storage
    can_smi = Chem.MolToSmiles(m)
    # Build scaffold SMILES (just the scaffold part)
    rwm = Chem.RWMol(m)
    scaffold_atoms_set = set(canonical_to_rdkit.values())
    to_remove = [a.GetIdx() for a in m.GetAtoms() if a.GetIdx() not in scaffold_atoms_set]
    to_remove.sort(reverse=True)
    for a_idx in to_remove:
        rwm.RemoveAtom(a_idx)
    try:
        scaffold_smi = Chem.MolToSmiles(rwm)
    except Exception:
        scaffold_smi = None

    # Build a serializable list of terminal detections for the label.
    # A1+A2 don't use this, but it lets downstream Stage 2 work without
    # re-detecting terminals from the SMILES.
    #
    # B5 addition: we also record `host_canonical_idx` — the canonical
    # scaffold slot (0..M_total-1) that each terminal attaches to. The
    # raw `host_atom_idx` is in original RDKit numbering; B5 needs
    # canonical scaffold slots to populate per-atom site_fragment_ids.
    rdkit_to_canonical: Dict[int, int] = {
        rdkit_idx: canon_pos
        for canon_pos, rdkit_idx in canonical_to_rdkit.items()
    }
    terminals_serializable = [
        {
            "name": t["name"],
            "atom_indices": list(t["atom_indices"]),
            "host_atom_idx": t["host_atom_idx"],
            "host_canonical_idx": rdkit_to_canonical.get(
                int(t["host_atom_idx"]), -1
            ),
            "anchor_atom_idx": t["anchor_atom_idx"],
            "attach_bond_class": t["attach_bond_class"],
            "host_is_aromatic": t["host_is_aromatic"],
        }
        for t in terminals
    ]

    return (
        {
            "smi": can_smi,
            "scaffold_smi": scaffold_smi,
            "R": R,
            "F": F,
            "L": L,
            "P_len": P_len,
            "P_pos": P_pos,
            "atom_ids": atom_ids,
            "M_total": M_total,
            "terminals": terminals_serializable,
        },
        None,
    )


# ─── Driver ──────────────────────────────────────────────────────────


def extract_dataset_baseline(
    csv_path: str,
    output_pkl: str,
    smiles_col: str = "smiles",
    cond_cols: Tuple[str, ...] = ("solubility_water_norm", "homo_lumo_gap_norm"),
    limit: Optional[int] = None,
    verbose: bool = True,
    collect_rejection_examples: int = 0,
) -> Dict[str, Any]:
    """Extract layout labels from a RedDB-format CSV.

    Returns a stats dict with retention numbers and rejection breakdown.

    If collect_rejection_examples > 0, also returns up to that many
    example SMILES for each rejection reason (useful for debugging
    why specific molecules fail).
    """
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    n_total = len(df)
    labels: List[Dict] = []
    rejection_counts: Dict[str, int] = defaultdict(int)
    rejection_examples: Dict[str, List[str]] = defaultdict(list)
    n_kept = 0

    for idx, row in df.iterrows():
        smi = row[smiles_col]
        if not isinstance(smi, str):
            rejection_counts["smiles_not_string"] += 1
            continue
        try:
            cond_vals = []
            cond_ok = True
            for c in cond_cols:
                if c not in row or pd.isna(row[c]):
                    cond_ok = False
                    break
                cond_vals.append(float(row[c]))
            if not cond_ok:
                rejection_counts["missing_condition"] += 1
                continue
            condition = np.array(cond_vals, dtype=np.float32)

            label, reason = extract_layout_baseline(smi)
            if label is None:
                rejection_counts[reason] += 1
                if (collect_rejection_examples > 0
                        and len(rejection_examples[reason])
                        < collect_rejection_examples):
                    rejection_examples[reason].append(smi)
                continue
            label["condition"] = condition
            labels.append(label)
            n_kept += 1
        except Exception as e:
            err_reason = f"unexpected_{type(e).__name__}"
            rejection_counts[err_reason] += 1
            if (collect_rejection_examples > 0
                    and len(rejection_examples[err_reason])
                    < collect_rejection_examples):
                rejection_examples[err_reason].append(smi)

    if verbose:
        print(f"Extracted {n_kept}/{n_total} molecules ({100.0 * n_kept / max(n_total, 1):.1f}%)")
        if rejection_counts:
            print("Rejection breakdown:")
            for reason, cnt in sorted(
                rejection_counts.items(), key=lambda x: -x[1]
            )[:25]:
                print(f"  {cnt:>6}  {reason}")

    if output_pkl:
        with open(output_pkl, "wb") as f:
            pickle.dump(labels, f, protocol=4)
        if verbose:
            print(f"Wrote {output_pkl}")

    return {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_rejected": n_total - n_kept,
        "rejection_counts": dict(rejection_counts),
        "rejection_examples": dict(rejection_examples),
        "labels": labels,
    }


def print_rejection_examples(
    stats: Dict[str, Any], n_examples_per_bucket: int = 5
) -> None:
    """Pretty-print example SMILES for each rejection reason.

    Call after extract_dataset_baseline(... collect_rejection_examples=N) to see
    what the rejected molecules actually look like — useful for deciding
    whether each rejection reason is a real bug vs. legitimate filter.
    """
    examples = stats.get("rejection_examples", {})
    counts = stats.get("rejection_counts", {})
    if not examples:
        print("No rejection examples collected. Re-run with "
              "collect_rejection_examples > 0.")
        return
    print(f"\nRejection examples (up to {n_examples_per_bucket} per bucket):")
    for reason in sorted(counts.keys(), key=lambda r: -counts[r]):
        cnt = counts[reason]
        ex_list = examples.get(reason, [])
        print(f"\n  [{cnt:>6}] {reason}")
        for ex in ex_list[:n_examples_per_bucket]:
            print(f"    {ex}")


# ────────────────────────────────────────────────────────────────────────
# unified encoder: extract_layout
# ────────────────────────────────────────────────────────────────────────
#
# Drop-in sibling of extract_layout_baseline() with two architectural extensions:
#   1) BRANCHED pendants (tree-structured side chains, not just linear chains)
#   2) SPIRO ring junctions (one shared atom at sp3-quaternary centers,
#      F_SPIRO=3 in the F-matrix)
#
# Production-validated on full ZINC250K and 60,346-row failure CSV:
#   the earlier encoder baseline:         83.14% retention
#   unified encoder:  93.25% retention  (+10.11 pp, 0 regressions)
#
# Label schema additions over the earlier encoder:
#   B_size              (R_MAX, P_MAX)            — atoms per branch slot
#   B_pos               (R_MAX, P_MAX)            — ring-position of branch root
#   B_parent            (R_MAX, P_MAX, B_LEN_MAX) — parent atom idx within branch tree
#                                                    (0 = ring atom, k = atom k of branch)
#   B_bond              (R_MAX, P_MAX, B_LEN_MAX) — bond class atom→parent
#   spiro_atom_positions (R_MAX, R_MAX)           — canon-traversal position of spiro atom
#                                                    (-1 if no spiro at this (k, neighbor))
#   F[i,j] may contain F_SPIRO=3 for spiro-joined ring pairs
#
# Legacy linear-pendant fields (P_len, P_pos) are NOT emitted by current —
# the branch-tree representation supersedes them. For linear pendants,
# B_size==1 is the natural single-atom branch.
#
# This function is the source of truth for preprocessing.
# Imports are local to allow this file to be imported even when some
# helpers are still being patched in.

from collections import deque as _deque
from rdkit.Chem import HybridizationType as _HybridizationType
from himodit.chem.decoder import (
    F_NONE as _F_NONE,
    F_FUSED as _F_FUSED,
    F_LINKED as _F_LINKED,
    F_SPIRO as _F_SPIRO,
    B_LEN_MAX as _B_LEN_MAX,
)

# capacities
# Capacity constants come from the decoder so encoder and decoder cannot
# drift apart; a mismatch would produce labels the decoder cannot read.
from himodit.chem.decoder import P_MAX_BRANCH  # noqa: E402

B_LEN_MAX = _B_LEN_MAX

# Bond class codes (same as the earlier encoder build_bond_classes output)
_BOND_NONE     = 0
_BOND_SINGLE   = 1
_BOND_AROMATIC = 2
_BOND_DOUBLE   = 3
_BOND_TRIPLE   = 4


def _rdkit_bond_class(bond):
    from rdkit.Chem import BondType as _BT
    bt = bond.GetBondType()
    if bond.GetIsAromatic():
        return _BOND_AROMATIC
    return {
        _BT.SINGLE:   _BOND_SINGLE,
        _BT.AROMATIC: _BOND_AROMATIC,
        _BT.DOUBLE:   _BOND_DOUBLE,
        _BT.TRIPLE:   _BOND_TRIPLE,
    }.get(bt)


def _is_valid_spiro_center(atom):
    """Test whether an atom is a chemically-valid spiro center.

    Valid spiro: sp3, non-aromatic, AND one of:
      - sp3-quaternary C (4 bonds, charge 0, no Hs)
      - sp3-quaternary N+ (4 bonds, charge +1)
      - sp3-quaternary Si (4 bonds, charge 0)
    """
    if atom.GetIsAromatic():
        return False, "aromatic"
    if atom.GetHybridization() != _HybridizationType.SP3:
        return False, f"hyb_{atom.GetHybridization()}"
    sym = atom.GetSymbol()
    chg = atom.GetFormalCharge()
    nH = atom.GetTotalNumHs()
    deg = atom.GetDegree()
    if sym == "C" and chg == 0 and nH == 0 and deg == 4: return True, "sp3-quat-C"
    if sym == "N" and chg == 1 and deg == 4:             return True, "sp3-quat-N+"
    if sym == "Si" and chg == 0 and deg == 4:            return True, "sp3-quat-Si"
    return False, f"sym={sym},ch={chg},nH={nH},deg={deg}"


def _classify_ring_pair_branched(m, ring_a_idx, ring_b_idx, all_rings):
    """ring-pair classifier: accepts spiro at valid centers, delegates rest to the earlier encoder."""
    ring_a = all_rings[ring_a_idx]
    ring_b = all_rings[ring_b_idx]
    shared = set(ring_a) & set(ring_b)
    if len(shared) == 1:
        atom_idx = next(iter(shared))
        atom = m.GetAtomWithIdx(atom_idx)
        is_valid, reason = _is_valid_spiro_center(atom)
        if is_valid:
            return ("spiro", {"shared_atom": atom_idx, "spiro_kind": reason})
        return ("spiro_invalid", {"shared_atoms": sorted(shared), "rejection_reason": reason})
    return _classify_ring_pair(m, ring_a_idx, ring_b_idx, all_rings)


def _check_topology_branched(m, rings):
    """topology check: spiro counts as ring-graph edge (chains allowed)."""
    n = len(rings)
    for i, r in enumerate(rings):
        if len(r) not in (3, 4, 5, 6, 7):
            return ("rejected", f"ring_{i}_size_{len(r)}")
    if n > R_MAX:
        return ("rejected", f"too_many_rings_{n}_max_{R_MAX}")

    relations = {}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            rel, info = _classify_ring_pair_branched(m, i, j, rings)
            relations[(i, j)] = (rel, info)
            if rel in ("spiro_invalid", "peri_invalid", "no_clean_endpoint"):
                return ("rejected", f"rings_{i}_{j}_{rel}")
            if rel in ("fused", "linked", "spiro"):
                G.add_edge(i, j, relation=rel, info=info)

    if n > 1:
        if not nx.is_connected(G):
            return ("rejected", "ring_graph_disconnected")
        if not nx.is_tree(G):
            return ("rejected", "ring_graph_has_cycle_peri_fusion")

    # Linear-fusion check (unchanged from the earlier encoder, FUSED edges only)
    for ring_idx in range(n):
        fused_neighbors = []
        for nbr in G.neighbors(ring_idx):
            if G.edges[ring_idx, nbr]["relation"] == "fused":
                shared = set(G.edges[ring_idx, nbr]["info"]["shared_atoms"])
                fused_neighbors.append((nbr, shared))
        if len(fused_neighbors) < 2:
            continue
        if len(fused_neighbors) > 2:
            return ("rejected", f"ring_{ring_idx}_too_many_fused_neighbors")
        e1 = fused_neighbors[0][1]
        e2 = fused_neighbors[1][1]
        if e1 & e2:
            return ("rejected", f"ring_{ring_idx}_angular_fusion_shared_atom")
        if not _are_edges_opposite(list(rings[ring_idx]), e1, e2):
            return ("rejected", f"ring_{ring_idx}_angular_fusion")

    return ("ok", {"relations": relations, "graph": G})


def _extract_branch_trees(m, R_MAX_, P_MAX_, B_LEN_MAX_,
                                scaffold_ring_atoms, all_linker_atoms,
                                terminal_atom_set, ring_traversals, n_rings):
    """Build tree-shaped branch labels for each ring's pendant slots."""
    B_size   = np.zeros((R_MAX_, P_MAX_), dtype=np.int64)
    B_pos    = np.zeros((R_MAX_, P_MAX_), dtype=np.int64)
    B_parent = np.zeros((R_MAX_, P_MAX_, B_LEN_MAX_), dtype=np.int64)
    B_bond   = np.zeros((R_MAX_, P_MAX_, B_LEN_MAX_), dtype=np.int64)
    branch_atom_ids = {}
    not_a_candidate = scaffold_ring_atoms | all_linker_atoms | terminal_atom_set
    already_in_branch = set()

    for new_k in range(n_rings):
        tv = ring_traversals[new_k]
        slot = 0
        for trav_pos, ring_atom_rdkit in enumerate(tv):
            atom = m.GetAtomWithIdx(ring_atom_rdkit)
            for nbr in atom.GetNeighbors():
                n_idx = nbr.GetIdx()
                if n_idx in not_a_candidate: continue
                if n_idx in already_in_branch: continue

                tree_atoms, tree_parent, tree_bond = [], [], []
                queue = _deque([(n_idx, 0)])
                visited = {ring_atom_rdkit}
                while queue:
                    cur, parent_within_tree = queue.popleft()
                    if cur in visited: continue
                    visited.add(cur)
                    if parent_within_tree == 0:
                        parent_rdkit_atom = ring_atom_rdkit
                    else:
                        parent_rdkit_atom = tree_atoms[parent_within_tree - 1]
                    bond = m.GetBondBetweenAtoms(cur, parent_rdkit_atom)
                    if bond is None:
                        return None, f"branched_walk_missing_bond_{cur}_{parent_rdkit_atom}"
                    bcls = _rdkit_bond_class(bond)
                    if bcls is None:
                        return None, f"branched_walk_unknown_bond_type"
                    tree_atoms.append(cur)
                    tree_parent.append(parent_within_tree)
                    tree_bond.append(bcls)
                    this_idx_in_tree = len(tree_atoms)
                    if len(tree_atoms) > B_LEN_MAX_:
                        return None, f"branch_too_large_{len(tree_atoms)}_max_{B_LEN_MAX_}"
                    cur_atom = m.GetAtomWithIdx(cur)
                    for nb2 in cur_atom.GetNeighbors():
                        nb2_idx = nb2.GetIdx()
                        if nb2_idx in not_a_candidate: continue
                        if nb2_idx in visited: continue
                        queue.append((nb2_idx, this_idx_in_tree))

                if len(visited) - 1 != len(tree_atoms):
                    return None, f"branch_contains_internal_cycle_ring_{new_k}_pos_{trav_pos}"
                if slot >= P_MAX_:
                    return None, f"ring_{new_k}_more_than_{P_MAX_}_branches"
                B_size[new_k, slot] = len(tree_atoms)
                B_pos[new_k, slot] = trav_pos
                for i, (p, b) in enumerate(zip(tree_parent, tree_bond)):
                    B_parent[new_k, slot, i] = p
                    B_bond[new_k, slot, i] = b
                branch_atom_ids[(new_k, slot)] = tree_atoms[:]
                already_in_branch.update(tree_atoms)
                slot += 1

    return (B_size, B_pos, B_parent, B_bond, branch_atom_ids), None


def extract_layout(
    smi: str,
    P_MAX_param: int = P_MAX_BRANCH,
    B_LEN_MAX_param: int = B_LEN_MAX,
    strict: bool = False,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Encode a SMILES string into a hierarchical layout label.

    Handles branch-tree side chains and spiro junctions at
    sp3-quaternary centres. Retention on ZINC250K is 93.6%.

    Parameters
    ----------
    smi
        Input SMILES. Parsed and sanitized by RDKit.
    P_MAX_param
        Branch slots per ring. Must match the P_MAX_BRANCH the models
        were built with, or the label tensors will not fit.
    B_LEN_MAX_param
        Maximum atoms in one branch tree.
    strict
        Reject molecules whose atoms are not fully accounted for. Branch
        trees grow from ring atoms only, so a substituent attached to a
        *linker* atom belongs to no category and is silently dropped -
        the label then describes a molecule slightly smaller than the
        input. This affects 6.45% of otherwise-accepted ZINC250K
        molecules. `strict=True` rejects them with reason
        `atoms_unaccounted_N`, trading roughly 6 percentage points of
        retention for a guarantee that every label round-trips. Default
        False, which reproduces the published retention figure.

    Returns
    -------
    (label, None) on success, or (None, reason) on rejection.

    The label dict contains:
        smi, R, F, L, spiro_atom_positions, B_size, B_pos, B_parent,
        B_bond, atom_ids, M_total, terminals, n_spiro_junctions,
        n_branches
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, "smiles_parse_failed"

    # Terminal detection (same as the earlier encoder)
    terminals = detect_terminals_in_molecule(m, CURATED_TERMINALS, _COMPILED_TERMINAL_PATTERNS)
    terminal_atom_set = set()
    for t in terminals:
        terminal_atom_set.update(t["atom_indices"])

    all_rings = list(m.GetRingInfo().AtomRings())
    rings = [r for r in all_rings if not any(a in terminal_atom_set for a in r)]
    if len(rings) == 0:
        return (None, "no_rings") if len(all_rings) == 0 else (None, "all_rings_in_terminals")

    # Topology check WITH spiro
    status, info = _check_topology_branched(m, rings)
    if status == "rejected":
        return None, info

    G = info["graph"]
    relations = info["relations"]
    ring_order = _canonical_ring_order(rings, G)
    n_rings = len(ring_order)

    R = np.zeros(R_MAX, dtype=np.int64)
    F = np.zeros((R_MAX, R_MAX), dtype=np.int64)
    L = np.zeros((R_MAX, R_MAX), dtype=np.int64)
    spiro_atom_positions = np.full((R_MAX, R_MAX), -1, dtype=np.int64)

    for new_k, orig_k in enumerate(ring_order):
        ring = rings[orig_k]
        sz = len(ring)
        is_arom = all(m.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
        try:
            R[new_k] = _ring_type_id(sz, is_arom)
        except ValueError:
            return None, f"ring_{new_k}_unsupported"

    # F/L matrix (with F_SPIRO support)
    for new_k in range(1, n_rings):
        orig_k = ring_order[new_k]
        anchor_new = None
        relation_str = None
        relation_info = None
        for new_p in range(new_k):
            orig_p = ring_order[new_p]
            edge_key = (min(orig_p, orig_k), max(orig_p, orig_k))
            rel, rel_info = relations[edge_key]
            if rel in ("fused", "linked", "spiro") and G.has_edge(orig_p, orig_k):
                anchor_new = new_p
                relation_str = rel
                relation_info = rel_info
                break
        if anchor_new is None:
            return None, f"ring_{new_k}_no_parent"
        if relation_str == "fused":   F[anchor_new, new_k] = _F_FUSED
        elif relation_str == "linked": F[anchor_new, new_k] = _F_LINKED
        elif relation_str == "spiro":  F[anchor_new, new_k] = _F_SPIRO
        F[new_k, anchor_new] = F[anchor_new, new_k]
        if relation_str == "linked":
            link_len = relation_info["linker_length"]
            if link_len > L_MAX:
                return None, f"linker_{anchor_new}_{new_k}_too_long_{link_len}"
            L[anchor_new, new_k] = link_len
            L[new_k, anchor_new] = link_len

    # Ring traversals (with spiro special handling)
    ring_traversals = {}
    first_ring_idx = 0
    orig_first = ring_order[first_ring_idx]
    next_rel = None
    if n_rings > 1:
        for new_k in range(1, n_rings):
            if F[0, new_k] != _F_NONE:
                orig_k = ring_order[new_k]
                edge_key = (min(orig_first, orig_k), max(orig_first, orig_k))
                rel, rel_info = relations[edge_key]
                next_rel = {"relation": rel, "info": rel_info}
                break
    try:
        if next_rel is not None and next_rel["relation"] == "spiro":
            spiro_atom = next_rel["info"]["shared_atom"]
            synth_info = {"shared_atoms": [spiro_atom, spiro_atom]}
            tv0 = _canonical_traversal_first_ring(
                m, rings[orig_first], {"relation": "fused", "info": synth_info},
            )
        else:
            tv0 = _canonical_traversal_first_ring(m, rings[orig_first], next_rel)
    except Exception as e:
        return None, f"ring_0_traversal_failed_{e}"
    ring_traversals[0] = tv0

    for new_k in range(1, n_rings):
        anchor_new = None
        for new_p in range(new_k):
            if F[new_p, new_k] != _F_NONE:
                anchor_new = new_p
                break
        if anchor_new is None:
            return None, f"ring_{new_k}_no_parent_traversal"
        orig_k = ring_order[new_k]
        orig_anchor = ring_order[anchor_new]
        edge_key = (min(orig_anchor, orig_k), max(orig_anchor, orig_k))
        rel, rel_info = relations[edge_key]
        next_rel_for_this = None
        for new_kk in range(new_k + 1, n_rings):
            if F[new_k, new_kk] != _F_NONE:
                orig_kk = ring_order[new_kk]
                edge_key2 = (min(orig_k, orig_kk), max(orig_k, orig_kk))
                rel2, info2 = relations[edge_key2]
                next_rel_for_this = {"relation": rel2, "info": info2}
                break
        try:
            if rel == "spiro":
                spiro_atom = rel_info["shared_atom"]
                synth_info = {"shared_atoms": [spiro_atom, spiro_atom]}
                tv = _canonical_traversal_subsequent_ring(
                    m, rings[orig_k],
                    {"relation": "fused", "info": synth_info},
                    next_rel_for_this,
                )
            else:
                tv = _canonical_traversal_subsequent_ring(
                    m, rings[orig_k],
                    {"relation": rel, "info": rel_info},
                    next_rel_for_this,
                )
        except Exception as e:
            return None, f"ring_{new_k}_traversal_failed_{e}"
        ring_traversals[new_k] = tv

    # Record spiro atom positions
    for new_k in range(n_rings):
        for new_kk in range(n_rings):
            if F[new_k, new_kk] == _F_SPIRO:
                orig_k = ring_order[new_k]
                orig_kk = ring_order[new_kk]
                edge_key = (min(orig_k, orig_kk), max(orig_k, orig_kk))
                _, rel_info = relations[edge_key]
                spiro_atom = rel_info["shared_atom"]
                tv = ring_traversals[new_k]
                if spiro_atom in tv:
                    spiro_atom_positions[new_k, new_kk] = tv.index(spiro_atom)

    # Build atom-position sets
    scaffold_ring_atoms = set()
    for tv in ring_traversals.values():
        scaffold_ring_atoms.update(tv)

    all_linker_atoms = set()
    for new_k in range(1, n_rings):
        anchor_new = None
        for new_p in range(new_k):
            if F[new_p, new_k] == _F_LINKED:
                anchor_new = new_p
                break
        if anchor_new is None: continue
        orig_k = ring_order[new_k]
        orig_anchor = ring_order[anchor_new]
        edge_key = (min(orig_anchor, orig_k), max(orig_anchor, orig_k))
        _, rel_info = relations[edge_key]
        all_linker_atoms.update(rel_info.get("linker_atoms", []))

    # Branched pendants
    branch_result, branch_err = _extract_branch_trees(
        m, R_MAX, P_MAX_param, B_LEN_MAX_param,
        scaffold_ring_atoms, all_linker_atoms, terminal_atom_set,
        ring_traversals, n_rings,
    )
    if branch_err is not None:
        return None, branch_err
    B_size, B_pos, B_parent, B_bond, branch_atom_ids = branch_result

    # Canonical layout (the earlier encoder-style explicit, plus spiro handling)
    canonical_to_rdkit = {}
    next_canonical = 0
    # Ring 0 fully
    for ra in ring_traversals[0]:
        canonical_to_rdkit[next_canonical] = ra
        next_canonical += 1
    # Subsequent rings + linkers
    for new_k in range(1, n_rings):
        anchor_new = None
        for new_p in range(new_k):
            if F[new_p, new_k] != _F_NONE:
                anchor_new = new_p
                break
        if anchor_new is None:
            return None, f"ring_{new_k}_no_canonical_anchor"
        relation = int(F[anchor_new, new_k])
        tv_k = ring_traversals[new_k]
        if relation == _F_FUSED:
            for i in range(1, len(tv_k) - 1):
                canonical_to_rdkit[next_canonical] = tv_k[i]
                next_canonical += 1
        elif relation == _F_LINKED:
            orig_k_loc = ring_order[new_k]
            orig_anchor_loc = ring_order[anchor_new]
            edge_key = (min(orig_anchor_loc, orig_k_loc), max(orig_anchor_loc, orig_k_loc))
            _, rel_info_loc = relations[edge_key]
            link_atoms = list(rel_info_loc.get("linker_atoms", []))
            if orig_anchor_loc > orig_k_loc:
                link_atoms = list(reversed(link_atoms))
            for la in link_atoms:
                canonical_to_rdkit[next_canonical] = la
                next_canonical += 1
            for ra in tv_k:
                canonical_to_rdkit[next_canonical] = ra
                next_canonical += 1
        elif relation == _F_SPIRO:
            for i in range(1, len(tv_k)):
                canonical_to_rdkit[next_canonical] = tv_k[i]
                next_canonical += 1
    # Pendant tree atoms
    for new_k in range(n_rings):
        for slot in range(P_MAX_param):
            if B_size[new_k, slot] == 0: continue
            for ra in branch_atom_ids[(new_k, slot)]:
                canonical_to_rdkit[next_canonical] = ra
                next_canonical += 1

    M_total = next_canonical
    if M_total > M_MAX:
        return None, f"M_total_{M_total}_exceeds_M_MAX_{M_MAX}"

    atom_ids = np.zeros(M_total, dtype=np.int64)
    for canon_pos, rdkit_idx in canonical_to_rdkit.items():
        atom = m.GetAtomWithIdx(rdkit_idx)
        vid = _rdkit_atom_to_vocab_id(atom)
        if vid is None:
            return None, f"atom_{rdkit_idx}_not_in_vocab_{atom.GetSymbol()}"
        atom_ids[canon_pos] = vid

    rdkit_to_canonical = {r: c for c, r in canonical_to_rdkit.items()}
    terminals_serializable = [
        {
            "name": t["name"],
            "atom_indices": list(t["atom_indices"]),
            "host_atom_idx": t["host_atom_idx"],
            "host_canonical_idx": rdkit_to_canonical.get(int(t["host_atom_idx"]), -1),
            "anchor_atom_idx": t["anchor_atom_idx"],
            "attach_bond_class": t["attach_bond_class"],
            "host_is_aromatic": t["host_is_aromatic"],
        }
        for t in terminals
    ]

    # Atom accounting. Every heavy atom should be either a scaffold atom
    # (counted in M_total) or part of a detected terminal. Anything left
    # over is an atom the encoder walked past without collecting - in
    # practice a substituent hanging off a linker atom, which the
    # ring-rooted branch walk never reaches. The label would then encode
    # a smaller molecule than the input, so under `strict` we reject
    # rather than silently truncate.
    n_unaccounted = m.GetNumAtoms() - (M_total + len(terminal_atom_set))
    if strict and n_unaccounted != 0:
        return None, f"atoms_unaccounted_{n_unaccounted}"

    return (
        {
            "smi": Chem.MolToSmiles(m),
            "R": R, "F": F, "L": L,
            "B_size": B_size, "B_pos": B_pos,
            "B_parent": B_parent, "B_bond": B_bond,
            "spiro_atom_positions": spiro_atom_positions,
            "atom_ids": atom_ids,
            "M_total": M_total,
            "terminals": terminals_serializable,
            "n_spiro_junctions": int((F == _F_SPIRO).sum() // 2),
            "n_branches": int((B_size > 0).sum()),
            "n_atoms_unaccounted": int(n_unaccounted),
        },
        None,
    )
