"""
Deterministic scaffold decoder.
===============================

Pure Python, no learnable parameters. Turns a hierarchical layout label
into the `(bond_classes, atom_mask)` tensors that A2 and the Terminal
stage consume, and that `himodit.chem.compose` turns into a molecule.

Because the bond matrix is constructed deterministically from the layout
rather than sampled, the model can never emit a self-contradictory bond
pattern: ring closure, fusion, and linker connectivity are guaranteed by
construction.

Layout state
------------
R : (R_MAX,) int
    Ring-type id per ring slot. 0 = PAD; see RING_TYPE_INFO for the
    (size, aromatic) pair each id denotes. Sizes 3-7 are supported.

F : (R_MAX, R_MAX) int, symmetric
    Ring-relation matrix.
      0 F_NONE    no relation
      1 F_FUSED   rings share an edge (2 atoms)
      2 F_LINKED  rings joined by a bond, optionally through a chain
      3 F_SPIRO   rings share exactly one sp3-quaternary atom

L : (R_MAX, R_MAX) int, symmetric
    Linker chain lengths. Meaningful only where F == F_LINKED.
    L[i, j] in [0, L_MAX] counts the atoms between the two rings.

spiro_atom_positions : (R_MAX, R_MAX) int
    Position of the shared atom in the anchor ring's canonical traversal,
    for pairs where F == F_SPIRO. -1 elsewhere (the NO_SPIRO sentinel).

B_size, B_pos : (R_MAX, P_MAX_BRANCH) int
    Per branch slot: number of atoms in the branch tree, and the ring
    position it is rooted at. B_size == 0 means the slot is empty.

B_parent, B_bond : (R_MAX, P_MAX_BRANCH, B_LEN_MAX) int
    Per branch atom i: the index of its parent (0 = the ring atom at
    B_pos, k >= 1 = atom k-1 of this branch) and the bond class joining
    them.

Atom numbering
--------------
Canonical and deterministic, matching the encoder exactly:

  1. Ring 0 contributes all of its atoms, in traversal order.
  2. Each subsequent ring k, in index order:
       fused   -> borrows 2 atoms from its anchor, appends size_k - 2
       linked  -> appends L[anchor, k] linker atoms, then all size_k
       spiro   -> borrows 1 atom from its anchor, appends size_k - 1
  3. Branch trees, in (ring, slot) order.

Bond classes
------------
Ring-internal bonds are aromatic if the ring type is aromatic, else
single. Linker bonds are single. Branch bonds carry whatever class the
encoder recorded in B_bond (single, aromatic, double, or triple).

Two decoders live here. `decode_scaffold` is the current one and takes
the branch-tree layout above. `decode_scaffold_baseline` takes the
earlier linear-pendant layout (P_len / P_pos) and is retained only so
the baseline retention figure in the README stays reproducible; new code
should not call it.
"""
from __future__ import annotations

from typing import Tuple, List, Optional, Dict, Set
import numpy as np


# ─── Constants ──────────────────────────────────────────────────────
M_MAX = 40       
R_MAX = 6        
# Branch slots per ring. This is the shape of the B_* label tensors, so
# the encoder, A3, and this decoder must all agree on it; they all read
# it from here.
P_MAX_BRANCH = 8

# Legacy linear-pendant slot count, used only by the baseline decoder.
# New code wants P_MAX_BRANCH.
P_MAX = 6
L_MAX = 10       
P_LEN_MAX = 10   

# Ring-type IDs
RING_PAD       = 0
# Ring vocabulary: PAD plus ten (size, aromaticity) combinations.
RING_6_AROM    = 1
RING_6_ALIPH   = 2
RING_5_AROM    = 3
RING_5_ALIPH   = 4
RING_3_AROM    = 5
RING_3_ALIPH   = 6
RING_4_AROM    = 7
RING_4_ALIPH   = 8
RING_7_AROM    = 9
RING_7_ALIPH   = 10

# F values
F_NONE   = 0
F_FUSED  = 1
F_LINKED = 2
F_SPIRO  = 3   # sp³-quaternary spiro center (one shared atom)

# capacity for branched pendants (replaces P_LEN_MAX as the limit-controlling
# constant for pendant trees). Linear-pendant code paths still use P_LEN_MAX so
# legacy the earlier encoder labels keep working.
# Maximum atoms in a single branch tree.
B_LEN_MAX = 15

RING_TYPE_INFO: Dict[int, Tuple[int, bool]] = {
    RING_PAD:     (0, False),
    RING_6_AROM:  (6, True),
    RING_6_ALIPH: (6, False),
    RING_5_AROM:  (5, True),
    RING_5_ALIPH: (5, False),
    RING_3_AROM:  (3, True),
    RING_3_ALIPH: (3, False),
    RING_4_AROM:  (4, True),
    RING_4_ALIPH: (4, False),
    RING_7_AROM:  (7, True),
    RING_7_ALIPH: (7, False),
}

# Bond class IDs (same as the earlier version / config.py)
BOND_NONE     = 0
BOND_SINGLE   = 1
BOND_AROMATIC = 2

# Atom vocab (same as the earlier version): {0:<PAD>, 1:c, 2:O, 3:C, 4:N, 5:n, 6:S, 7:F, 8:s, 9:o}
# Aromatic IDs include n+ (id 12) and n- (id 14).
AROMATIC_ATOM_IDS = {1, 5, 8, 9, 12, 14}
ATOM_PAD = 0


def ring_size(ring_type: int) -> int:
    if ring_type not in RING_TYPE_INFO:
        raise ValueError(f"Unknown ring_type: {ring_type}")
    return RING_TYPE_INFO[ring_type][0]


def ring_is_aromatic(ring_type: int) -> bool:
    if ring_type not in RING_TYPE_INFO:
        raise ValueError(f"Unknown ring_type: {ring_type}")
    return RING_TYPE_INFO[ring_type][1]


def _validate_layout(
    R: np.ndarray, F: np.ndarray, L: np.ndarray,
    P_len: np.ndarray, P_pos: np.ndarray,
) -> int:
    """Validate the layout structure and return the number of non-PAD rings."""
    R = np.asarray(R, dtype=np.int64)
    F = np.asarray(F, dtype=np.int64)
    L = np.asarray(L, dtype=np.int64)
    P_len = np.asarray(P_len, dtype=np.int64)
    P_pos = np.asarray(P_pos, dtype=np.int64)

    if R.shape != (R_MAX,):
        raise ValueError(f"R shape must be ({R_MAX},), got {R.shape}")
    if F.shape != (R_MAX, R_MAX):
        raise ValueError(f"F shape must be ({R_MAX},{R_MAX}), got {F.shape}")
    if L.shape != (R_MAX, R_MAX):
        raise ValueError(f"L shape must be ({R_MAX},{R_MAX}), got {L.shape}")
    if P_len.shape != (R_MAX, P_MAX):
        raise ValueError(f"P_len shape must be ({R_MAX},{P_MAX}), got {P_len.shape}")
    if P_pos.shape != (R_MAX, P_MAX):
        raise ValueError(f"P_pos shape must be ({R_MAX},{P_MAX}), got {P_pos.shape}")

    # Count non-PAD rings; must be left-packed
    n_rings = 0
    seen_pad = False
    for k in range(R_MAX):
        if int(R[k]) == RING_PAD:
            seen_pad = True
        else:
            if seen_pad:
                raise ValueError(
                    f"R must be left-packed: non-PAD ring at slot {k} "
                    f"after PAD slot. R={R.tolist()}"
                )
            n_rings += 1

    if n_rings == 0:
        return 0

    # F symmetry, zero diagonal, valid values
    for i in range(R_MAX):
        if F[i, i] != 0:
            raise ValueError(f"F diagonal must be 0; F[{i},{i}]={F[i,i]}")
        for j in range(R_MAX):
            if F[i, j] != F[j, i]:
                raise ValueError(
                    f"F must be symmetric; F[{i},{j}]={F[i,j]} vs "
                    f"F[{j},{i}]={F[j,i]}"
                )
            if F[i, j] not in (F_NONE, F_FUSED, F_LINKED, F_SPIRO):
                raise ValueError(
                    f"F[{i},{j}]={F[i,j]} not in {{0=none, 1=fused, 2=linked, 3=spiro}}"
                )

    for i in range(R_MAX):
        for j in range(R_MAX):
            if F[i, j] != F_NONE:
                if i >= n_rings or j >= n_rings:
                    raise ValueError(
                        f"F[{i},{j}]={F[i,j]} refers to PAD ring "
                        f"(n_rings={n_rings})"
                    )

    # L symmetry; L only meaningful where F=2 (linked)
    for i in range(R_MAX):
        for j in range(R_MAX):
            if L[i, j] != L[j, i]:
                raise ValueError(
                    f"L must be symmetric; L[{i},{j}]={L[i,j]} vs L[{j},{i}]"
                )
            if F[i, j] != F_LINKED and L[i, j] != 0:
                raise ValueError(
                    f"L[{i},{j}]={L[i,j]} but F[{i},{j}]={F[i,j]} != linked. "
                    "L only meaningful for linked pairs."
                )
            if L[i, j] < 0 or L[i, j] > L_MAX:
                raise ValueError(
                    f"L[{i},{j}]={L[i,j]} out of range [0, {L_MAX}]"
                )

    # Each ring k > 0 must have exactly one anchor
    for k in range(1, n_rings):
        anchors = [k_prev for k_prev in range(k) if F[k_prev, k] != F_NONE]
        if len(anchors) == 0:
            raise ValueError(
                f"Ring {k} has no anchor (no prior ring with F[*,{k}] != 0). "
                "Disconnected ring layouts not supported in the earlier encoder."
            )
        if len(anchors) > 1:
            raise ValueError(
                f"Ring {k} has multiple anchors {anchors}. the earlier encoder supports "
                "tree-shaped scaffolds (one anchor per ring), not cycles."
            )

    # P_len: validity + PAD enforcement
    for i in range(R_MAX):
        for p in range(P_MAX):
            if i >= n_rings and P_len[i, p] != 0:
                raise ValueError(
                    f"P_len[{i},{p}]={P_len[i,p]} but ring {i} is PAD"
                )
            if P_len[i, p] < 0 or P_len[i, p] > P_LEN_MAX:
                raise ValueError(
                    f"P_len[{i},{p}]={P_len[i,p]} out of range [0, {P_LEN_MAX}]"
                )

    # P_pos: must be valid position when P_len > 0; must be 0 when P_len == 0
    for i in range(n_rings):
        sz = ring_size(int(R[i]))
        for p in range(P_MAX):
            if P_len[i, p] == 0:
                if P_pos[i, p] != 0:
                    raise ValueError(
                        f"P_pos[{i},{p}]={P_pos[i,p]} but P_len is 0; "
                        "expected P_pos==0 when no pendant."
                    )
            else:
                if P_pos[i, p] < 0 or P_pos[i, p] >= sz:
                    raise ValueError(
                        f"P_pos[{i},{p}]={P_pos[i,p]} out of range "
                        f"[0, {sz - 1}] for ring {i} of size {sz}."
                    )

    return n_rings


def _ring_anchor(F: np.ndarray, k: int) -> int:
    for k_prev in range(k):
        if F[k_prev, k] != F_NONE:
            return k_prev
    raise ValueError(f"No anchor for ring {k}")


def _far_edge_position(sz: int) -> Tuple[int, int]:
    far_a = (sz // 2) - 1 + (sz % 2)
    return far_a, far_a + 1


def _structural_positions_on_ring(
    R: np.ndarray, F: np.ndarray, n_rings: int, ring_atoms: List[List[int]],
    far_edges: Dict[int, Tuple[int, int]],
) -> List[Set[int]]:
    """For each ring, compute the set of traversal positions that are
    "structural" — used by fusion edges or linker attachments. Pendant
    positions must NOT be in this set.

    Returns: structural_positions[k] = set of traversal positions on ring k
             that are bonded to another ring.
    """
    structural: List[Set[int]] = [set() for _ in range(R_MAX)]
    if n_rings == 0:
        return structural

    # Ring 0's anchor edge is conceptually at positions (sz-1, 0); these
    # are NOT used by anything until ring 1 attaches to ring 0's far edge.
    # The far edges propagate via the placement rule.

    for k in range(1, n_rings):
        anchor = _ring_anchor(F, k)
        relation = int(F[anchor, k])

        # Ring `anchor`'s far edge is where ring k attaches
        far_a_pos, far_b_pos = far_edges[anchor]

        if relation == F_FUSED:
            # Ring `anchor` uses both far_a_pos and far_b_pos for the shared edge
            structural[anchor].add(far_a_pos)
            structural[anchor].add(far_b_pos)
            # Ring k traversal: [shared_b, n0, ..., n_last, shared_a]
            # The fusion edge sits at positions (sz-1, 0) of ring k's traversal.
            sz_k = len(ring_atoms[k])
            structural[k].add(0)
            structural[k].add(sz_k - 1)
        elif relation == F_LINKED:
            # Ring `anchor` uses only far_a_pos for the linker attachment
            structural[anchor].add(far_a_pos)
            # Ring k uses position 0 (first atom) for the linker attachment
            structural[k].add(0)
        # F_NONE: no change

    return structural


def compute_atom_count(
    R: np.ndarray, F: np.ndarray, L: np.ndarray,
    P_len: np.ndarray, P_pos: np.ndarray,
) -> int:
    """Total atoms implied by the layout (rings + linker chains + pendants).

    Delegates to build_atom_layout for unified validation. This ensures
    any structural collision (pendant on a fusion edge, etc.) is caught
    here too, not only in build_bond_classes.
    """
    layout = build_atom_layout(R, F, L, P_len, P_pos)
    return layout["M_total"]


def build_atom_layout(
    R: np.ndarray, F: np.ndarray, L: np.ndarray,
    P_len: np.ndarray, P_pos: np.ndarray,
) -> Dict:
    """Compute atom indices for all components.

    Returns a dict with:
      ring_atoms:      list[R_MAX] of list[int]
      structural_positions: list[R_MAX] of set[int] — fusion/linker positions
      linker_atoms:    dict {(i,j): list[int]}
      pendant_atoms:   dict {(ring_i, pendant_idx): (attach_pos, list[int])}
                       attach_pos is the value FROM P_pos (model-supplied).
      M_total:         total atom count
    """
    n_rings = _validate_layout(R, F, L, P_len, P_pos)
    R = np.asarray(R, dtype=np.int64)
    F = np.asarray(F, dtype=np.int64)
    L = np.asarray(L, dtype=np.int64)
    P_len = np.asarray(P_len, dtype=np.int64)
    P_pos = np.asarray(P_pos, dtype=np.int64)

    ring_atoms: List[List[int]] = [[] for _ in range(R_MAX)]
    linker_atoms: Dict[Tuple[int, int], List[int]] = {}
    pendant_atoms: Dict[Tuple[int, int], Tuple[int, List[int]]] = {}

    if n_rings == 0:
        return {
            "ring_atoms": ring_atoms,
            "structural_positions": [set() for _ in range(R_MAX)],
            "linker_atoms": linker_atoms,
            "pendant_atoms": pendant_atoms,
            "M_total": 0,
        }

    next_atom = 0
    far_edges: Dict[int, Tuple[int, int]] = {}

    # Ring 0
    sz0 = ring_size(int(R[0]))
    ring_atoms[0] = list(range(sz0))
    next_atom = sz0
    far_edges[0] = _far_edge_position(sz0)

    # Subsequent rings
    for k in range(1, n_rings):
        anchor = _ring_anchor(F, k)
        sz = ring_size(int(R[k]))
        relation = int(F[anchor, k])

        far_a_pos, far_b_pos = far_edges[anchor]
        anchor_atoms = ring_atoms[anchor]
        far_a_atom = anchor_atoms[far_a_pos]
        far_b_atom = anchor_atoms[far_b_pos]

        if relation == F_FUSED:
            new_count = sz - 2
            new_atoms = list(range(next_atom, next_atom + new_count))
            next_atom += new_count
            ring_atoms[k] = [far_b_atom] + new_atoms + [far_a_atom]
            far_edges[k] = _far_edge_position(sz)
        elif relation == F_LINKED:
            n_link = int(L[anchor, k])
            linker_atom_ids = list(range(next_atom, next_atom + n_link))
            next_atom += n_link
            ring_k_atoms = list(range(next_atom, next_atom + sz))
            next_atom += sz
            ring_atoms[k] = ring_k_atoms
            linker_atoms[(anchor, k)] = linker_atom_ids
            far_edges[k] = _far_edge_position(sz)

    # Compute structural positions for collision checking
    structural = _structural_positions_on_ring(
        R, F, n_rings, ring_atoms, far_edges
    )

    # Pendants — placement at explicit P_pos
    pendant_positions_used: List[Set[int]] = [set() for _ in range(R_MAX)]

    for i in range(n_rings):
        for p in range(P_MAX):
            n_pend = int(P_len[i, p])
            if n_pend == 0:
                continue
            attach_pos = int(P_pos[i, p])

            # Validate: must not collide with structural positions
            if attach_pos in structural[i]:
                raise ValueError(
                    f"Pendant at ring {i} slot {p} requested attach pos "
                    f"{attach_pos}, but that position is structural "
                    f"(used by fusion or linker). Structural positions on "
                    f"ring {i}: {sorted(structural[i])}"
                )

            # Validate: must not collide with another pendant on this ring
            if attach_pos in pendant_positions_used[i]:
                raise ValueError(
                    f"Pendant at ring {i} slot {p} requested attach pos "
                    f"{attach_pos}, but another pendant on this ring "
                    f"already occupies it."
                )
            pendant_positions_used[i].add(attach_pos)

            pend_atom_ids = list(range(next_atom, next_atom + n_pend))
            next_atom += n_pend
            pendant_atoms[(i, p)] = (attach_pos, pend_atom_ids)

    return {
        "ring_atoms": ring_atoms,
        "structural_positions": structural,
        "linker_atoms": linker_atoms,
        "pendant_atoms": pendant_atoms,
        "M_total": next_atom,
    }


def aromatic_constraint_mask_baseline(
    R: np.ndarray, F: np.ndarray, L: np.ndarray,
    P_len: np.ndarray, P_pos: np.ndarray,
    M_MAX_out: Optional[int] = None,
) -> np.ndarray:
    if M_MAX_out is None:
        M_MAX_out = M_MAX  # call-time lookup; robust to monkey-patching
    layout = build_atom_layout(R, F, L, P_len, P_pos)
    out = np.zeros(M_MAX_out, dtype=bool)
    for k, atoms in enumerate(layout["ring_atoms"]):
        if not atoms:
            continue
        if ring_is_aromatic(int(R[k])):
            for a in atoms:
                if a < M_MAX_out:
                    out[a] = True
    return out


def build_bond_classes(
    R: np.ndarray, F: np.ndarray, L: np.ndarray,
    P_len: np.ndarray, P_pos: np.ndarray,
    M_MAX_out: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if M_MAX_out is None:
        M_MAX_out = M_MAX  # call-time lookup; robust to monkey-patching
    layout = build_atom_layout(R, F, L, P_len, P_pos)
    M_total = layout["M_total"]
    if M_total > M_MAX_out:
        raise ValueError(
            f"M_total={M_total} exceeds M_MAX_out={M_MAX_out}; reject sample."
        )

    bond_classes = np.zeros((M_MAX_out, M_MAX_out), dtype=np.int64)
    atom_mask = np.zeros(M_MAX_out, dtype=bool)
    atom_mask[:M_total] = True

    bond_assigned: Dict[Tuple[int, int], int] = {}

    def _set_bond(a: int, b: int, cls: int):
        """Assign bond class for edge (a,b), with aromatic precedence at
        fusion edges.

        When two ring perceptions disagree at a shared edge, AROMATIC wins
        over SINGLE. This handles the common chemistry case of an aromatic
        ring fused to an aliphatic ring (e.g., naphthoquinone, indanone)
        where the aromatic ring's perception of the shared edge as AROMATIC
        is correct, and the aliphatic ring's perception as SINGLE is
        overridden.

        Other mismatches (SINGLE vs DOUBLE, etc.) still raise — those are
        genuinely contradictory bond orders, not aromaticity perception
        ambiguity.
        """
        key = (min(a, b), max(a, b))
        if key in bond_assigned:
            existing = bond_assigned[key]
            if existing == cls:
                return  # Already correct, no-op
            # Resolve aromatic-vs-single mismatch by aromatic precedence
            if {existing, cls} == {BOND_AROMATIC, BOND_SINGLE}:
                # Keep AROMATIC, ignore the SINGLE override
                if existing != BOND_AROMATIC:
                    bond_assigned[key] = BOND_AROMATIC
                    bond_classes[a, b] = BOND_AROMATIC
                    bond_classes[b, a] = BOND_AROMATIC
                # else existing is already AROMATIC; nothing to update
                return
            # Genuine contradiction — different bond orders
            raise ValueError(
                f"Bond class contradiction at edge ({a},{b}): "
                f"existing class {existing}, new class {cls}. "
                "This is a non-aromaticity bond-order mismatch, which "
                "indicates a real layout inconsistency."
            )
        else:
            bond_assigned[key] = cls
            bond_classes[a, b] = cls
            bond_classes[b, a] = cls

    # Ring bonds
    for k in range(R_MAX):
        rt = int(R[k])
        if rt == RING_PAD:
            break
        atoms = layout["ring_atoms"][k]
        if not atoms:
            continue
        cls = BOND_AROMATIC if ring_is_aromatic(rt) else BOND_SINGLE
        sz = len(atoms)
        for i in range(sz):
            a = atoms[i]
            b = atoms[(i + 1) % sz]
            _set_bond(a, b, cls)

    # Linker chain bonds
    R_arr = np.asarray(R, dtype=np.int64)
    F_arr = np.asarray(F, dtype=np.int64)
    n_rings = sum(1 for r in R_arr if int(r) != RING_PAD)
    for k in range(1, n_rings):
        anchor = _ring_anchor(F_arr, k)
        if F_arr[anchor, k] != F_LINKED:
            continue
        linker = layout["linker_atoms"][(anchor, k)]
        far_a_pos, _ = _far_edge_position(ring_size(int(R_arr[anchor])))
        anchor_end_atom = layout["ring_atoms"][anchor][far_a_pos]
        ring_k_end_atom = layout["ring_atoms"][k][0]

        if len(linker) == 0:
            _set_bond(anchor_end_atom, ring_k_end_atom, BOND_SINGLE)
        else:
            _set_bond(anchor_end_atom, linker[0], BOND_SINGLE)
            for i in range(len(linker) - 1):
                _set_bond(linker[i], linker[i + 1], BOND_SINGLE)
            _set_bond(linker[-1], ring_k_end_atom, BOND_SINGLE)

    # Pendant chain bonds
    for (ring_i, p_idx), (attach_pos, pend_atoms) in layout["pendant_atoms"].items():
        if len(pend_atoms) == 0:
            continue
        ring_atoms_i = layout["ring_atoms"][ring_i]
        attach_atom = ring_atoms_i[attach_pos]
        _set_bond(attach_atom, pend_atoms[0], BOND_SINGLE)
        for i in range(len(pend_atoms) - 1):
            _set_bond(pend_atoms[i], pend_atoms[i + 1], BOND_SINGLE)

    return bond_classes, atom_mask


def decode_scaffold_baseline(
    R: np.ndarray, F: np.ndarray, L: np.ndarray,
    P_len: np.ndarray, P_pos: np.ndarray,
    atom_ids_compact: np.ndarray,
    M_MAX_out: Optional[int] = None,
    aromatic_atom_ids: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if M_MAX_out is None:
        M_MAX_out = M_MAX  # call-time lookup; robust to monkey-patching
    """Compose the layout + atom IDs into the earlier version-compatible scaffold tensors.

    Returns
    -------
    atom_ids_padded : (M_MAX_out,) int
    bond_classes    : (M_MAX_out, M_MAX_out) int
    atom_mask       : (M_MAX_out,) bool
    """
    if aromatic_atom_ids is None:
        aromatic_atom_ids = AROMATIC_ATOM_IDS

    R = np.asarray(R, dtype=np.int64)
    F = np.asarray(F, dtype=np.int64)
    L = np.asarray(L, dtype=np.int64)
    P_len = np.asarray(P_len, dtype=np.int64)
    P_pos = np.asarray(P_pos, dtype=np.int64)
    atom_ids_compact = np.asarray(atom_ids_compact, dtype=np.int64)

    M_total = compute_atom_count(R, F, L, P_len, P_pos)
    if atom_ids_compact.shape != (M_total,):
        raise ValueError(
            f"atom_ids_compact shape {atom_ids_compact.shape} doesn't match "
            f"M_total={M_total}."
        )
    if M_total > M_MAX_out:
        raise ValueError(
            f"M_total={M_total} exceeds M_MAX_out={M_MAX_out}; reject sample."
        )

    constraint = aromatic_constraint_mask_baseline(R, F, L, P_len, P_pos, M_MAX_out=M_MAX_out)
    for a in range(M_total):
        if constraint[a] and atom_ids_compact[a] not in aromatic_atom_ids:
            raise ValueError(
                f"Atom {a} at aromatic-ring position must have aromatic ID, "
                f"got {atom_ids_compact[a]}"
            )

    atom_ids_padded = np.zeros(M_MAX_out, dtype=np.int64)
    atom_ids_padded[:M_total] = atom_ids_compact
    bond_classes, atom_mask = build_bond_classes(
        R, F, L, P_len, P_pos, M_MAX_out=M_MAX_out
    )

    return atom_ids_padded, bond_classes, atom_mask


def decode_layout_batch(
    R_batch: np.ndarray,
    F_batch: np.ndarray,
    L_batch: np.ndarray,
    P_len_batch: np.ndarray,
    P_pos_batch: np.ndarray,
    atom_ids_compact_batch: List[np.ndarray],
    M_MAX_out: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if M_MAX_out is None:
        M_MAX_out = M_MAX  # call-time lookup; robust to monkey-patching
    """Batch wrapper for decode_scaffold_baseline."""
    R_batch = np.asarray(R_batch, dtype=np.int64)
    F_batch = np.asarray(F_batch, dtype=np.int64)
    L_batch = np.asarray(L_batch, dtype=np.int64)
    P_len_batch = np.asarray(P_len_batch, dtype=np.int64)
    P_pos_batch = np.asarray(P_pos_batch, dtype=np.int64)
    B = R_batch.shape[0]
    if (F_batch.shape[0] != B or L_batch.shape[0] != B
        or P_len_batch.shape[0] != B or P_pos_batch.shape[0] != B
        or len(atom_ids_compact_batch) != B):
        raise ValueError("Batch dimension mismatch")

    out_atom_ids = np.zeros((B, M_MAX_out), dtype=np.int64)
    out_bonds = np.zeros((B, M_MAX_out, M_MAX_out), dtype=np.int64)
    out_mask = np.zeros((B, M_MAX_out), dtype=bool)

    for b in range(B):
        a, c, m = decode_scaffold_baseline(
            R_batch[b], F_batch[b], L_batch[b],
            P_len_batch[b], P_pos_batch[b],
            atom_ids_compact_batch[b],
            M_MAX_out=M_MAX_out,
        )
        out_atom_ids[b] = a
        out_bonds[b] = c
        out_mask[b] = m

    return out_atom_ids, out_bonds, out_mask


def list_valid_pendant_positions(
    R: np.ndarray, F: np.ndarray, L: np.ndarray,
    ring_idx: int,
) -> List[int]:
    """Helper: list traversal positions on `ring_idx` that can legally
    host a pendant (i.e., not used by fusion or linker).

    Useful for Step 2's dataset extraction (knowing which positions are
    valid candidates) and for any sampler-time validation. Does not
    consider P_pos itself, so pendant-pendant collision must be checked
    by the caller.
    """
    # Build a temporary layout with no pendants; use _structural_positions_on_ring
    # via build_atom_layout's internal call.
    P_len = np.zeros((R_MAX, P_MAX), dtype=np.int64)
    P_pos = np.zeros((R_MAX, P_MAX), dtype=np.int64)
    layout = build_atom_layout(R, F, L, P_len, P_pos)
    n_rings = sum(1 for r in np.asarray(R, dtype=np.int64) if int(r) != RING_PAD)
    if ring_idx < 0 or ring_idx >= n_rings:
        return []
    sz = ring_size(int(R[ring_idx]))
    structural = layout["structural_positions"][ring_idx]
    return [pos for pos in range(sz) if pos not in structural]


# ────────────────────────────────────────────────────────────────────────
# decoder: decode_scaffold
# ────────────────────────────────────────────────────────────────────────
#
# Reconstruct the bond_classes matrix from labels. differs from
# the earlier encoder in two ways:
#   1) F_SPIRO=3 ring relations: single bond between the spiro atom and
#      itself (it's one atom shared by both rings, no extra bonds needed
#      since the rings' own bonds connect to it).
#   2) Branched pendants: B_size/B_pos/B_parent/B_bond replace P_len/P_pos.
#
# Builds (M_MAX, M_MAX) bond_classes int matrix where entries are:
#   0 = no bond, 1 = single, 2 = aromatic, 3 = double, 4 = triple

def decode_scaffold(
    R: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    B_size: np.ndarray,
    B_pos: np.ndarray,
    B_parent: np.ndarray,
    B_bond: np.ndarray,
    spiro_atom_positions: np.ndarray,
    atom_ids: np.ndarray,
    M_MAX_out: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode labels back to bond_classes and atom_mask.

    Returns (bond_classes, atom_mask) where:
      bond_classes: (M_MAX_out, M_MAX_out) int — bond class between every pair
      atom_mask:    (M_MAX_out,) bool — True for real atoms, False for PAD
    """
    if M_MAX_out is None:
        M_MAX_out = M_MAX

    # First, walk the canonical layout to figure out which canonical index
    # corresponds to each ring/linker/branch atom. Then emit bonds.

    # Determine n_rings
    n_rings = int((R != RING_PAD).sum())
    if n_rings == 0:
        # All-empty molecule (shouldn't happen for real samples)
        bond_classes = np.zeros((M_MAX_out, M_MAX_out), dtype=np.int64)
        atom_mask = np.zeros(M_MAX_out, dtype=bool)
        return bond_classes, atom_mask

    # Ring sizes and aromaticity from R
    ring_sizes = []
    ring_aromatic = []
    for k in range(n_rings):
        sz, is_arom = RING_TYPE_INFO[int(R[k])]
        ring_sizes.append(sz)
        ring_aromatic.append(is_arom)

    # Build canonical layout — same algorithm as the encoder
    # Track: for each ring k, the canonical positions of its atoms (in traversal order)
    ring_canonical_positions: Dict[int, List[int]] = {}
    next_canon = 0
    # Ring 0: all atoms
    ring_canonical_positions[0] = list(range(next_canon, next_canon + ring_sizes[0]))
    next_canon += ring_sizes[0]

    # Subsequent rings + linkers
    for k in range(1, n_rings):
        # Find anchor (the connected previous ring with F != F_NONE)
        anchor = -1
        for p in range(k):
            if F[p, k] != F_NONE:
                anchor = p
                break
        if anchor < 0:
            raise ValueError(f"Decoder: ring {k} has no anchor in F-matrix")
        relation = int(F[anchor, k])
        sz_k = ring_sizes[k]

        if relation == F_FUSED:
            # ring k's atoms at positions 0 and sz_k-1 ARE shared with anchor ring;
            # they sit at the last two canonical positions of the anchor ring.
            # ring k contributes atoms 1..sz_k-2 (new) plus borrows 0 and sz_k-1.
            anchor_canon = ring_canonical_positions[anchor]
            # By the earlier encoder convention, the fusion atoms in ring k's traversal are
            # at positions (0, sz_k-1) and they are the (sz_anchor-1, 0) of anchor ring
            # i.e. the LAST two atoms of anchor's traversal are the shared ones.
            shared_canon_a = anchor_canon[-1]  # atom 0 of ring k
            shared_canon_b = anchor_canon[-2]  # atom sz_k-1 of ring k? (depends on direction)
            # Internal atoms 1..sz_k-2 get fresh canonical positions
            internal_canons = list(range(next_canon, next_canon + sz_k - 2))
            next_canon += sz_k - 2
            ring_canonical_positions[k] = [shared_canon_a] + internal_canons + [shared_canon_b]
        elif relation == F_LINKED:
            link_len = int(L[anchor, k])
            link_canons = list(range(next_canon, next_canon + link_len))
            next_canon += link_len
            ring_canons = list(range(next_canon, next_canon + sz_k))
            next_canon += sz_k
            ring_canonical_positions[k] = ring_canons
            # Track linker positions for bond emission later
            ring_canonical_positions[(anchor, k, 'linker')] = link_canons
        elif relation == F_SPIRO:
            # ring k's atom 0 is the spiro atom; it equals some position in anchor ring.
            # Find the spiro atom's canonical position in anchor ring.
            spiro_pos_in_anchor = int(spiro_atom_positions[anchor, k])
            if spiro_pos_in_anchor < 0:
                raise ValueError(f"Decoder: F_SPIRO but spiro_atom_positions[{anchor},{k}]=-1")
            shared_canon = ring_canonical_positions[anchor][spiro_pos_in_anchor]
            # ring k's atoms 1..sz_k-1 are new
            new_canons = list(range(next_canon, next_canon + sz_k - 1))
            next_canon += sz_k - 1
            ring_canonical_positions[k] = [shared_canon] + new_canons
        else:
            raise ValueError(f"Decoder: unknown F-value {relation} at ({anchor},{k})")

    # Branch atoms — for each (k, slot), track canonical positions
    branch_canonical_positions: Dict[Tuple[int, int], List[int]] = {}
    R_MAX_in, P_MAX_in = B_size.shape[0], B_size.shape[1]
    for k in range(n_rings):
        for slot in range(P_MAX_in):
            sz = int(B_size[k, slot])
            if sz == 0:
                continue
            canons = list(range(next_canon, next_canon + sz))
            next_canon += sz
            branch_canonical_positions[(k, slot)] = canons

    # Now emit bonds
    M_total = next_canon
    if M_total > M_MAX_out:
        # Without this check the overflow surfaces as an opaque IndexError
        # from the bond writes below, which is a confusing way to learn
        # that a sampled layout simply describes too large a molecule.
        raise ValueError(
            f"Decoded layout needs {M_total} atoms but M_MAX_out is "
            f"{M_MAX_out}. The layout describes a molecule larger than "
            f"the model's atom budget; either raise M_MAX or reject this "
            f"sample."
        )
    bond_classes = np.zeros((M_MAX_out, M_MAX_out), dtype=np.int64)
    atom_mask = np.zeros(M_MAX_out, dtype=bool)
    atom_mask[:M_total] = True

    def _set_bond(i, j, cls):
        if cls == 0: return
        bond_classes[i, j] = cls
        bond_classes[j, i] = cls

    # Ring internal bonds
    for k in range(n_rings):
        canons = ring_canonical_positions[k]
        sz = len(canons)
        bond_cls = BOND_AROMATIC if ring_aromatic[k] else BOND_SINGLE
        for i in range(sz):
            _set_bond(canons[i], canons[(i + 1) % sz], bond_cls)

    # Linker bonds (LINKED relation)
    for k in range(1, n_rings):
        anchor = -1
        for p in range(k):
            if F[p, k] != F_NONE:
                anchor = p; break
        if F[anchor, k] != F_LINKED:
            continue
        link_canons = ring_canonical_positions[(anchor, k, 'linker')]
        anchor_canons = ring_canonical_positions[anchor]
        k_canons = ring_canonical_positions[k]
        # Linker connects last atom of anchor to first atom of k
        # (the earlier encoder convention: linkers attach at ring atom 0)
        if len(link_canons) == 0:
            # Direct ring-ring bond
            _set_bond(anchor_canons[0], k_canons[0], BOND_SINGLE)
        else:
            _set_bond(anchor_canons[0], link_canons[0], BOND_SINGLE)
            for li in range(len(link_canons) - 1):
                _set_bond(link_canons[li], link_canons[li + 1], BOND_SINGLE)
            _set_bond(link_canons[-1], k_canons[0], BOND_SINGLE)

    # Spiro: NO extra bond needed — the spiro atom is shared, so the ring's own
    # internal bonds already account for connectivity. The shared canonical
    # position appears in both rings' canons[] lists, and each ring's internal
    # cycle bonds touch it correctly.

    # Branch bonds: tree edges within each branch + branch-root to ring-atom
    for (k, slot), branch_canons in branch_canonical_positions.items():
        sz = int(B_size[k, slot])
        root_pos = int(B_pos[k, slot])
        ring_canons = ring_canonical_positions[k]
        # parent_within_tree==0 means "the ring atom at root_pos"
        # parent_within_tree==i (>=1) means "atom i-1 of branch_canons"
        for i in range(sz):
            p = int(B_parent[k, slot, i])
            b = int(B_bond[k, slot, i])
            if p == 0:
                parent_canon = ring_canons[root_pos]
            else:
                parent_canon = branch_canons[p - 1]
            _set_bond(branch_canons[i], parent_canon, b)

    return bond_classes, atom_mask


def aromatic_constraint_mask(
    bond_classes: np.ndarray,
    atom_mask: np.ndarray,
) -> np.ndarray:
    """aromatic-constraint mask, derived from decoded bond_classes.

    An atom is aromatic-constrained iff it has any incident BOND_AROMATIC
    edge. By construction of decode_scaffold, only ring atoms in
    aromatic rings receive BOND_AROMATIC edges; branch atoms always get
    BOND_SINGLE/DOUBLE/TRIPLE per the encoder's branch-bond convention.
    So this is rigorous, not heuristic.

    Args:
        bond_classes: (M_MAX, M_MAX) int — output of decode_scaffold
        atom_mask:    (M_MAX,)        bool — output of decode_scaffold

    Returns:
        (M_MAX,) bool — True for real atoms that must be aromatic.
    """
    has_arom = (bond_classes == BOND_AROMATIC).any(axis=1)
    return has_arom & atom_mask
