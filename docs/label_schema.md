# Label schema

What `extract_layout` produces and every stage consumes. One label
describes one molecule.

```python
from himodit.chem.encoder import extract_layout

label, reason = extract_layout("CC(=O)Nc1ccc(O)cc1")
```

On success `reason` is `None`; on rejection `label` is `None` and
`reason` names the rule that was violated.

---

## Capacity constants

All shapes below derive from these. They live in
`himodit.chem.decoder` and are imported everywhere else, so encoder,
models, and decoder cannot drift apart.

| Constant | Value | Meaning |
|---|---|---|
| `R_MAX` | 6 | ring slots |
| `M_MAX` | 40 | scaffold atom slots |
| `L_MAX` | 10 | maximum linker length |
| `P_MAX_BRANCH` | 8 | branch slots per ring |
| `B_LEN_MAX` | 15 | maximum atoms in one branch tree |

---

## Fields

### `R` — ring types

`(R_MAX,)` int. One entry per ring slot, left-packed, `0` for empty.

| id | ring | id | ring |
|---|---|---|---|
| 0 | PAD | 6 | 3-membered aliphatic |
| 1 | 6-membered aromatic | 7 | 4-membered aromatic |
| 2 | 6-membered aliphatic | 8 | 4-membered aliphatic |
| 3 | 5-membered aromatic | 9 | 7-membered aromatic |
| 4 | 5-membered aliphatic | 10 | 7-membered aliphatic |
| 5 | 3-membered aromatic | | |

### `F` — ring relations

`(R_MAX, R_MAX)` int, symmetric, zero diagonal.

| value | name | meaning |
|---|---|---|
| 0 | `F_NONE` | no direct relation |
| 1 | `F_FUSED` | rings share an edge (2 atoms) |
| 2 | `F_LINKED` | joined by a bond, optionally through a chain |
| 3 | `F_SPIRO` | share exactly one sp3-quaternary atom |

The ring graph induced by non-zero entries must be a **tree**. Cycles
are peri-fusion and are rejected.

### `L` — linker lengths

`(R_MAX, R_MAX)` int, symmetric. Meaningful only where `F == F_LINKED`,
zero elsewhere. `L[i,j]` counts the atoms *between* the two rings, so
`L = 0` is a direct ring-to-ring bond, as in biphenyl.

### `spiro_atom_positions` — shared-atom locations

`(R_MAX, R_MAX)` int. Where `F == F_SPIRO`, the position of the shared
atom in the anchor ring's canonical traversal. **`-1` elsewhere.**

This field uses three different encodings across the codebase. All
conversions live in `himodit/pipeline.py`:

| Component | "no spiro" | positions |
|---|---|---|
| Encoder and decoder | `-1` | `0..6` |
| A1 (`spiro_pos_class`) | `0` | `1..7` |
| A3 (`NO_SPIRO_CLS`) | `7` | `0..6` |

A1 shifts because embedding layers cannot take a negative index; A3 uses
an in-vocabulary sentinel instead. Use `to_a3_spiro` and
`to_decoder_spiro` rather than open-coding the arithmetic.

### `B_size`, `B_pos` — branch slots

Both `(R_MAX, P_MAX_BRANCH)` int.

- `B_size[k,p]` — atoms in the branch tree at ring `k`, slot `p`.
  `0` means the slot is empty.
- `B_pos[k,p]` — the position on ring `k` the branch is rooted at.
  Zero when the slot is empty.

### `B_parent`, `B_bond` — branch structure

Both `(R_MAX, P_MAX_BRANCH, B_LEN_MAX)` int. Indexed by atom `i` within
the branch.

- `B_parent[k,p,i]` — `0` means the parent is the ring atom at
  `B_pos[k,p]`; a value `j >= 1` means atom `j-1` of this same branch.
- `B_bond[k,p,i]` — bond class joining atom `i` to its parent:
  `1` single, `2` aromatic, `3` double, `4` triple.

Entries at `i >= B_size[k,p]` are zero padding.

### `atom_ids` — element identities

`(M_total,)` int, in canonical atom order. This is A2's prediction
target.

| id | symbol | id | symbol | id | symbol |
|---|---|---|---|---|---|
| 0 | `<PAD>` | 6 | S | 12 | n+ |
| 1 | c | 7 | F | 13 | N- |
| 2 | O | 8 | s | 14 | n- |
| 3 | C | 9 | o | 15 | P+ |
| 4 | N | 10 | O- | | |
| 5 | n | 11 | N+ | | |

Lowercase is aromatic. Halogens other than F are absent by design: in
ZINC250K every Cl, Br, and I sits in a terminal position, so they are
handled by the terminal vocabulary instead.

### `terminals` — detected functional groups

List of dicts, one per detected terminal fragment:

```python
{
    "name": "OH",                # vocabulary entry
    "atom_indices": [12],        # RDKit indices in the source molecule
    "host_atom_idx": 5,          # scaffold atom it attaches to
    "host_canonical_idx": 3,     # that atom's canonical index
    "anchor_atom_idx": 12,       # fragment-side atom of the bond
    "attach_bond_class": 1,
    "host_is_aromatic": True,
}
```

`host_canonical_idx` is the field the Terminal stage trains on; it is
`-1` if the host did not make it into the canonical layout.

### Scalars

| Field | Meaning |
|---|---|
| `smi` | canonical SMILES of the input |
| `M_total` | scaffold atom count |
| `n_spiro_junctions` | number of spiro junctions |
| `n_branches` | number of occupied branch slots |
| `n_atoms_unaccounted` | heavy atoms the encoder did not collect (see below) |
| `condition` | `(condition_dim,)` float32, added by `scripts/preprocess.py` |

`n_atoms_unaccounted` should be `0`. A non-zero value means the label
describes a smaller molecule than the input — see
[limitations.md](limitations.md#1-substituents-on-linker-atoms-are-silently-dropped).
`strict=True` rejects those molecules instead.

---

## Canonical atom order

The decoder reconstructs atom indices deterministically, and the encoder
must agree exactly:

1. **Ring 0** contributes all its atoms in traversal order.
2. **Each subsequent ring `k`**, in index order, by its relation to its
   anchor:
   - *fused* — borrows 2 atoms from the anchor, appends `size_k - 2` new
   - *linked* — appends `L[anchor,k]` linker atoms, then all `size_k`
   - *spiro* — borrows 1 atom from the anchor, appends `size_k - 1` new
3. **Branch trees**, in `(ring, slot)` order, each in tree order.

`M_total` is the total. Because the order is deterministic, no atom
mapping needs to be stored.

---

## Which stage reads what

| Stage | Reads | Predicts |
|---|---|---|
| A1 | `condition` | `R`, `F`, `L`, `spiro_pos_class` |
| A3 | `R`, `F`, `L`, `spiro_pos`, `condition` | `B_size`, `B_pos`, `B_parent`, `B_bond` |
| A2 | decoded `bond_classes`, `atom_mask`, `arom_mask`, `condition` | `atom_ids` |
| Terminal | `atom_ids`, `bond_classes`, `atom_mask`, `condition` | per-atom fragment id |

A2 never sees the raw layout, only the decoded bond matrix — training and
inference inputs are therefore identical.
