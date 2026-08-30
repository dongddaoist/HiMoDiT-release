# Known limitations

Measured behaviours that constrain what this model can represent or how
its numbers should be read. Each entry says what the effect is, how it
was measured, and what to do about it.

---

## 1. Substituents on linker atoms are silently dropped

**Effect.** Branch trees are grown outward from *ring* atoms only. A
substituent attached to a **linker** atom — one of the chain atoms
joining two rings — belongs to none of the three categories the encoder
collects (ring, linker, terminal), so it is never walked. The resulting
label describes a molecule slightly smaller than the input, with no
error raised.

**Example.**

```
CC[C@@H](OC(=O)Cc1c(C)nc(C)[nH]c1=O)c1cccc([N+](=O)[O-])c1
```

The ethyl group hangs off the linker carbon between the two rings. The
terminal `CH3` is captured by the terminal detector, but the middle
carbon is not captured by anything. The label encodes 24 of the 25 heavy
atoms.

**Measured on ZINC250K** (4,000-molecule sample, 3,765 accepted):

| | |
|---|---|
| Labels with atoms dropped | 243 (**6.45%** of accepted) |
| Total atoms dropped | 370 |
| Bond-count mismatches | 243 (the same molecules, 1:1) |

Every affected molecule has `F_LINKED` with a non-zero linker length.
The bond delta is always negative — the decoder emits *fewer* bonds than
the source scaffold, never more.

**Why it exists.** The predecessor encoder validated every label by
decoding it and comparing against the source, which caught this class of
problem as a `decoder_roundtrip_failed` rejection. The current encoder
dropped that check when branch trees were introduced.

**What to do.** Pass `strict=True` to `extract_layout`, or `--strict` to
`scripts/preprocess.py`. This rejects any molecule whose atoms are not
fully accounted for, restoring an exact round-trip guarantee at a cost of
roughly six percentage points of retention.

The default is `strict=False` so the published retention figure stays
reproducible. Which setting is right depends on what the labels are for:
strict mode for anything where label fidelity matters, loose mode for
maximum training-set coverage.

**Fixing it properly** would mean extending the branch walk to root from
linker atoms as well as ring atoms, which requires a new label field for
the linker position a branch attaches to. That is a schema change and a
retrain, not a patch.

---

## 2. A3 can emit non-causal branch parents

**Effect.** A3 samples `B_parent[i]` for every branch atom independently,
with nothing constraining the parent index to be smaller than `i`. Two
bad cases follow:

- `B_parent[i] > i` names a parent that comes later in the branch, which
  is not a tree.
- `B_parent[i] - 1 >= B_size` indexes past the end of the branch, which
  raises `IndexError` inside the decoder and costs the whole molecule.

The training script already measures this — `evaluate_structural_validity`
reports `rate_parent_causal` — but nothing enforces it at sampling time.

**What to do.** Pass `enforce_causal_parent=True` to
`generate_batch`, or `--enforce-causal-parent` to `scripts/generate.py`.
This clamps each parent to `min(B_parent[i], i, B_size)`, so an
out-of-range parent falls back to the nearest legal one rather than
losing the molecule.

Off by default, because the published numbers were produced without it
and its effect on validity has not been measured at scale. It should
only help, but "should" is not a measurement.

---

## 3. Peri-fused and angular ring systems are rejected

The ring graph must be a tree. Systems where three or more rings share
atoms in a cycle — pyrene, acenaphthylene — cannot be expressed, and
neither can angular fusion such as phenanthrene, where two fusion bonds
on the same ring are not opposite each other.

On ZINC250K these are the largest remaining rejection buckets:

| Reason | Share of evaluated molecules |
|---|---|
| `ring_graph_has_cycle_peri_fusion` | ~3.1% |
| `ring_*_angular_fusion` | ~1.2% |
| `rings_*_peri_invalid` | ~0.8% |

Lifting the tree constraint would change A1's F-matrix from a tree
adjacency to a general graph, which is a substantially harder prediction
problem. It was left in place deliberately.

---

## 4. Ring-free molecules cannot be represented

The whole hierarchy is anchored on ring 0, so a molecule with no rings
has nowhere to start. These are rejected as `no_rings`, about 0.4% of
ZINC250K. Acyclic chemistry needs a different decomposition.

---

## 5. Capacity ceilings

Fixed by the label tensor shapes. Exceeding any of them is a rejection,
not a truncation.

| Constant | Value | Meaning |
|---|---|---|
| `R_MAX` | 6 | rings per molecule |
| `M_MAX` | 40 | scaffold atoms |
| `L_MAX` | 10 | atoms in a linker chain |
| `P_MAX_BRANCH` | 8 | branch slots per ring |
| `B_LEN_MAX` | 15 | atoms in one branch tree |
| ring sizes | 3–7 | larger rings rejected |

Raising any of these requires re-encoding the dataset and retraining
every stage, since the token layouts derive from them.

---

## 6. Scaffold bonds are limited to single and aromatic

`decode_scaffold` emits `BOND_SINGLE` or `BOND_AROMATIC` for ring-internal
bonds, chosen by whether the ring type is aromatic. There is no way to
express a ring-internal double bond, so quinoid rings and other
cross-conjugated systems cannot be represented exactly in the scaffold.

Branch bonds are unaffected — `B_bond` carries the full five-class
vocabulary including double and triple.

---

## 7. Reported validity depends on the repair cascade

`himodit.chem.compose` repairs molecules before giving up: aromaticity is
re-perceived after a kekulization failure, and over-valent atoms lose a
non-ring bond after a valence failure. Validity figures include these
repairs.

This is a defensible choice — the decoder's job is to turn a graph into a
molecule consistently, and RDKit's aromaticity perception is part of that
pipeline rather than an external fix — but it is not the same measurement
as validity before repair, and comparisons against models that do not
repair should say so.

---

## 8. Capacity preset names are approximate

The capacity strings are labels, not parameter counts. Actual sizes:

| Stage | Preset | Actual parameters |
|---|---|---|
| A1 | `3M` | 3.6 M |
| A3 | `10M` | 19.2 M |
| A2 | `10M` | 8.5 M |
| Terminal | `9M` | 8.6 M |

A3's `10M` preset is nearly twice its nominal size, because its four
output heads over `R_MAX × P_MAX_BRANCH × B_LEN_MAX` positions dominate
the parameter count.
