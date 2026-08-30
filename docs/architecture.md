# Architecture

HiMoDiT generates a molecule by deciding its structure in four passes,
from the most abstract property to the most local one. Each pass sees the
output of the previous one and the property condition.

```
        condition (z-scored property targets)
                    │
        ┌───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
   │   A1    │→│   A3    │→│   A2    │→│ Terminal │
   │  rings  │ │ branches│ │  atoms  │ │ decoration│
   └─────────┘ └─────────┘ └─────────┘ └──────────┘
        │           │           │           │
        └─────┬─────┘           │           │
              ▼                 │           │
      decode_scaffold ──────────┘           │
     (deterministic, no parameters)         │
              │                             │
              └──────────► compose ◄────────┘
                             │
                             ▼
                          SMILES
```

## Why a cascade

A molecular graph has structure at several scales at once: which rings
exist, how they connect, what hangs off them, what each atom is. Models
that denoise the whole adjacency matrix at once have to discover that
hierarchy from scratch, and can produce graphs that are internally
inconsistent — a bond pattern that is not a valid ring, an aromatic flag
on an atom in an aliphatic ring.

Here the hierarchy is built into the representation. Ring topology is
decided first, in a space where the only expressible objects are valid
ring systems. Atom identities are decided last, against a bond graph that
is already fixed. The stages cannot contradict each other because they
never make overlapping decisions.

## The deterministic decoder

The step that does the most work carries no parameters at all.

`decode_scaffold` takes the layout — ring types, relations, linker
lengths, spiro positions, branch trees — and constructs the bond matrix
by following the canonical atom ordering. Ring closure is guaranteed
because rings are emitted as cycles. Fusion shares exactly two atoms
because that is what the fusion branch does. Nothing is sampled, so
nothing can be malformed.

This is what lets A2 treat the bond graph as a clean, known input rather
than something to be jointly denoised, which is the failure mode that
motivated the design.

---

## Stage A1 — ring layout

**`condition → (R, F, L, spiro_pos_class)`**

Masked absorbing-state discrete diffusion over the macro layout. Tokens
are 6 ring slots plus 15 unordered ring pairs, 21 in total. Ring tokens
predict ring type; pair tokens predict the relation, linker length, and
spiro position for that pair.

Attention is plain self-attention. The pairwise structure here is the
prediction target rather than a known input graph, so there is nothing to
bias attention on.

Sampling is iterative confidence-based unmasking: at each step the most
confident still-masked positions are committed, with Gumbel perturbation
for diversity. Post-processing enforces the invariants the decoder
requires — symmetry, zeroing linker lengths off linked pairs, downgrading
a spiro relation whose position came back as the sentinel.

## Stage A3 — branch topology

**`(R, F, L, spiro_pos) + condition → (B_size, B_pos, B_parent, B_bond)`**

Predicts the side-chain trees. Tokens are the ring and pair tokens from
A1 plus 48 branch-slot tokens, 69 in total. Slot tokens start as
positional embeddings and gather context through self-attention, then
read out through four heads.

Unlike A1 and A2 this is a single forward pass rather than an iterative
denoiser — branch topology is a smaller and more local decision, and
iterating did not earn its cost. Classifier-free guidance still applies,
through a paired unconditional pass.

A3 predicts topology only, never atom identity. Every atom id belongs to
A2.

## Stage A2 — atom identity

**`decoded scaffold + condition → atom_ids`**

Masked discrete diffusion over element identities, one token per atom
slot. This is the only stage whose attention is edge-biased: here the
bond graph *is* known, so the score between two atoms is biased by a
learned per-head function of the bond class joining them, letting heads
route information along aromatic paths.

Atoms in aromatic rings are constrained to aromatic identities by a hard
logit mask at sampling time. During training the constraint is left to
the data, which contains only valid combinations; the mask guarantees it
holds even for layouts outside the training distribution.

## Stage 2 — terminal decoration

**`scaffold + condition → per-atom fragment id`**

Per-atom categorical over 23 classes: "leave bare", or one of 22
functional groups. Single forward pass with bond-biased attention.

`himodit.chem.compose` then grafts the chosen fragments, skipping any
graft that would over-saturate its host, and sanitizes with a repair
cascade.

---

## Conditioning

Every stage takes the same z-scored property vector and injects it
through AdaLN-Zero modulation: the normalization gain and bias are
predicted from the condition, zero-initialized so each block starts as
the identity.

Classifier-free guidance is trained by dropping the condition with some
probability (0.1 for A1, A2, and Terminal; 0.3 for A3) and applied at
sampling by extrapolating between the conditional and unconditional
logits.

Because all four stages are conditioned, a property target influences the
molecule at every scale — ring count, branch size, heteroatom placement,
and functional groups — rather than only the last one.

---

## Capacity

| Stage | Preset | Parameters | Tokens |
|---|---|---|---|
| A1 | `3M` | 3.6 M | 21 |
| A3 | `10M` | 19.2 M | 69 |
| A2 | `10M` | 8.5 M | 40 |
| Terminal | `9M` | 8.6 M | 40 |

Preset names are labels rather than measurements; A3's four output heads
over `6 × 8 × 15` positions make it much larger than its name suggests.

---

## What this is not

The model is **masked absorbing-state discrete diffusion**, in the D3PM
and MaskGIT line, not flow matching. Earlier iterations of this project
were named for a flow-matching formulation that the implementation never
actually used.

Generation is **not single-step**. A1 and A2 each run a configurable
number of unmasking steps (20 by default), A3 and Terminal are single
passes. The cascade is fast because the token sequences are short — tens
of tokens, not hundreds of atoms — not because it is one-shot.
