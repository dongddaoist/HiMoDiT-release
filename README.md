# HiMoDiT

**Hierarchical Molecular Diffusion Transformer** — property-conditioned
generation of drug-like molecules through a chemically structured latent
space.

Instead of denoising an adjacency matrix, HiMoDiT decides a molecule's
structure in four passes, from the most abstract feature to the most
local: which rings exist and how they connect, what branches off them,
what each atom is, and finally what functional groups decorate it. The
bond matrix between passes is built by a deterministic decoder rather
than sampled, so the stages cannot contradict each other and every
scaffold is a valid ring system by construction.

```
condition ──►   rings ──►   branches ──►   atoms ──► Terminal ──► SMILES
                    └──────────┬─────────┘
                     decode_scaffold
```

---

## Results

### Encoder coverage on ZINC250K

How much of a real drug-like library the representation can express.
Measured with `scripts/validate_encoder.py`; reproduce with the command
in [Reproducing the numbers](#reproducing-the-numbers).

| Encoder | Retention | Molecules |
|---|---:|---|
| Baseline (linear side chains, no spiro) | 83.36% | 120,000 |
| **Current (branch trees + spiro)** | **93.38%** | **249,455 (all)** |
| Current, `--strict` | 87.43% | 120,000 |

The +10 point gain over the baseline comes from two representational
changes: side chains became trees with parent indices rather than linear
chains, and spiro junctions at sp3-quaternary centres became expressible
instead of rejected.

`--strict` additionally rejects molecules the encoder cannot represent
without dropping atoms — see [limitations](docs/limitations.md#1-substituents-on-linker-atoms-are-silently-dropped).

### Round-trip fidelity

Do accepted labels rebuild the source molecule's scaffold exactly?
Measured over 37,276 accepted labels:

| Outcome | Share |
|---|---:|
| Exact match, atoms and bonds | **93.64%** |
| Bond-count mismatch | 6.36% |

The mismatches are all molecules with a substituent on a linker atom,
which the ring-rooted branch walk never reaches. This is a known
representational gap, documented with its reproduction in
[docs/limitations.md](docs/limitations.md). `--strict` preprocessing
removes it, trading about six points of retention for an exact
round-trip guarantee.

### Generation

Generation metrics depend on trained checkpoints and are not reproduced
here. Run `scripts/generate.py` after training to produce them.

> **If you are reproducing previously reported validity figures, re-run
> them.** A defect fixed in this release caused failed scaffold decodes
> to be scored as valid molecules: `Chem.MolFromSmiles("")` returns a
> valid zero-atom `Mol` rather than `None`, so an empty assembly passed
> the usual `if mol is None` check. `compute_vun` now counts zero-atom
> molecules separately as `n_empty`, and `assemble_molecule` returns
> `None` for an empty scaffold.

---

## Installation

Requires Python 3.9+ and RDKit.

```bash
git clone https://github.com/dongddaoist/himodit.git
cd himodit
pip install -e .
```

Or without installing:

```bash
pip install -r requirements.txt
export PYTHONPATH=.
```

For GPU training, install the PyTorch build matching your CUDA version
from [pytorch.org](https://pytorch.org) first.


---

## Quick start

### Encode a molecule

```python
from himodit.chem.encoder import extract_layout

label, reason = extract_layout("CC(=O)Nc1ccc(O)cc1")   # paracetamol
print(label["R"])           # ring types, left-packed
print(label["M_total"])     # scaffold atom count
print(label["terminals"])   # detected functional groups
```

`label` is `None` on rejection, with `reason` naming the rule violated.
Every field is documented in [docs/label_schema.md](docs/label_schema.md).

### Decode a layout back to a bond graph

```python
from himodit.chem.decoder import decode_scaffold, M_MAX

bond_classes, atom_mask = decode_scaffold(
    label["R"], label["F"], label["L"],
    label["B_size"], label["B_pos"], label["B_parent"], label["B_bond"],
    label["spiro_atom_positions"], label["atom_ids"], M_MAX_out=M_MAX,
)
```

### Generate molecules

```python
import torch
from himodit.pipeline import HiMoDiT

model = HiMoDiT.from_checkpoints("checkpoints/")

smiles = model.generate(n=1000, cfg_scale=1.5)

# Or steer toward a property target, in z-scored units:
condition = torch.tensor([[1.5, -0.5]] * 100)   # high logP, low SAS
smiles = model.generate(n=100, condition=condition)
```

---

## Full pipeline

### 1. Preprocess

```bash
python scripts/preprocess.py \
    --csv data/250k_rndm_zinc_drugs_clean_3.csv \
    --out data/labels.pkl \
    --properties logP SAS
```

Writes the label pickle plus a JSON retention report. Add `--strict` for
exact round-trip labels at lower retention.

The ZINC250K CSV used here is the standard
`250k_rndm_zinc_drugs_clean_3.csv` with `smiles`, `logP`, `qed`, and
`SAS` columns. Any CSV with a SMILES column and numeric property columns
works.

### 2. Train

Stages must be trained in order — each consumes what the previous one
conditions on.

```bash
python scripts/train.py --all \
    --labels data/labels.pkl \
    --ckpt-root checkpoints/
```

Or one stage at a time:

```bash
python scripts/train.py --stage a1 --labels data/labels.pkl \
    --ckpt-dir checkpoints/a1 --epochs 40 --capacity 3M
```

| Stage | Predicts | Default capacity | Default epochs |
|---|---|---|---|
| `a1` | ring layout | 3M | 40 |
| `a3` | branch topology | 10M | 100 |
| `a2` | atom identities | 10M | 40 |
| `terminal` | decoration | 9M | 40 |

All four **auto-resume** from `latest.pt`, so an interrupted run is
restarted by re-issuing the same command. This matters on hosted
notebooks with session limits — the full sequence takes longer than a
single session. To restart from scratch, delete the checkpoint directory.

Each stage writes `latest.pt`, `best_model.pt`, `ema.pt`,
`history.json`, and `config.json`.

### 3. Generate and evaluate

```bash
python scripts/generate.py \
    --ckpt-root checkpoints/ \
    --n 1000 \
    --labels data/labels.pkl \
    --csv data/250k_rndm_zinc_drugs_clean_3.csv \
    --out generated.csv --report metrics.json
```

Reports validity, uniqueness, novelty, and the Pearson correlation
between requested and achieved property values. `--labels` is needed for
novelty; `--csv` for the property statistics used to z-score conditions
during training.

Generate at least 1000 samples for a reportable figure — smaller runs
overstate uniqueness.

### Notebooks

`notebooks/01_train.ipynb` and `notebooks/02_evaluate.ipynb` wrap the
same functions for hosted environments, with Drive mounting and
per-stage plots.

---

## Reproducing the numbers

```bash
# Encoder retention and round-trip fidelity
python scripts/validate_encoder.py \
    --csv data/250k_rndm_zinc_drugs_clean_3.csv \
    --full --report encoder_report.json
```

Runs the baseline encoder, the current encoder, and strict mode over the
same molecules, then checks round-trip fidelity on a subset. Takes about
ten minutes on the full file; pass `--sample 20000` for a quick check
that lands within a few tenths of a point.

---

## Repository layout

```
himodit/
├── chem/
│   ├── encoder.py            SMILES -> layout labels
│   ├── decoder.py            layout -> bond matrix (deterministic)
│   ├── compose.py            scaffold + terminals -> RDKit molecule
│   ├── terminal_smarts.py    K=22 functional-group vocabulary
│   └── terminal_detection.py terminal fragment matching
├── models/
│   ├── layers.py             shared DiT blocks, AdaLN, edge-biased attention
│   ├── ring_layout.py        A1
│   ├── branch_topology.py    A3
│   ├── ring_atom.py          A2
│   └── terminal_fragment.py  Terminal
├── training/
│   ├── common.py             EMA, LR schedule, checkpointing
│   └── a1.py a2.py a3.py terminal.py
├── pipeline.py               end-to-end generation
└── metrics.py                V.U.N and controllability

scripts/     preprocess, train, generate, validate_encoder
docs/        architecture, label_schema, limitations
tests/       110 tests, CPU only
```

---

## Design notes

**The decoder has no parameters.** Ring closure, fusion sharing exactly
two atoms, spiro sharing exactly one — all are properties of the
construction algorithm, not things the model has to learn and can get
wrong. This is what lets A2 treat the bond graph as a clean known input
rather than something to denoise jointly.

**Conditioning enters at every stage.** All four stages take the same
z-scored property vector through AdaLN-Zero modulation, so a property
target influences ring count, branch size, heteroatom placement, and
functional groups, not just the last of these.

**This is masked absorbing-state discrete diffusion**, in the D3PM and
MaskGIT line — not flow matching, despite earlier naming in this
project's history. Generation is not single-step: A1 and A2 each run 20
unmasking steps by default. It is fast because the token sequences are
short (21 and 40 tokens), not because it is one-shot.

More detail in [docs/architecture.md](docs/architecture.md).

---

## Limitations

Summarised here, measured and documented in
[docs/limitations.md](docs/limitations.md):

- Substituents on **linker atoms** are silently dropped (6.4% of accepted
  labels). Use `--strict` to reject them instead.
- A3 can emit **non-causal branch parents**, which cost the molecule at
  decode time. `--enforce-causal-parent` clamps them.
- **Peri-fused and angular** ring systems are rejected — the ring graph
  must be a tree.
- **Ring-free molecules** cannot be represented; the hierarchy is
  anchored on ring 0.
- Scaffold ring bonds are **single or aromatic only**; ring-internal
  double bonds are not expressible.
- Reported validity **includes the repair cascade** in
  `himodit.chem.compose`.

---

## Citation

```bibtex
@software{himodit,
  title  = {HiMoDiT: Hierarchical Molecular Diffusion Transformer},
  year   = {2026},
  url    = {[https://github.com/TODO/himodit](https://github.com/dongddaoist/HiMoDiT-release)}
}
```

See [CITATION.cff](CITATION.cff).

---

## License

MIT — see [LICENSE](LICENSE).
