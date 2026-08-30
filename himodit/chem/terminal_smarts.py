"""
Terminal fragment vocabulary (K = 22).
======================================

The functional groups the Terminal stage can graft onto a scaffold. Each
entry carries a detection SMARTS used during encoding and an id that
maps to a construction spec in `himodit.chem.compose` (model class =
id + 1, since class 0 means "no decoration").

`allowed_attachment_bonds` restricts which bond order may join the
fragment to its host: single-bonded substituents list [1], while
double-bonded ones (=O, =NH, =S) list [3].

Charge-form coverage
--------------------
OH, COOH, NH2, and SO3H each match both the neutral and the ionized
form, since ZINC250K stores many of these deprotonated or protonated.
NO2 uses the explicit three-atom zwitterion pattern rather than a
recursive one-atom SMARTS: the terminal test counts bonds crossing out
of the matched atom set, so the match has to enclose all three atoms for
the fragment boundary to be found correctly.

Halogens (Cl, Br, I) appear only here and not in the scaffold atom
vocabulary. In ZINC250K every halogen sits in a terminal position, so a
scaffold vocabulary slot for them would never be used.
"""
from __future__ import annotations

CURATED_TERMINALS = [
    # ─── Core substituents (IDs 0-5) ───────────────────────────────────
    {"name": "OH", "detection_smarts": "[OX2H1,OX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 0,
     "notes": "Matches both -OH and alkoxide -[O-]."},
    {"name": "COOH", "detection_smarts": "[CX3](=[OX1])[OX2H1,OX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 1,
     "notes": "Matches both -C(=O)OH and carboxylate -C(=O)[O-]."},
    {"name": "NH2", "detection_smarts": "[NX3H2,NX4H3+]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 2,
     "notes": "Matches both -NH2 and protonated -[NH3+]."},
    {"name": "SO3H", "detection_smarts": "[SX4](=[OX1])(=[OX1])[OX2H1,OX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 3,
     "notes": "Matches both -S(=O)(=O)OH and sulfonate -S(=O)(=O)[O-]."},
    {"name": "F", "detection_smarts": "[F]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 4},
    {"name": "CH3", "detection_smarts": "[CX4H3]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 5},

    # ─── the earlier encoder double-bond entries (IDs 6-8) ──────────────────────────────
    {"name": "=O", "detection_smarts": "[OX1]",
     "allowed_attachment_bonds": [3], "flags": [], "v5_3_id": 6,
     "notes": "Carbonyl oxygen. Common in quinones, ketones."},
    {"name": "=NH", "detection_smarts": "[ND1]",
     "allowed_attachment_bonds": [3], "flags": [], "v5_3_id": 7,
     "notes": "Imine nitrogen. [ND1] correctly rejects internal "
              "imines R-N=R\' (D=2)."},
    {"name": "=S", "detection_smarts": "[SX1]",
     "allowed_attachment_bonds": [3], "flags": [], "v5_3_id": 8,
     "notes": "Thione sulfur."},

    # ─── the earlier encoder additions (IDs 9-15) ──────────────────────────────────
    {"name": "Cl", "detection_smarts": "[Cl]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 9,
     "notes": "Chlorine. Single-atom terminal."},
    {"name": "Br", "detection_smarts": "[Br]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 10,
     "notes": "Bromine. Single-atom terminal."},
    {"name": "I", "detection_smarts": "[I]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 11,
     "notes": "Iodine. Single-atom terminal."},
    {"name": "CN", "detection_smarts": "[CX2]#[NX1]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 12,
     "notes": "Nitrile group. 2-atom terminal."},
    # Non-recursive three-atom form, so the crossing-bond
    # logic in _is_terminal_match correctly identifies the fragment boundary.
    {"name": "NO2", "detection_smarts": "[NX3+](=[OX1])[OX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 13,
     "notes": "Three-atom SMARTS. A one-atom recursive form would not "
              "Matches RDKit canonical zwitterion [N+](=O)[O-] form."},
    {"name": "OCH3", "detection_smarts": "[OX2][CX4H3]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 14,
     "notes": "Methoxy group. 2-atom terminal."},
    {"name": "CF3", "detection_smarts": "[CX4]([F])([F])[F]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 15,
     "notes": "Trifluoromethyl group. 4-atom terminal."},

    # ─── Rarer groups (IDs 16-21) ──────────────────────────────────────
    {"name": "Thiol", "detection_smarts": "[SX2H][#6]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 16,
     "notes": "thiol -SH attached to scaffold C."},
    {"name": "AcylHalide", "detection_smarts": "[CX3](=[OX1])[F,Cl,Br,I]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 17,
     "notes": "acyl halide -C(=O)X. Default emit -C(=O)Cl."},
    {"name": "Cyanate", "detection_smarts": "[OX2][CX2]#[NX1]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 18,
     "notes": "cyanate -OC#N."},
    {"name": "Thiocyanate", "detection_smarts": "[SX2][CX2]#[NX1]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 19,
     "notes": "thiocyanate -SC#N."},
    {"name": "Isothiocyanate", "detection_smarts": "[NX2]=[CX2]=[SX1]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 20,
     "notes": "isothiocyanate -N=C=S."},
    {"name": "Isonitrile", "detection_smarts": "[NX2+]#[CX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 21,
     "notes": "isonitrile -[N+]#[C-]."},
]


def _validate_smarts():
    from rdkit import Chem
    failed = []
    for t in CURATED_TERMINALS:
        patt = Chem.MolFromSmarts(t["detection_smarts"])
        if patt is None:
            failed.append((t["name"], t["detection_smarts"]))
    if failed:
        msg = "Curated SMARTS failed to compile:\n"
        for name, smarts in failed:
            msg += f"  {name}: {smarts}\n"
        raise RuntimeError(msg)


_validate_smarts()
