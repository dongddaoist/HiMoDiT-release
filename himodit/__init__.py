"""HiMoDiT - hierarchical molecular diffusion transformer.

A property-conditioned generative model for drug-like molecules, built as
a four-stage cascade over a chemically structured latent space:

    A1        condition            -> ring layout (types, fusion, linkers)
    A3        layout + condition   -> branch topology (side-chain trees)
    A2        scaffold + condition -> atom identities
    Terminal  scaffold + condition -> functional-group decoration

Every stage is a discrete-diffusion or one-pass transformer with AdaLN
conditioning; the bond matrix between stages is built by a deterministic
decoder rather than sampled, so the stages cannot contradict each other.

Quick start
-----------
    from himodit.chem.encoder import extract_layout
    label, reason = extract_layout("CC(=O)Nc1ccc(O)cc1")

    from himodit.pipeline import HiMoDiT
    model = HiMoDiT.from_checkpoints("checkpoints/")
    smiles = model.generate(n=100)
"""

__version__ = "1.0.0"
