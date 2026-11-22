# ENG 2440 Assignment 3 — Blood–brain barrier penetration from molecular graphs

Zichen Zhao (25011189) · ENG 2440 Medical Imaging & AI in Healthcare

A bond-aware graph neural network and a Morgan-fingerprint baseline are compared on the MoleculeNet
**BBBP** dataset, under a stratified random split and a Bemis–Murcko scaffold split.

**Headline result.** The graph network does not beat the fingerprint baseline (test ROC-AUC 0.919 vs
0.903 ± 0.009 random, 0.876 vs 0.866 ± 0.011 scaffold — both gaps inside the uncertainty). The *splitting
strategy* moves the score far more than the model does: both models drop 0.037–0.043 ROC-AUC under the
scaffold split, three to four times the seed-to-seed spread.

## Contents

| Path | What it is |
|---|---|
| `25011189_ENG2440_A3.ipynb` | The executed notebook, outputs visible. Runs top to bottom. |
| `helpers/` | Supporting modules imported by the notebook (see below). |
| `report/main.tex`, `report/tables/` | Report source; tables are generated from `outputs/tables/`. |
| `figures/` | Figure PDFs the report includes, plus `methods.json` (the color/style map). |
| `outputs/tables/` | CSVs written by the notebook; the report's tables are built from these. |
| `requirements.txt` | Pinned environment. |
| `ai_use_declaration.txt` | AI-use declaration. |

### `helpers/`

| Module | Responsibility |
|---|---|
| `data.py` | Loading, canonical-SMILES cleaning, scaffolds, both split strategies, split verification. |
| `features.py` | Meaning and vocabulary size of each categorical atom/bond column. |
| `models.py` | `MolecularGNN`: embedding encoders, GINEConv message passing, pooling, head. |
| `train.py` | Training loop, seeding, early stopping on validation ROC-AUC, prediction. |
| `baseline.py` | Morgan fingerprints, Tanimoto similarity, logistic-regression baseline. |
| `metrics.py` | Metrics, the validation threshold rule, bootstrap confidence intervals. |
| `plotting.py` | Molecule drawing, graph view, confusion panels, tables. |
| `paper_plot_style.py` | House figure style and the automatic figure audit. |
| `checklist.py` | Offline audit of the executed notebook against every requirement. |

## Run order

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab 25011189_ENG2440_A3.ipynb     # then Run All, top to bottom
```

The notebook is self-contained and runs in one pass. On first run PyTorch Geometric downloads BBBP into
`data/MoleculeNet/` (internet needed once; `data/` is gitignored and must not be committed). Everything
after that is local. Expect roughly 5 minutes end to end on a laptop CPU.

To rebuild the report after re-running the notebook:

```bash
python report/make_tables.py && tectonic report/main.tex
```

To re-audit the notebook and the report against the brief:

```bash
python -m helpers.checklist 25011189_ENG2440_A3.ipynb report/main.tex
```

This confirms the notebook was executed top to bottom without errors, that every requirement has evidence
in it, that all eight figures saved and passed the figure audit, and that every decimal number quoted in
the notebook prose or the report also appears in a code output.

## Reproducibility

- **One seed (0)** fixes both splits, the baseline, and seeds 0–2 for the three graph-network runs.
- **Training runs on the CPU deliberately.** At this size the CPU is faster than the Apple-silicon GPU
  (13 s vs 20 s for 40 epochs) *and* exactly reproducible — GPU scatter-add accumulates messages in a
  non-deterministic order, so the same seed gives slightly different numbers each run.
- **RDKit is pinned to 2025.3.6.** In 2026.09.2 `MurckoDecompose` raises `RuntimeError: bad any cast` on
  every molecule, which breaks the scaffold split entirely. Do not upgrade without re-checking that.
- Each test subset is read exactly once, in Section F. Every hyperparameter and the decision threshold
  are chosen on validation and frozen beforehand.

## Notes

- `data/` holds the downloaded MoleculeNet files and is never committed.
- Figures are drawn at their final printed width (6.693 in, the report's measured `\textwidth`) and are
  included at that width, never rescaled, so font sizes in figures match the body text.
