"""Offline requirement audit for ENG2440 Assignment 3.

Run it from a terminal after executing the notebook:

    python -m helpers.checklist 25011189_ENG2440_A3.ipynb [report/main.tex]

It re-reads the saved .ipynb and checks that
* the notebook was executed top to bottom in one run, without errors,
* every requirement of Section 3 of the brief has its evidence in the notebook: a printed output, a piece
  of code, a heading or a written answer,
* every expected figure was saved and passed the automatic figure audit, and
* decimal numbers quoted in the notebook markdown, and (if a report .tex is given) in the report, also
  appear in some code output -- a guard against stale or invented numbers.
"""

import json
import re
import sys
from pathlib import Path

# (id, marks, requirement, [evidence]); evidence items are ("md" | "code" | "out" | "fig", text to find)
RUBRIC = [
    ("A1", 2, "Input, output, unit of analysis, intended technical use; what the labels mean",
     [("md", "**Input.**"), ("md", "**Output.**"), ("md", "**Unit of analysis.**"),
      ("md", "**Intended technical use.**"), ("md", "**What the labels mean.**")]),
    ("A2", 3, "Size, missing/non-finite labels, invalid/empty graphs, class counts and proportions; "
              "canonicalize, find duplicates incl. different original SMILES, check label consistency",
     [("out", "distributed CSV"), ("out", "missing labels"), ("out", "non-finite labels"),
      ("out", "class counts"), ("out", "class proportions"), ("out", "graphs with no atom"),
      ("out", "unique canonical molecules"), ("out", "molecules appearing more than once"),
      ("out", "canonical molecules with both labels"), ("out", "written as:"),
      ("code", "data.clean(dataset)")]),
    ("A3", 2, ">= 6 structures from both classes, each with its SMILES and label",
     [("fig", "A_molecules"), ("out", "penetrating      p_np=1"), ("out", "non-penetrating  p_np=0")]),
    ("A4", 3, "Atom- and bond-count distributions; one graph's x/edge_index/edge_attr/y/smiles",
     [("fig", "A_sizes"), ("out", "atoms per molecule"), ("out", "x.shape"), ("out", "edge_index.shape"),
      ("out", "edge_attr.shape"), ("out", "smiles          :")]),
    ("CLEAN", 0, "Cleaning rule applied before splitting, with counts reported",
     [("md", "**The cleaning rule.**"), ("out", "after cleaning :")]),
    ("B1", 3, "Representation explained; every categorical column identified; encoding stated",
     [("out", "Atom feature columns in x"), ("out", "Bond feature columns in edge_attr"),
      ("md", "**How the codes are encoded.**"), ("md", "trainable embedding"),
      ("out", "embedded atom vectors")]),
    ("B2", 3, "Same molecule as chemical structure and node-edge graph, correspondence explained",
     [("fig", "B_views"), ("md", "**How the two views line up.**"), ("out", "directed edges")]),
    ("B3", 3, "DataLoader; shapes of one batch explained",
     [("code", "DataLoader("), ("out", "batch                :"), ("out", "batch.shape"),
      ("out", "y.shape")]),
    ("B4", 3, "Checks: finite labels, >=1 atom per graph, labels match graph count, nothing lost",
     [("code", "assert torch.isfinite(y).all()"), ("code", "assert molecules_seen == n_molecules"),
      ("out", "all 16 batches pass")]),
    ("C1", 4, "70:15:15 stratified random and Bemis-Murcko scaffold splits, one fixed seed; "
              "same scaffold in one subset; acyclic/invalid handled",
     [("code", "data.stratified_random_split"), ("code", "data.scaffold_split"),
      ("md", "**Molecules with no ring system**"), ("out", "molecules with no ring system")]),
    ("C2", 3, "Coverage, pairwise-disjoint indices, no shared canonical SMILES",
     [("out", "covers_every_molecule = True"), ("out", "pairwise_disjoint_indices = True"),
      ("out", "no_shared_canonical_smiles = True")]),
    ("C3", 3, "Counts and positive proportions per subset; scaffold counts and disjoint scaffolds",
     [("out", "positive_proportion"), ("out", "disjoint_scaffolds = True"),
      ("out", "unique scaffolds per subset")]),
    ("C4", 3, "Test sets untouched until final evaluation; why a random split is optimistic",
     [("md", "### C.4 Why a random split flatters a model"), ("md", "**The test sets stay closed"),
      ("out", "test molecules with a training neighbor above Tanimoto 0.6"), ("fig", "C_splits")]),
    ("D1", 3, "Morgan/ECFP fingerprints; radius and length stated and explained",
     [("md", "### D.1 What a Morgan fingerprint represents"), ("out", "radius             : 2"),
      ("out", "fingerprint matrix :")]),
    ("D2", 3, "Logistic regression on training data only; imbalance handling estimated on training only",
     [("md", "### D.2 Fitting the classifier on training data only"), ("md", "class_weight"),
      ("code", "baseline.fit_baseline")]),
    ("D3", 2, "Validation selects hyperparameters and one threshold, both frozen before test",
     [("out", "chosen C          :"), ("out", "chosen threshold  :"), ("md", "frozen")]),
    ("D4", 2, "Seed and all settings reported, for both split strategies",
     [("out", "Baseline settings, frozen before any test evaluation")]),
    ("E1", 5, "Bond-aware MPNN, >=2 layers, pooling, binary head; encodings and parameter count",
     [("code", "GINEConv"), ("out", "trainable parameters    :"), ("out", "node features encoded as"),
      ("out", "edge features encoded as"), ("out", "pooling and head")]),
    ("E2", 5, "Loss, optimiser, LR, batch size, epochs, imbalance, stopping rule justified",
     [("md", "| loss |"), ("md", "| optimiser |"), ("md", "| learning rate |"), ("md", "| batch size |"),
      ("md", "| epoch budget |"), ("md", "| early stopping |"), ("md", "pos_weight")]),
    ("E3", 4, "Training and validation loss plus the validation metric; curves interpreted",
     [("fig", "E_curves"), ("md", "**Reading the curves.**")]),
    ("E4", 4, "Three seeds per split; mean and SD; seed variation vs split effect",
     [("out", "validation ROC-AUC, mean +/- SD over three seeds"),
      ("out", "spread between the two splits"), ("out", "largest spread within a split")]),
    ("F1", 4, "ROC-AUC, PR-AUC, accuracy, sensitivity, specificity, F1 for both models and both splits, "
              "at validation-selected thresholds",
     [("out", "fingerprint : ROC-AUC"), ("out", "GNN         : ROC-AUC"), ("code", "metrics.scores_at")]),
    ("F2", 3, "ROC and PR curves and a confusion matrix for each final test evaluation",
     [("fig", "F_curves"), ("fig", "F_confusion")]),
    ("F3", 3, "One comparison table: model, split, threshold, all metrics; mean and SD for the GNN",
     [("out", "Test results. GNN entries are mean")]),
    ("F4", 4, "Baseline vs GNN and random vs scaffold compared; claims matched to evidence",
     [("md", "**The graph network does not beat the fingerprint baseline.**"),
      ("md", "**Changing the split moves the score more than changing the model does.**"),
      ("md", "**What this does not show.**"), ("out", "test ROC-AUC, GNN minus fingerprint")]),
    ("G1", 2, ">= 2 false positives and 2 false negatives with labels and predicted probabilities",
     [("fig", "G_errors"), ("out", "false positives:"), ("out", "false negatives:")]),
    ("G2", 3, "Nearest training molecule by Tanimoto for each error; pair, similarity, labels shown",
     [("out", "nearest training molecule"), ("out", "Tanimoto"),
      ("md", "**Does similarity explain the errors?**")]),
    ("G3", 3, ">= 3 limitations incl. label uncertainty, dataset size, no external validation; "
              "no causal or clinical claim",
     [("md", "**The label is uncertain"), ("md", "**The dataset is small"),
      ("md", "**Nothing here was validated outside this dataset.**"),
      ("md", "**Why none of this establishes causality or clinical suitability.**")]),
]

EXPECTED_FIGURES = ["A_molecules", "A_sizes", "B_views", "C_splits", "E_curves", "F_curves",
                    "F_confusion", "G_errors"]
NUMBER_RE = re.compile(r"(?<![\w.])(\d+\.\d{2,})(?![\w.])")      # numbers with 2+ decimals


def _text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _outputs(cell):
    chunks = []
    for out in cell.get("outputs", []):
        if "text" in out:
            chunks.append("".join(out["text"]) if isinstance(out["text"], list) else out["text"])
        for key in ("text/plain", "text/html", "text/markdown"):
            data = out.get("data", {}).get(key)
            if data:
                chunks.append("".join(data) if isinstance(data, list) else data)
    return "\n".join(chunks)


def _found(num, outputs):
    if num in outputs:
        return True
    d = len(num.split(".")[1])
    target = float(num)
    for m in re.finditer(r"-?\d+\.\d+", outputs):
        v = m.group(0)
        if len(v.split(".")[1]) > d and abs(round(float(v), d) - target) < 1e-9:
            return True                                        # rounded from a longer printed number
        if abs(round(abs(float(v)) * 100, d) - target) < 1e-9 and len(v.split(".")[1]) >= d + 2:
            return True                                        # a fraction quoted as a percentage
    return False


def audit(nb_path, report=None, fig_dir="figures"):
    cells = json.loads(Path(nb_path).read_text())["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    md = "\n\n".join(_text(c) for c in cells if c["cell_type"] == "markdown")
    src = "\n\n".join(_text(c) for c in code)
    out = "\n\n".join(_outputs(c) for c in code)
    figs = {n for n in EXPECTED_FIGURES if (Path(fig_dir) / f"{n}.pdf").exists()}
    problems, warnings = [], []

    counts = [c.get("execution_count") for c in code if _text(c).strip()]
    if any(n is None for n in counts):
        problems.append("some code cells were never executed")
    elif counts != list(range(1, len(counts) + 1)):
        problems.append("execution counts are not 1..N: the notebook was not run top to bottom in one go")
    for c in code:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                problems.append(f"error output: {o.get('ename')}: {o.get('evalue', '')[:120]}")

    where = {"md": md, "code": src, "out": out}
    met = 0
    for rid, marks, desc, evidence in RUBRIC:
        missing = [text for kind, text in evidence
                   if (kind == "fig" and text not in figs) or (kind != "fig" and text not in where[kind])]
        if missing:
            problems.append(f"[{rid}] {desc[:60]}: missing {missing}")
        else:
            met += 1

    for name in EXPECTED_FIGURES:
        if name not in figs:
            problems.append(f"missing figure {fig_dir}/{name}.pdf")
        log = Path(fig_dir) / "_audit" / f"{name}.json"
        if log.exists():
            errors = [i for i in json.loads(log.read_text())["issues"] if i["severity"] == "error"]
            if errors:
                problems.append(f"figure {name}: {len(errors)} figure-audit error(s)")

    texts = [("notebook markdown", md)]
    if report:
        tex = Path(report).read_text()
        # Only the prose is checked: the preamble holds layout constants (float fractions, the measured
        # figure width) and the bibliography holds volume, page and DOI numbers -- none are claims.
        if r"\begin{document}" in tex:
            tex = tex[tex.find(r"\begin{document}"):]
        tex = tex[:tex.find(r"\begin{thebibliography}")] if r"\begin{thebibliography}" in tex else tex
        texts.append((Path(report).name, tex))
    for name, text in texts:
        text = re.sub(r"doi:\S+|arXiv:\S+|\d+\.\d+\\,?(in|cm|pt)\b|\d{4}\.\d+", "", text)
        for num in sorted(set(NUMBER_RE.findall(text))):
            if not _found(num, out):
                warnings.append(f"{name}: number {num} not found in any code output")
        for placeholder in ("TODO", "TBD", "XXX", "???"):
            if placeholder in text:
                problems.append(f"{name} still contains placeholder {placeholder!r}")

    print(f"Audit of {nb_path}" + (f" and {report}" if report else ""))
    print(f"  code cells executed        : {len(counts)}")
    print(f"  requirements with evidence : {met}/{len(RUBRIC)}")
    print(f"  figures present            : {len(figs)}/{len(EXPECTED_FIGURES)}")
    for w in warnings:
        print("  WARN ", w)
    for p in problems:
        print("  FAIL ", p)
    print("RESULT:", "PASS" if not problems else f"FAIL ({len(problems)} problems)")
    return not problems


if __name__ == "__main__":
    ok = audit(sys.argv[1] if len(sys.argv) > 1 else "25011189_ENG2440_A3.ipynb",
               sys.argv[2] if len(sys.argv) > 2 else None)
    sys.exit(0 if ok else 1)
