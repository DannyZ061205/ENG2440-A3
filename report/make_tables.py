"""Generate the report's LaTeX tables from the notebook's saved CSVs.

Run from the project root after executing the notebook:

    python report/make_tables.py

Writing the tables from the same files the notebook saved means no number is ever retyped by hand.
"""
from pathlib import Path

import pandas as pd

TABLES = Path("outputs/tables")
OUT = Path("report/tables")
OUT.mkdir(parents=True, exist_ok=True)


def esc(text):
    return str(text).replace("±", r"$\pm$").replace("_", r"\_").replace("%", r"\%")


def comparison_table():
    df = pd.read_csv(TABLES / "comparison.csv")
    df = df.sort_values(["split", "model"], ascending=[True, False]).reset_index(drop=True)
    # n is the same for every row of a split, so it goes in the caption; the column pushed the
    # table past the right margin.
    lines = [r"\begin{tabular}{llccccccc}", r"\toprule",
             r"Model & Split & Thresh. & ROC-AUC $\uparrow$ & PR-AUC $\uparrow$ & Acc. $\uparrow$ & "
             r"Sens. $\uparrow$ & Spec. $\uparrow$ & F1 $\uparrow$ \\", r"\midrule"]
    for _, r in df.iterrows():
        name = "Fingerprint" if r["model"] == "fingerprint" else "GNN"
        threshold = f"{float(r['threshold']):.3f}" if str(r["threshold"])[0].isdigit() and "±" not in str(r["threshold"]) else esc(r["threshold"])
        lines.append(f"{name} & {r['split']} & {threshold} & {esc(r['ROC-AUC'])} & {esc(r['PR-AUC'])} & "
                     f"{esc(r['accuracy'])} & {esc(r['sensitivity'])} & {esc(r['specificity'])} & "
                     f"{esc(r['F1'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "comparison.tex").write_text("\n".join(lines) + "\n")


def splits_table():
    df = pd.read_csv(TABLES / "splits.csv")
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Split & Subset & Molecules & Positive & Unique scaffolds \\", r"\midrule"]
    for _, r in df.iterrows():
        scaffolds = "--" if pd.isna(r["unique_scaffolds"]) else f"{int(r['unique_scaffolds'])}"
        lines.append(f"{r['split']} & {r['subset']} & {int(r['molecules'])} & "
                     f"{int(r['positives'])} ({r['positive_proportion']:.3f}) & {scaffolds} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "splits.tex").write_text("\n".join(lines) + "\n")


def errors_table():
    df = pd.read_csv(TABLES / "errors.csv")
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Case & Error & True & Predicted $p$ & Nearest train label & Tanimoto \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{esc(r['case'])} & {r['error']} & {int(r['true label'])} & {r['predicted p']:.3f} & "
                     f"{int(r['its label'])} & {r['Tanimoto']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "errors.tex").write_text("\n".join(lines) + "\n")


def features_table():
    atoms = pd.read_csv(TABLES / "atom_features.csv")
    bonds = pd.read_csv(TABLES / "bond_features.csv")
    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r"Column & Meaning & Vocabulary & Codes seen \\", r"\midrule",
             r"\multicolumn{4}{l}{\itshape Atom features, \texttt{x}} \\"]
    for _, r in atoms.iterrows():
        lines.append(f"\\texttt{{{esc(r['feature'])}}} & {esc(r['meaning'])} & {r['vocabulary']} & {r['codes seen']} \\\\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{4}{l}{\itshape Bond features, \texttt{edge\_attr}} \\")
    for _, r in bonds.iterrows():
        lines.append(f"\\texttt{{{esc(r['feature'])}}} & {esc(r['meaning'])} & {r['vocabulary']} & {r['codes seen']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "features.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    comparison_table()
    splits_table()
    errors_table()
    features_table()
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables to {OUT}")
