"""Loading, cleaning and splitting the BBBP molecules.

The PyTorch Geometric `MoleculeNet` loader silently drops SMILES that RDKit cannot parse, so the raw CSV is
read as well: that is the only way to report how many records the brief's "invalid molecular graphs" check
actually removes. Everything downstream works on indices into the PyG dataset, so a molecule's graph, its
label and its SMILES never drift apart.

Nothing here looks at a test subset: the cleaning rule and both split rules depend only on the molecules
themselves, never on model behavior.
"""
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")           # RDKit prints a parse error per bad SMILES; we count them instead


def canonical(smiles):
    """RDKit canonical SMILES, or None when the string cannot be parsed into a non-empty molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def audit_raw_csv(path):
    """Counts the brief asks for, taken from the distributed CSV before PyG touches it."""
    df = pd.read_csv(path)
    labels = pd.to_numeric(df["p_np"], errors="coerce")
    can = [canonical(s) for s in df["smiles"]]
    return {
        "records": len(df),
        "columns": list(df.columns),
        "missing_labels": int(labels.isna().sum()),
        "non_binary_labels": int((~labels.dropna().isin([0, 1])).sum()),
        "unparseable_smiles": sum(c is None for c in can),
        "valid_graphs": sum(c is not None for c in can),
    }


def clean(dataset):
    """Apply the brief's cleaning rule to a PyG dataset; return a per-record table and the retained indices.

    One record per canonical molecule is kept when its duplicates agree on the label. When a canonical
    molecule carries both labels, every record in that group is removed: there is no way to tell which
    label is right, and keeping either would put a wrong label in the data.
    """
    rows = []
    for i, graph in enumerate(dataset):
        rows.append({
            "index": i,
            "smiles": graph.smiles,
            "canonical": canonical(graph.smiles),
            "y": int(graph.y.view(-1)[0].item()),
            "n_atoms": graph.num_nodes,
            "n_bonds": graph.edge_index.shape[1] // 2,
        })
    df = pd.DataFrame(rows)

    df["valid"] = df["canonical"].notna() & (df["n_atoms"] > 0)
    valid = df[df["valid"]]
    group = valid.groupby("canonical")["y"]
    counts, distinct = group.size(), group.nunique()

    conflicting = set(distinct[distinct > 1].index)
    df["conflicting"] = df["canonical"].isin(conflicting)
    # Keep the first record of each consistent group; mark the rest as duplicates.
    keep_rows = valid[~valid["canonical"].isin(conflicting)].drop_duplicates("canonical")
    df["kept"] = df["index"].isin(set(keep_rows["index"]))

    summary = {
        "loaded": len(df),
        "invalid_or_empty": int((~df["valid"]).sum()),
        "unique_canonical": int(counts.size),
        "molecules_seen_more_than_once": int((counts > 1).sum()),
        "duplicate_records": int((counts - 1).sum()),
        "conflicting_groups": len(conflicting),
        "records_in_conflicting_groups": int(df["conflicting"].sum()),
        "kept": int(df["kept"].sum()),
    }
    return df, summary


def duplicate_examples(df, n=3):
    """A few canonical molecules that appear more than once, for display."""
    valid = df[df["valid"]]
    counts = valid.groupby("canonical")["y"].agg(["size", "nunique"])
    repeated = counts[counts["size"] > 1].sort_values("size", ascending=False)
    out = []
    for can in list(repeated.index)[:n]:
        g = valid[valid["canonical"] == can]
        out.append({"canonical": can, "records": len(g), "labels": sorted(set(g["y"])),
                    "original_smiles": list(g["smiles"])})
    return out


def scaffold_of(smiles, include_chirality=False):
    """Bemis--Murcko scaffold as canonical SMILES; '' for a molecule with no ring system."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_table(canonical_smiles):
    """Scaffold per molecule plus the counts the report quotes."""
    scaffolds = [scaffold_of(s) for s in canonical_smiles]
    counts = Counter(scaffolds)
    return scaffolds, {
        "molecules": len(scaffolds),
        "unique_scaffolds": len(counts),
        "acyclic_molecules": counts.get("", 0),
        "scaffolds_with_one_molecule": sum(1 for v in counts.values() if v == 1),
        "largest_scaffold_size": max(counts.values()),
    }


def stratified_random_split(labels, fractions=(0.70, 0.15, 0.15), seed=0):
    """Class-proportional 70:15:15 split of molecule positions, shuffled with one fixed seed."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train, val, test = [], [], []
    for value in np.unique(labels):                       # stratify: split each class with the same ratios
        idx = np.flatnonzero(labels == value)
        rng.shuffle(idx)
        n_train = int(round(fractions[0] * len(idx)))
        n_val = int(round(fractions[1] * len(idx)))
        train += list(idx[:n_train])
        val += list(idx[n_train:n_train + n_val])
        test += list(idx[n_train + n_val:])
    return tuple(np.sort(np.asarray(part, dtype=int)) for part in (train, val, test))


def scaffold_split(scaffolds, fractions=(0.70, 0.15, 0.15), seed=0):
    """Bemis--Murcko split: a scaffold's molecules always stay together.

    Groups are filled largest-first so the big scaffold families land in training and the held-out subsets
    are made of scaffolds the model never saw; ties are broken with the fixed seed so the split is
    reproducible. Molecules with no ring system share the empty scaffold and are treated as one group,
    which keeps the rule "same scaffold, same subset" true for them as well.
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for position, scaffold in enumerate(scaffolds):
        groups.setdefault(scaffold, []).append(position)
    order = sorted(groups.values(), key=lambda g: (-len(g), rng.random()))

    n = len(scaffolds)
    limits = (fractions[0] * n, (fractions[0] + fractions[1]) * n)
    train, val, test = [], [], []
    for group in order:
        if len(train) + len(group) <= limits[0]:
            train += group
        elif len(train) + len(val) + len(group) <= limits[1]:
            val += group
        else:
            test += group
    return tuple(np.sort(np.asarray(part, dtype=int)) for part in (train, val, test))


def verify_split(parts, canonical_smiles, n_molecules, scaffolds=None):
    """Coverage, pairwise disjointness, no shared canonical SMILES, and (optionally) disjoint scaffolds."""
    names = ("train", "validation", "test")
    sets = [set(p.tolist()) for p in parts]
    assert sum(len(p) for p in parts) == n_molecules, "the three subsets do not cover every molecule"
    assert set().union(*sets) == set(range(n_molecules)), "some molecule is in no subset"
    for a in range(3):
        for b in range(a + 1, 3):
            assert not sets[a] & sets[b], f"{names[a]} and {names[b]} share molecules"
    smiles_sets = [{canonical_smiles[i] for i in p} for p in parts]
    for a in range(3):
        for b in range(a + 1, 3):
            shared = smiles_sets[a] & smiles_sets[b]
            assert not shared, f"{names[a]} and {names[b]} share {len(shared)} canonical SMILES"
    result = {"covers_every_molecule": True, "pairwise_disjoint_indices": True, "no_shared_canonical_smiles": True}
    if scaffolds is not None:
        scaffold_sets = [{scaffolds[i] for i in p} for p in parts]
        for a in range(3):
            for b in range(a + 1, 3):
                shared = scaffold_sets[a] & scaffold_sets[b]
                assert not shared, f"{names[a]} and {names[b]} share {len(shared)} scaffolds"
        result["disjoint_scaffolds"] = True
        result["unique_scaffolds_per_subset"] = {n: len(s) for n, s in zip(names, scaffold_sets)}
    return result


def split_summary(parts, labels, scaffolds=None):
    """Molecule count, positive count and positive proportion per subset (plus scaffold counts)."""
    labels = np.asarray(labels)
    rows = []
    for name, part in zip(("train", "validation", "test"), parts):
        y = labels[part]
        row = {"subset": name, "molecules": len(part), "positives": int(y.sum()),
               "positive_proportion": float(y.mean())}
        if scaffolds is not None:
            row["unique_scaffolds"] = len({scaffolds[i] for i in part})
        rows.append(row)
    return pd.DataFrame(rows)
