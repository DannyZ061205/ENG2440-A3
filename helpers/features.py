"""What the integer columns of `x` and `edge_attr` mean, and how large each category vocabulary is.

PyTorch Geometric builds BBBP graphs with `torch_geometric.utils.smiles.from_smiles`, which writes one
integer per feature: the position of the atom's or bond's category in a fixed list (`x_map`, `e_map`).
Those integers are category codes. Atom 7 is not "one more" than atom 6, and bond type 12 (aromatic) is
not four times bond type 3 (triple), so the codes must never enter the model as numbers.
"""
import pandas as pd
import torch
from torch_geometric.utils.smiles import e_map, x_map

ATOM_COLUMNS = list(x_map)            # 9 atom features, in the order they appear in `x`
BOND_COLUMNS = list(e_map)            # 3 bond features, in the order they appear in `edge_attr`

ATOM_VOCAB = [len(x_map[k]) for k in ATOM_COLUMNS]
BOND_VOCAB = [len(e_map[k]) for k in BOND_COLUMNS]

MEANING = {
    "atomic_num": "chemical element (code = atomic number)",
    "chirality": "local three-dimensional arrangement at the atom",
    "degree": "number of bonded neighbors in the graph",
    "formal_charge": "assigned charge (code 5 is neutral)",
    "num_hs": "attached hydrogens, which the graph does not store as nodes",
    "num_radical_electrons": "unpaired electrons",
    "hybridization": "orbital hybridization (sp, sp2, sp3, ...)",
    "is_aromatic": "atom belongs to an aromatic ring",
    "is_in_ring": "atom belongs to any ring",
    "bond_type": "single, double, triple or aromatic bond",
    "stereo": "double-bond stereochemistry (none, Z, E, ...)",
    "is_conjugated": "bond is part of a conjugated system",
}


def _describe(columns, vocab, maps, values):
    rows = []
    for position, (name, size) in enumerate(zip(columns, vocab)):
        seen = sorted(set(values[:, position].tolist()))
        names = [str(maps[name][v]) for v in seen]
        rows.append({
            "column": position,
            "feature": name,
            "meaning": MEANING[name],
            "vocabulary": size,
            "codes seen": len(seen),
            "categories seen": ", ".join(names[:4]) + (", ..." if len(names) > 4 else ""),
        })
    return pd.DataFrame(rows)


def describe_atom_features(dataset):
    values = torch.cat([g.x for g in dataset])
    return _describe(ATOM_COLUMNS, ATOM_VOCAB, x_map, values)


def describe_bond_features(dataset):
    values = torch.cat([g.edge_attr for g in dataset])
    return _describe(BOND_COLUMNS, BOND_VOCAB, e_map, values)
