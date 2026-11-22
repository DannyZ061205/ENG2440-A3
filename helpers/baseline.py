"""The non-neural comparison: Morgan (ECFP) fingerprints with logistic regression.

A Morgan fingerprint describes a molecule by the substructures around its atoms. For every atom the
generator collects the circular neighborhood out to `radius` bonds, hashes each distinct neighborhood to
an integer, and sets that bit. With radius 2 the neighborhoods reach two bonds from each atom, which is
the ECFP4 convention. The result is a fixed-length bit vector: bit k is on when the molecule contains at
least one substructure that hashes to k, so the vector says which fragments are present, not where they
are or how they connect.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def generator(radius=2, n_bits=2048):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def fingerprints(canonical_smiles, radius=2, n_bits=2048):
    """One row per molecule, one column per bit."""
    gen = generator(radius, n_bits)
    out = np.zeros((len(canonical_smiles), n_bits), dtype=np.uint8)
    for i, smiles in enumerate(canonical_smiles):
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(smiles)), out[i])
    return out


def tanimoto(a, b):
    """Tanimoto similarity between two bit vectors: shared bits / bits set in either."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def nearest_training_molecule(query_bits, train_bits):
    """Index of the most similar training molecule and its Tanimoto similarity."""
    shared = train_bits @ query_bits
    union = train_bits.sum(1) + query_bits.sum() - shared
    similarity = np.divide(shared, union, out=np.zeros(len(train_bits), float), where=union > 0)
    best = int(np.argmax(similarity))
    return best, float(similarity[best])


def fit_baseline(x_train, y_train, x_val, y_val, grid=(0.01, 0.1, 1.0, 10.0), seed=0, max_iter=5000):
    """Pick the regularization strength on validation ROC-AUC, then refit nothing: the model is kept as is.

    Class weighting is `balanced`, which scales each class by the inverse of its frequency in the data it
    is fitted on. That is the training subset only, so no validation or test label influences the weights.
    """
    chosen, results = None, []
    for c in grid:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter, random_state=seed)
        model.fit(x_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
        results.append({"C": c, "val_roc_auc": auc})
        if chosen is None or auc > chosen["val_roc_auc"]:
            chosen = {"C": c, "val_roc_auc": auc, "model": model}
    return chosen, results
