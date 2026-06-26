"""
Organ Cascade — the COMBINATORIAL test ("TF words").

v3 showed no SINGLE TF motif discriminates organs (each clusters ~equally in
every organ's super-enhancers). Miles's hypothesis: organ identity is in the
COMBINATION of motifs at an enhancer -- the cis-regulatory grammar / TF
collective. A lone motif is an ambiguous token; the enhancer is a sentence; the
combination is the word. (This is exactly the LLM thesis: meaning is
combinatorial + contextual, hence attention over combinations, not a lookup.)

Test: from the v3 motif-count matrix (612 SEs x 90 TFs), can a classifier read
an SE's ORGAN from its motif COMBINATION, when the best single motif cannot?
Primary features = COMPOSITION (row-normalized counts) so the model must use the
relative mix of TFs, not total motif load (which could encode length/GC/density).
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

from medic.organ_cascade_wiring import WIRE_DIR

MIN_PER_ORGAN = 12


def load_labeled():
    d = np.load(WIRE_DIR / "se_counts.npz", allow_pickle=True)
    M = d["M"].astype(float); keys = list(d["keys"]); regs = list(d["regs"])
    ot = json.load(open(WIRE_DIR / "organ_targets.json"))
    # region key -> set of organs it appears under
    reg2org = {}
    for o, d2 in ot.items():
        for tf, regions in d2.items():
            for (c, s, e) in regions:
                reg2org.setdefault(f"{c}:{s}-{e}", set()).add(o)
    key_idx = {k: i for i, k in enumerate(keys)}
    # organ-unique SEs only (clean single label)
    X, y, used_keys = [], [], []
    for k, orgs in reg2org.items():
        if len(orgs) == 1 and k in key_idx:
            X.append(M[key_idx[k]]); y.append(next(iter(orgs))); used_keys.append(k)
    X = np.array(X); y = np.array(y)
    # keep organs with enough unique SEs
    organs, counts = np.unique(y, return_counts=True)
    keep = set(organs[counts >= MIN_PER_ORGAN])
    m = np.array([t in keep for t in y])
    X, y = X[m], y[m]
    # drop SEs with no motif counts at all
    tot = X.sum(axis=1)
    X, y = X[tot > 0], y[tot > 0]
    return X, y, regs


def comp(X):  # composition (row-normalized) -> the combinatorial mix
    return X / X.sum(axis=1, keepdims=True)


def cv_scores(X, y, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    pred = cross_val_predict(clf, X, y, cv=skf)
    return (balanced_accuracy_score(y, pred), accuracy_score(y, pred),
            f1_score(y, pred, average="macro"), pred)


def main():
    X, y, regs = load_labeled()
    organs = sorted(set(y))
    n = len(organs)
    print("=" * 74)
    print("ORGAN CASCADE — combinatorial 'TF words' test")
    print("=" * 74)
    print(f"organ-unique SEs: {len(y)}  organs: {n}  (chance balanced-acc = {1/n:.2f})")
    for o in organs:
        print(f"   {o:16s} {np.sum(y==o)} SEs")

    Xc = comp(X)

    # 1. combinatorial classifier (all 90 TFs, composition)
    bacc, acc, f1, pred = cv_scores(Xc, y)
    print("-" * 74)
    print(f"COMBINATORIAL (90-TF composition): balanced-acc {bacc:.3f}  "
          f"acc {acc:.3f}  macroF1 {f1:.3f}")

    # 2. best SINGLE TF (marginal) baseline
    best = (-1, None)
    for j in range(Xc.shape[1]):
        b, *_ = cv_scores(Xc[:, [j]], y, seed=0)
        if b > best[0]:
            best = (b, regs[j])
    print(f"BEST SINGLE TF ({best[1]}): balanced-acc {best[0]:.3f}")

    # 3. label-permutation null for the combinatorial model
    rng = np.random.default_rng(0)
    null = []
    for _ in range(20):
        yp = rng.permutation(y)
        b, *_ = cv_scores(Xc, yp, seed=1)
        null.append(b)
    null = np.array(null)
    p = (np.sum(null >= bacc) + 1) / (len(null) + 1)
    print(f"PERMUTATION NULL: mean {null.mean():.3f} (max {null.max():.3f})  "
          f"-> p = {p:.3f}")
    print("-" * 74)
    verdict = ("COMBINATORIAL >> single  -> organ identity is in the TF-WORD"
               if bacc > best[0] + 0.05 and p < 0.05
               else "combination not clearly better than single / not significant")
    print("VERDICT:", verdict)

    # 4. interpret: each organ's top TF-word (positive LR coefficients)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xc, y)
    print("-" * 74)
    print("Per-organ 'TF word' (top +coefficient motifs):")
    classes = list(clf.classes_)
    for o in classes:
        coef = clf.coef_[classes.index(o)]
        top = [regs[k] for k in np.argsort(coef)[::-1][:6]]
        print(f"   {o:16s} {top}")

    json.dump({"organs": organs, "n_se": int(len(y)),
               "combinatorial_bacc": float(bacc), "acc": float(acc),
               "macro_f1": float(f1), "best_single_tf": best[1],
               "best_single_bacc": float(best[0]),
               "null_mean": float(null.mean()), "p_value": float(p),
               "confusion_labels": classes,
               "verdict": verdict},
              open(WIRE_DIR / "combinatorial.json", "w"), indent=2)

    # save confusion for figure
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred, labels=organs, normalize="true")
    np.savez(WIRE_DIR / "combinatorial_cm.npz", cm=cm,
             labels=np.array(organs, object))


if __name__ == "__main__":
    main()
