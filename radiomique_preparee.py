"""
RADIOMIQUE PREPAREE : normalisation + selection de features
============================================================
But : tester si la radiomique, une fois CORRECTEMENT preparee, rejoint la
performance du radiologue -- ou si son echec est structurel.

Deux corrections apportees a la radiomique brute :

  1. NORMALISATION PAR PHASE (StandardScaler dans le pipeline)
     L'intensite IRM n'a pas d'unite absolue : on a mesure un facteur 35x
     entre patients sur les features d'intensite. La standardisation z-score
     remet toutes les features sur une echelle comparable.

  2. SELECTION DE FEATURES (SelectKBest)
     106 features pour 147 patients => sur-apprentissage du bruit.
     On ne garde que les k features les plus liees a la classe.

POINT METHODO CRUCIAL : la selection de features est faite DANS la validation
croisee (via le Pipeline), pas avant. Sinon on "regarde" les etiquettes de
test pour choisir les features = fuite de donnees = score artificiellement
gonfle. Le Pipeline garantit que selection + modele sont reapprises sur chaque
pli d'entrainement uniquement.

Lancer :  python radiomique_preparee.py
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score

from data_loader import build_dataset


def score_pipeline(X, y, etapes):
    """Evalue un pipeline complet en validation croisee 5-fold stratifiee."""
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = Pipeline(etapes)
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    return balanced_accuracy_score(y, y_pred)


def main():
    merged, rad_feats, rx_feats = build_dataset()
    y = merged["classe"].values
    X_rx = merged[rx_feats].values
    X_rad = merged[rad_feats].values

    base = [("imput", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]
    rf = lambda: RandomForestClassifier(300, class_weight="balanced",
                                        random_state=0)

    print("=== Reference ===")
    score_rad = score_pipeline(X_rad, y, base + [("clf", rf())])
    print(f"Radiologue (19 var)                  : {score_rad:.3f}")

    score_rx_brut = score_pipeline(X_rx, y, base + [("clf", rf())])
    print(f"Radiomique brute (106 feat)          : {score_rx_brut:.3f}")

    print("\n=== Radiomique preparee : effet de la selection ===")
    ks = [5, 10, 15, 20, 30, 50, 106]
    scores_k = []
    for k in ks:
        k_eff = min(k, len(rx_feats))
        etapes = base + [
            ("select", SelectKBest(f_classif, k=k_eff)),
            ("clf", rf()),
        ]
        s = score_pipeline(X_rx, y, etapes)
        scores_k.append(s)
        print(f"Radiomique + top-{k:<3d} features        : {s:.3f}")

    best_k = ks[int(np.argmax(scores_k))]
    best_s = max(scores_k)
    print(f"\nMeilleur : top-{best_k} features -> {best_s:.3f}")
    print(f"(radiologue = {score_rad:.3f}, radiomique brute = {score_rx_brut:.3f})")

    # Graphe : score radiomique en fonction du nombre de features,
    # avec les deux lignes de reference.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores_k, "o-", color="#1D9E75", label="Radiomique preparee")
    ax.axhline(score_rad, color="#3266ad", ls="--",
               label=f"Radiologue ({score_rad:.2f})")
    ax.axhline(score_rx_brut, color="#888780", ls=":",
               label=f"Radiomique brute ({score_rx_brut:.2f})")
    ax.axhline(0.33, color="red", ls="--", lw=0.8, label="hasard (0.33)")
    ax.set_xlabel("Nombre de features selectionnees (k)")
    ax.set_ylabel("Precision equilibree (CV 5-fold)")
    ax.set_title("La radiomique remonte-t-elle apres preparation ?")
    ax.set_ylim(0.3, 0.75)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("radiomique_preparee.png", dpi=150, bbox_inches="tight")
    print("\nGraphe sauvegarde : radiomique_preparee.png")

    # Quelles features ont ete jugees les plus discriminantes ?
    sel = SelectKBest(f_classif, k=15)
    Xs = Pipeline(base).fit_transform(X_rx)
    sel.fit(Xs, y)
    tops = sorted(zip(rx_feats, sel.scores_), key=lambda t: -t[1])[:10]
    print("\n=== Top 10 features radiomiques (score F) ===")
    for nom, sc in tops:
        print(f"  {nom:45s} {sc:6.1f}")


if __name__ == "__main__":
    main()