"""
GRAPHE 3 -- CLASSIFICATION
==========================
Idee : on AUTORISE le modele a apprendre les vrais types, puis on le teste
en validation croisee (sur des patients qu'il n'a jamais vus). C'est le juge
de paix chiffre.

Deux graphes :
  - A GAUCHE : barres de precision equilibree pour les 3 mondes x 2 modeles,
               avec la ligne du hasard (0.33).
  - A DROITE : matrice de confusion du meilleur cas (radiologue + RandomForest),
               pour voir OU sont les erreurs (les Mixtes).

Lancer :  python graphe_classification.py
Produit  :  graphe_classification.png
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from data_loader import build_dataset


def evalue(X, y, modele):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf",   modele),
    ])
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    return balanced_accuracy_score(y, y_pred), y_pred


def main():
    merged, rad_feats, rx_feats = build_dataset()
    y = merged["classe"].values
    mondes = {"Radiologue": rad_feats,
              "Radiomique": rx_feats,
              "Combine":    rad_feats + rx_feats}

    logreg = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")
    rf     = lambda: RandomForestClassifier(300, class_weight="balanced",
                                            random_state=0)

    scores_lr, scores_rf = [], []
    for cols in mondes.values():
        scores_lr.append(evalue(merged[cols].values, y, logreg())[0])
        scores_rf.append(evalue(merged[cols].values, y, rf())[0])

    fig, (ax_bar, ax_cm) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Barres ---
    x = np.arange(len(mondes)); w = 0.38
    ax_bar.bar(x - w/2, scores_lr, w, label="Reg. logistique", color="#3266ad")
    ax_bar.bar(x + w/2, scores_rf, w, label="Random Forest",   color="#1D9E75")
    ax_bar.axhline(0.33, color="red", ls="--", lw=1, label="hasard (0.33)")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(mondes.keys())
    ax_bar.set_ylabel("Precision equilibree (CV 5-fold)")
    ax_bar.set_ylim(0, 0.8)
    ax_bar.set_title("Pouvoir discriminant par approche")
    ax_bar.legend(fontsize=8)
    for i in range(len(mondes)):
        ax_bar.text(i - w/2, scores_lr[i] + 0.01, f"{scores_lr[i]:.2f}",
                    ha="center", fontsize=8)
        ax_bar.text(i + w/2, scores_rf[i] + 0.01, f"{scores_rf[i]:.2f}",
                    ha="center", fontsize=8)

    # --- Matrice de confusion (radiologue + RF) ---
    _, y_pred = evalue(merged[rad_feats].values, y, rf())
    labels = ["CCK", "CHC", "Mixtes"]
    cm = confusion_matrix(y, y_pred, labels=labels)
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(3)); ax_cm.set_xticklabels(labels)
    ax_cm.set_yticks(range(3)); ax_cm.set_yticklabels(labels)
    ax_cm.set_xlabel("Predit"); ax_cm.set_ylabel("Reel")
    ax_cm.set_title("Confusion -- Radiologue + RandomForest")
    for i in range(3):
        for j in range(3):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black",
                       fontsize=12)
    fig.colorbar(im, ax=ax_cm, fraction=0.046)

    fig.suptitle("Classification supervisee : le verdict chiffre", fontsize=13)
    fig.tight_layout()
    fig.savefig("graphe_classification.png", dpi=150, bbox_inches="tight")
    print("Sauvegarde : graphe_classification.png")


if __name__ == "__main__":
    main()