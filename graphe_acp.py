"""
GRAPHE 1 -- ACP
================
Idee : projeter les patients (decrits par beaucoup de variables) sur un plan 2D,
en gardant l'angle de projection qui etale le plus le nuage.
On trace cote a cote l'espace radiologue et l'espace radiomique pour VOIR
lequel separe le mieux les types de tumeurs.

Lancer :  python graphe_acp.py
Produit  :  graphe_acp.png
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_loader import build_dataset

COULEURS = {"CHC": "#3266ad", "CCK": "#D85A30", "Mixtes": "#1D9E75"}
MARQUEURS = {"CHC": "o", "CCK": "s", "Mixtes": "^"}


def projette_acp(X):
    """Impute (mediane) -> standardise (Z-score) -> ACP 2D."""
    pipe = Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),               # indispensable : echelles differentes
        ("pca",   PCA(n_components=2, random_state=0)),
    ])
    coords = pipe.fit_transform(X)
    var = pipe.named_steps["pca"].explained_variance_ratio_
    return coords, var


def trace(ax, coords, classes, var, titre):
    for cls in ("CHC", "CCK", "Mixtes"):
        m = classes == cls
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=COULEURS[cls], marker=MARQUEURS[cls], s=40,
                   alpha=0.75, edgecolors="white", linewidths=0.4,
                   label=f"{cls} (n={m.sum()})")
    ax.set_xlabel(f"PC1 -- {var[0]*100:.0f}% de la variance")
    ax.set_ylabel(f"PC2 -- {var[1]*100:.0f}% de la variance")
    ax.set_title(titre)
    ax.axhline(0, color="grey", lw=0.4, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8)


def main():
    merged, rad_feats, rx_feats = build_dataset()
    classes = merged["classe"].values

    coords_rad, var_rad = projette_acp(merged[rad_feats])
    coords_rx,  var_rx  = projette_acp(merged[rx_feats])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    trace(a1, coords_rad, classes, var_rad, "Espace radiologue (19 variables)")
    trace(a2, coords_rx,  classes, var_rx,  "Espace radiomique (106 features)")
    fig.suptitle("ACP : les types se separent-ils ?  (gauche=oui, droite=non)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("graphe_acp.png", dpi=150, bbox_inches="tight")
    print("Sauvegarde : graphe_acp.png")


if __name__ == "__main__":
    main()