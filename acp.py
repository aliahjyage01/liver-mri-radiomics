"""
Analyse en Composantes Principales (ACP) des trois "mondes" de features.

But : projeter les 147 patients en 2D et VOIR si les types de tumeurs
se separent. On standardise (Z-score) avant l'ACP car les features ont
des echelles tres differentes (taille en mm, intensites IRM sans unite, etc.).

Genere une figure cote a cote : espace radiologue vs espace radiomique.
"""

import matplotlib
matplotlib.use("Agg")  # backend sans affichage (sauvegarde fichier)
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_loader import build_dataset

COULEURS = {"CHC": "#3266ad", "CCK": "#D85A30", "Mixtes": "#1D9E75"}
MARQUEURS = {"CHC": "o", "CCK": "s", "Mixtes": "^"}


def acp_2d(X, n_components=2):
    """Impute (mediane) -> standardise -> ACP. Renvoie coords et variance expliquee."""
    pipe = Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca",   PCA(n_components=n_components, random_state=0)),
    ])
    coords = pipe.fit_transform(X)
    var = pipe.named_steps["pca"].explained_variance_ratio_
    return coords, var


def trace_acp(ax, coords, classes, var, titre):
    for cls in ("CHC", "CCK", "Mixtes"):
        mask = classes == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COULEURS[cls], marker=MARQUEURS[cls],
                   s=35, alpha=0.75, edgecolors="white", linewidths=0.4,
                   label=f"{cls} (n={mask.sum()})")
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
    ax.set_title(titre)
    ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="grey", lw=0.4, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.4, alpha=0.5)


def main(sortie="acp_comparaison.png"):
    merged, rad_feats, rx_feats = build_dataset()
    classes = merged["classe"].values

    coords_rad, var_rad = acp_2d(merged[rad_feats])
    coords_rx,  var_rx  = acp_2d(merged[rx_feats])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    trace_acp(ax1, coords_rad, classes, var_rad,
              "Espace radiologue (qualitatif)")
    trace_acp(ax2, coords_rx, classes, var_rx,
              "Espace radiomique (PyRadiomics)")
    fig.suptitle("ACP : separation des types de tumeurs selon l'approche",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {sortie}")
    print(f"Variance expliquee radiologue (PC1,PC2) : "
          f"{var_rad[0]:.2f}, {var_rad[1]:.2f}")
    print(f"Variance expliquee radiomique (PC1,PC2) : "
          f"{var_rx[0]:.2f}, {var_rx[1]:.2f}")


if __name__ == "__main__":
    main()