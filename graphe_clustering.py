"""
GRAPHE 2 -- CLUSTERING
======================
Idee : k-means range les patients en 3 paquets SANS connaitre les vrais types,
juste d'apres les variables. On compare ensuite ses paquets aux vrais types.
S'ils coincident, c'est que la structure naturelle des donnees epouse deja
les types (ARI proche de 1). Sinon, ARI proche de 0.

Le graphe a 2 etages, sur la projection ACP radiologue (pour pouvoir dessiner) :
  - en HAUT  : couleur = type REEL
  - en BAS   : couleur = paquet TROUVE par k-means
Si les deux etages se ressemblent, le clustering a "retrouve" les types.
On affiche aussi les ARI des deux espaces en barres.

Lancer :  python graphe_clustering.py
Produit  :  graphe_clustering.png
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from data_loader import build_dataset

COULEURS = {"CHC": "#3266ad", "CCK": "#D85A30", "Mixtes": "#1D9E75"}
PALETTE_CLUSTER = ["#7F77DD", "#EF9F27", "#5DCAA5"]  # 3 paquets, couleurs neutres


def prepare(X):
    return Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]).fit_transform(X)


def ari_de(X, y):
    Xs = prepare(X)
    labels = KMeans(n_clusters=3, n_init=20, random_state=0).fit_predict(Xs)
    return adjusted_rand_score(y, labels), labels


def main():
    merged, rad_feats, rx_feats = build_dataset()
    y = merged["classe"].values

    # ARI des deux espaces (pour les barres)
    ari_rad, labels_rad = ari_de(merged[rad_feats], y)
    ari_rx,  _          = ari_de(merged[rx_feats], y)

    # Projection ACP radiologue, juste pour pouvoir dessiner en 2D
    coords = Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca",   PCA(2, random_state=0)),
    ]).fit_transform(merged[rad_feats])

    fig = plt.figure(figsize=(12, 6))
    ax_vrai = fig.add_subplot(2, 2, 1)
    ax_clus = fig.add_subplot(2, 2, 3)
    ax_bar  = fig.add_subplot(1, 2, 2)

    # Etage 1 : types reels
    for cls in ("CHC", "CCK", "Mixtes"):
        m = y == cls
        ax_vrai.scatter(coords[m, 0], coords[m, 1], c=COULEURS[cls],
                        s=28, alpha=0.75, label=cls, edgecolors="white", lw=0.3)
    ax_vrai.set_title("Types REELS", fontsize=10)
    ax_vrai.legend(fontsize=7); ax_vrai.set_xticks([]); ax_vrai.set_yticks([])

    # Etage 2 : paquets trouves par k-means (espace radiologue)
    for k in range(3):
        m = labels_rad == k
        ax_clus.scatter(coords[m, 0], coords[m, 1], c=PALETTE_CLUSTER[k],
                        s=28, alpha=0.75, label=f"paquet {k}",
                        edgecolors="white", lw=0.3)
    ax_clus.set_title("Paquets TROUVES par k-means (sans connaitre les types)",
                      fontsize=10)
    ax_clus.legend(fontsize=7); ax_clus.set_xticks([]); ax_clus.set_yticks([])

    # Barres ARI
    ax_bar.bar(["Radiologue", "Radiomique"], [ari_rad, ari_rx],
               color=["#3266ad", "#888780"])
    ax_bar.axhline(0, color="black", lw=0.6)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("ARI  (1 = paquets alignes sur les types, 0 = hasard)")
    ax_bar.set_title("Le clustering retrouve-t-il les types ?")
    for i, v in enumerate([ari_rad, ari_rx]):
        ax_bar.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)

    fig.suptitle("Clustering non supervise : structure naturelle vs vrais types",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("graphe_clustering.png", dpi=150, bbox_inches="tight")
    print(f"Sauvegarde : graphe_clustering.png  (ARI rad={ari_rad:.2f}, "
          f"rx={ari_rx:.2f})")


if __name__ == "__main__":
    main()