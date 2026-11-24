"""
Clustering non supervise (k-means, k=3) sur les trois mondes de features.

Question : si on laisse l'algorithme regrouper les patients SANS lui dire
les vrais types, est-ce que les groupes trouves correspondent aux types
de tumeurs ?

On mesure avec l'ARI (Adjusted Rand Index) :
    ARI = 1.0  -> groupes parfaitement alignes sur les vrais types
    ARI ~ 0    -> aucun lien (hasard)
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from data_loader import build_dataset


def clustering_ari(X, y_vrai, k=3):
    Xs = Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]).fit_transform(X)
    labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(Xs)
    return adjusted_rand_score(y_vrai, labels)


def main():
    merged, rad_feats, rx_feats = build_dataset()
    y = merged["classe"].values
    mondes = {
        "Radiologue": rad_feats,
        "Radiomique": rx_feats,
        "Combine":    rad_feats + rx_feats,
    }
    print("=== Clustering k-means (k=3) vs vrais types ===")
    print("ARI : 1.0 = parfait, 0 = hasard\n")
    for nom, cols in mondes.items():
        ari = clustering_ari(merged[cols], y)
        print(f"{nom:12s} ARI = {ari:.3f}")


if __name__ == "__main__":
    main()