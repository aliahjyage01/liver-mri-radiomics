"""
Classification supervisee : la vraie comparaison chiffree.

On entraine deux modeles (regression logistique, Random Forest) a predire
le type de tumeur (CHC / CCK / Mixtes), pour chacun des trois mondes,
en validation croisee 5-fold stratifiee.

Metrique : precision EQUILIBREE (balanced accuracy), qui corrige le
desequilibre des classes (87 / 37 / 23). Repere : le hasard pur sur
3 classes equilibrees ~ 0.33.

C'est ce script qui produit le verdict :
    Radiologue > Combine > Radiomique.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             confusion_matrix)

from data_loader import build_dataset


def evalue(X, y, modele):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pipe = Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf",   modele),
    ])
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    return (balanced_accuracy_score(y, y_pred),
            f1_score(y, y_pred, average="macro"),
            y_pred)


def main():
    merged, rad_feats, rx_feats = build_dataset()
    y = merged["classe"].values
    mondes = {
        "Radiologue": rad_feats,
        "Radiomique": rx_feats,
        "Combine":    rad_feats + rx_feats,
    }
    modeles = {
        "LogReg":       LogisticRegression(max_iter=2000,
                                           class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300,
                                               class_weight="balanced",
                                               random_state=0),
    }

    print("=== Classification CHC/CCK/Mixtes (CV 5-fold) ===")
    print("Repere hasard ~ 0.33\n")
    print(f"{'Modele':14s}{'Monde':12s}{'BalAcc':>8s}{'MacroF1':>9s}")
    for nom_m, modele in modeles.items():
        for nom_w, cols in mondes.items():
            ba, f1, _ = evalue(merged[cols].values, y, modele)
            print(f"{nom_m:14s}{nom_w:12s}{ba:8.3f}{f1:9.3f}")
        print()

    # Matrice de confusion du meilleur cas (radiologue + RF).
    _, _, y_pred = evalue(merged[rad_feats].values, y,
                          RandomForestClassifier(n_estimators=300,
                                                 class_weight="balanced",
                                                 random_state=0))
    labels = ["CCK", "CHC", "Mixtes"]
    cm = confusion_matrix(y, y_pred, labels=labels)
    print("Matrice de confusion -- Radiologue + RandomForest")
    print("(lignes = vrai, colonnes = predit)", labels)
    print(cm)


if __name__ == "__main__":
    main()