"""
Stabilite du verdict : validation croisee repetee sur 20 graines.

Avec 147 patients, un seul decoupage 5-fold peut faire varier la precision
equilibree de plusieurs points. On repete donc la CV 5-fold stratifiee avec
20 melanges differents et on rapporte moyenne, ecart-type, min et max.

Si les intervalles [min, max] du radiologue et de la radiomique ne se
recouvrent pas, le verdict ne depend pas du decoupage.

Lancer :  python repeated_cv.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score

from data_loader import build_dataset

N_GRAINES = 20


def main():
    merged, rad_feats, rx_feats = build_dataset()
    y = merged["classe"].values
    mondes = {"Radiologue": rad_feats,
              "Radiomique": rx_feats,
              "Combine":    rad_feats + rx_feats}
    modeles = {
        "RandomForest": lambda: RandomForestClassifier(
            300, class_weight="balanced", random_state=0, n_jobs=-1),
        "LogReg": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced"),
    }

    print(f"=== CV 5-fold stratifiee repetee ({N_GRAINES} graines) ===")
    print(f"{'Monde':12s}{'Modele':14s}{'moy':>7s}{'e-t':>7s}{'min':>7s}{'max':>7s}")
    for nom_w, cols in mondes.items():
        for nom_m, fabrique in modeles.items():
            scores = []
            for graine in range(N_GRAINES):
                pipe = Pipeline([
                    ("imput", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf",   fabrique()),
                ])
                cv = StratifiedKFold(5, shuffle=True, random_state=graine)
                y_pred = cross_val_predict(pipe, merged[cols].values, y, cv=cv)
                scores.append(balanced_accuracy_score(y, y_pred))
            s = np.array(scores)
            print(f"{nom_w:12s}{nom_m:14s}{s.mean():7.3f}{s.std():7.3f}"
                  f"{s.min():7.3f}{s.max():7.3f}")


if __name__ == "__main__":
    main()
