"""
Chargement et fusion propre des 4 fichiers Excel.

Pieges geres ici :
  - patient_num N'EST PAS unique : il est reutilise d'une classe a l'autre.
    -> la vraie cle est le couple (classe, patient_num).
  - Les colonnes 'diagnostics_*' sont des metadonnees (versions, hash, spacing),
    PAS des features radiomiques. On ne garde que les colonnes 'original_*'.
  - Un patient a plusieurs lignes radiomiques (une par phase d'injection
    ART / PORT / TARD) -> on agrege par moyenne pour avoir 1 ligne / patient.
  - Les echelles d'intensite IRM n'ont pas d'unite absolue : la standardisation
    (Z-score) est faite plus tard, dans le pipeline d'analyse, pas ici.
"""

import pandas as pd

# Correspondance Type_tumeur (fichier relecture) -> nom de classe.
# Deduit des effectifs : 1=CCK (23), 2=CHC (87), 3=Mixtes (37).
TYPE_MAP = {1: "CCK", 2: "CHC", 3: "Mixtes"}


def load_radiologue(path="Relectures_imageries.xlsx"):
    """Relecture qualitative du radiologue : 1 ligne par patient, ~19 variables."""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]  # certains entetes ont des espaces
    df["classe"] = df["Type_tumeur"].map(TYPE_MAP)
    df = df.rename(columns={"Patient_number": "patient_num"})
    features = [c for c in df.columns
                if c not in ("Type_tumeur", "patient_num", "classe")]
    return df, features


def load_radiomique(path="global_excel_resampled.xlsx"):
    """
    Features PyRadiomics sur le volume 3D (rrechantillonne a 2 mm).
    Agregation des phases d'injection par moyenne -> 1 ligne par patient.
    """
    df = pd.read_excel(path)
    features = [c for c in df.columns if c.startswith("original_")]
    agg = (df.groupby(["classe_name", "patient_num"])[features]
             .mean()
             .reset_index()
             .rename(columns={"classe_name": "classe"}))
    return agg, features


def build_dataset(rad_path="Relectures_imageries.xlsx",
                  rx_path="global_excel_resampled.xlsx"):
    """
    Fusionne les deux sources sur la cle composite (classe, patient_num)
    et renvoie :
        merged    : DataFrame 147 lignes (1 par patient)
        rad_feats : liste des colonnes radiologue
        rx_feats  : liste des colonnes radiomiques (constantes retirees)
    """
    rad, rad_feats = load_radiologue(rad_path)
    rx, rx_feats = load_radiomique(rx_path)

    merged = rad[["classe", "patient_num"] + rad_feats].merge(
        rx, on=["classe", "patient_num"], how="inner")

    # On retire les features radiomiques constantes (variance nulle = inutiles).
    X = merged[rx_feats].apply(pd.to_numeric, errors="coerce")
    rx_feats = [c for c in rx_feats if X[c].std() > 0]

    return merged, rad_feats, rx_feats


if __name__ == "__main__":
    m, rf, xf = build_dataset()
    print(f"Patients fusionnes : {len(m)}")
    print(f"Features radiologue : {len(rf)}")
    print(f"Features radiomique : {len(xf)}")
    print("Repartition des classes :", m["classe"].value_counts().to_dict())