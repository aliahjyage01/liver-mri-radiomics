"""
Script principal : lance l'analyse complete dans l'ordre.

    python main.py

Etapes :
    1. Chargement / fusion propre des donnees
    2. ACP (genere acp_comparaison.png)
    3. Clustering non supervise (ARI)
    4. Classification supervisee (verdict chiffre)

Question traitee :
    "La signature radiomique apporte-t-elle plus que la relecture
     qualitative du radiologue pour distinguer CHC / CCK / Mixtes ?"
"""

import data_loader
import acp
import clustering
import classification


def main():
    print("#" * 60)
    print("# 1. DONNEES")
    print("#" * 60)
    m, rf, xf = data_loader.build_dataset()
    print(f"Patients : {len(m)} | radiologue : {len(rf)} features | "
          f"radiomique : {len(xf)} features")
    print("Classes :", m["classe"].value_counts().to_dict())

    print("\n" + "#" * 60)
    print("# 2. ACP")
    print("#" * 60)
    acp.main()

    print("\n" + "#" * 60)
    print("# 3. CLUSTERING")
    print("#" * 60)
    clustering.main()

    print("\n" + "#" * 60)
    print("# 4. CLASSIFICATION")
    print("#" * 60)
    classification.main()


if __name__ == "__main__":
    main()