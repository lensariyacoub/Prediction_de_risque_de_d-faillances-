import pandas as pd
import os

# Chemin du fichier Excel
file_path = os.path.join("data", "Copie de DRiM_GAME_2024_Séries_défaillances_d'entreprise.xlsx")

# Vérification de l'existence
if not os.path.exists(file_path):
    print(f"❌ Fichier introuvable à : {file_path}")
else:
    print("✅ Fichier trouvé, chargement...")

    # Chargement des données
    df = pd.read_excel(file_path)

    # Vérification des colonnes en double
    duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        print(f"\n❗ Colonnes dupliquées détectées : {duplicated_columns}")
        df = df.loc[:, ~df.columns.duplicated()]
        print("✅ Colonnes dupliquées supprimées.")
    else:
        print("✅ Aucune colonne dupliquée.")

    # Aperçu
    print("\n🧾 Aperçu des données :")
    print(df.head())

    # Enregistrement au format CSV
    output_csv_path = os.path.join("data", "defaillances_nettoye.csv")
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Données nettoyées sauvegardées dans : {output_csv_path}")
# Chargement
df = pd.read_csv("data/defaillances_nettoye.csv")

# Aperçu des premières lignes
print("\n📌 Aperçu des données :")
print(df.head())

# Types de variables
print("\n🔎 Types de colonnes :")
print(df.dtypes)

# Valeurs manquantes
print("\n🚨 Valeurs manquantes par colonne :")
print(df.isna().sum())

# Statistiques descriptives
print("\n📊 Statistiques descriptives :")
print(df.describe(include='all'))

# Chargement du fichier nettoyé
file_path = os.path.join("data", "defaillances_nettoye.csv")

# Lecture du CSV avec encodage explicite (UTF-8 ou autre selon besoin)
df = pd.read_csv(file_path, encoding='utf-8')

# Configuration de l’affichage pour éviter les coupures ou "..."
pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
pd.set_option('display.width', 1000)        # Largeur max de la console
pd.set_option('display.max_colwidth', None) # Ne pas tronquer les colonnes de texte
pd.set_option('display.float_format', '{:.2f}'.format)  # Format flottant plus lisible

# Affichage du DataFrame complet (tu peux limiter à 20 lignes par exemple)
print(df.head(20))