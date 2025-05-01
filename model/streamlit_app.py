import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 📂 Chargement du fichier CSV nettoyé
file_path = os.path.join("data", "defaillances_nettoye.csv")
if not os.path.exists(file_path):
    st.error(f"❌ Fichier non trouvé : {file_path}")
    st.stop()

# ✅ Chargement des données
df = pd.read_csv(file_path, encoding='utf-8')
df = df.loc[:, ~df.columns.duplicated()]  # Supprimer les colonnes dupliquées

# 📅 Titre principal
st.title("📊 Analyse des défaillances d'entreprises")

# 🔹 Sidebar : Activer ou désactiver les filtres
st.sidebar.title("🔧 Filtres interactifs")
use_filters = st.sidebar.checkbox("🔘 Activer les filtres", value=True)

if use_filters:
    secteur_choisi = st.sidebar.selectbox("Secteur d'activité", sorted(df["SecteurActivité"].dropna().unique()))
    tranche_choisie = st.sidebar.selectbox("Tranche d'effectifs", sorted(df["TrancheEffectifs"].dropna().unique()))

    score_min, score_max = st.sidebar.slider("Score sectoriel",
        float(df["Score sectoriel"].min()),
        float(df["Score sectoriel"].max()),
        (float(df["Score sectoriel"].min()), float(df["Score sectoriel"].max()))
    )

    valeur_min, valeur_max = st.sidebar.slider("Valeur ajoutée (€)",
        float(df["Valeur ajaoutée"].min()),
        float(df["Valeur ajaoutée"].max()),
        (float(df["Valeur ajaoutée"].min()), float(df["Valeur ajaoutée"].max()))
    )

    df_affiche = df[
        (df["SecteurActivité"] == secteur_choisi) &
        (df["TrancheEffectifs"] == tranche_choisie) &
        (df["Score sectoriel"].between(score_min, score_max)) &
        (df["Valeur ajaoutée"].between(valeur_min, valeur_max))
    ]
else:
    df_affiche = df

# 📄 Aperçu des données
st.subheader("🔍 Données affichées")
st.dataframe(df_affiche.head(20))

# 📊 Statistiques de base
st.subheader("🧬 Types de colonnes")
st.write(df_affiche.dtypes)

st.subheader("⚠️ Valeurs manquantes")
st.write(df_affiche.isnull().sum())

st.subheader("📊 Statistiques descriptives")
st.write(df_affiche.describe())

# 🔺 Visualisation : histogramme taux de défaillance 3 mois
st.subheader("📉 Taux de défaillance à 3 mois")
fig, ax = plt.subplots()
ax.hist(df_affiche["Tauxdedéfaillances3mois"], bins=30, color='skyblue', edgecolor='black')
ax.set_xlabel("Taux de défaillance (3 mois)")
ax.set_ylabel("Nombre d'observations")
st.pyplot(fig)

# 🏛️ Taux moyen par secteur
st.subheader("📌 Taux de défaillance moyen par secteur")
moyennes_secteurs = df_affiche.groupby("SecteurActivité")["Tauxdedéfaillances3mois"].mean().sort_values(ascending=False).head(10)
st.bar_chart(moyennes_secteurs)

# 📊 Taux moyen par tranche d'effectifs
st.subheader("📊 Taux de défaillance moyen par taille d'entreprise")
df_grouped = df_affiche.groupby('TrancheEffectifs')[['Tauxdedéfaillances3mois', 'Taux de défaillances 6 mois']].mean()
st.bar_chart(df_grouped)

# 🏢 Top 10 secteurs défaillance 6 mois
st.subheader("🏢 Top 10 secteurs - Taux de défaillance à 6 mois")
top_secteurs = df_affiche.groupby("SecteurActivité")["Taux de défaillances 6 mois"].mean().sort_values(ascending=False).head(10)
st.bar_chart(top_secteurs)

# 🌐 Camembert classification
st.subheader("📍 Répartition des entreprises par classification")
classification_counts = df_affiche["Classification"].value_counts()
fig, ax = plt.subplots()
ax.pie(classification_counts, labels=classification_counts.index, autopct='%1.1f%%')
ax.axis('equal')
st.pyplot(fig)

# 🔢 Matrice de corrélation
st.subheader("🔎 Matrice de corrélation")
numerical_cols = df_affiche.select_dtypes(include=['float64', 'int64'])
corr_matrix = numerical_cols.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)


# 🔍 Création de la variable cible binaire
# Risque élevé = 1 si le taux est supérieur à la médiane
seuil = df["Tauxdedéfaillances3mois"].median()
df["Risque_Eleve"] = (df["Tauxdedéfaillances3mois"] > seuil).astype(int)

# Vérification
st.subheader("🎯 Répartition de la variable cible (Risque élevé)")
st.write(df["Risque_Eleve"].value_counts())

st.subheader("🛠️ Préparation des données pour le modèle")

# 🧹 1. Sélection des variables explicatives
colonnes_utiles = [
    "SecteurActivité",
    "TrancheEffectifs",
    "Score sectoriel",
    "Valeur ajaoutée",
    "Nombre de créations dans les 3 mois",
    "Classification"
]

df_ml = df[colonnes_utiles + ["Risque_Eleve"]].dropna()

# 🧠 2. Encodage des variables catégorielles
df_ml_encoded = pd.get_dummies(df_ml, columns=["SecteurActivité", "TrancheEffectifs", "Classification"])

# 🔄 3. Séparation X / y
X = df_ml_encoded.drop("Risque_Eleve", axis=1)
y = df_ml_encoded["Risque_Eleve"]

# ✅ Aperçu
st.write("✅ Dimensions de X :", X.shape)
st.write("🎯 Variable cible (y) :")
st.write(y.value_counts())


st.subheader("🧪 Entraînement du modèle")

# 1. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Prédictions
y_pred = model.predict(X_test)

# 4. Résultats
st.write("✅ **Matrice de confusion**")
st.write(confusion_matrix(y_test, y_pred))

st.write("📊 **Rapport de classification**")
st.text(classification_report(y_test, y_pred))

st.sidebar.title("🔍 Simulation de prédiction")

# Saisie utilisateur
secteur_input = st.sidebar.selectbox("Secteur", sorted(df["SecteurActivité"].dropna().unique()))
tranche_input = st.sidebar.selectbox("Tranche d’effectifs", sorted(df["TrancheEffectifs"].dropna().unique()))
classification_input = st.sidebar.selectbox("Classification", sorted(df["Classification"].dropna().unique()))
score_input = st.sidebar.number_input("Score sectoriel", value=float(df["Score sectoriel"].median()))
valeur_input = st.sidebar.number_input("Valeur ajoutée (€)", value=float(df["Valeur ajaoutée"].median()))
creations_input = st.sidebar.number_input("Nombre de créations dans les 3 mois", value=0)

# Bouton de prédiction
if st.sidebar.button("🔮 Lancer la prédiction"):

    # Créer un DataFrame avec 1 ligne
    input_df = pd.DataFrame({
        "Score sectoriel": [score_input],
        "Valeur ajaoutée": [valeur_input],
        "Nombre de créations dans les 3 mois": [creations_input],
        "SecteurActivité_" + secteur_input: [1],
        "TrancheEffectifs_" + tranche_input: [1],
        "Classification_" + classification_input: [1]
    })

    # Ajouter les colonnes manquantes (comme dans X)
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Réordonner comme X
    input_df = input_df[X.columns]

    # Prédiction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.success(f"🔎 Résultat : {'Risque élevé' if prediction == 1 else 'Risque faible'} (probabilité : {proba:.2%})")

nb_defaillantes = df["Risque_Eleve"].sum()
nb_total = len(df)
pourcentage = nb_defaillantes / nb_total * 100

st.subheader("📊 Entreprises à risque élevé détectées")
st.write(f"Nombre d'entreprises classées à **risque élevé** : `{nb_defaillantes}` sur `{nb_total}`")
st.write(f"Ce qui représente environ **{pourcentage:.2f}%** des données.")
