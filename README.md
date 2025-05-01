# 🏢 Prédiction du risque de défaillance d'entreprises

Ce projet a pour objectif de prédire le **risque de défaillance d'une entreprise** en se basant sur des variables économiques, sectorielles et structurelles.  
L'interface est construite avec **Streamlit**, et le modèle de prédiction utilise un **Random Forest Classifier**.

---

## 🎯 Objectifs

- 🔍 Explorer les données d'entreprises françaises issues de l'INSEE.
- 🧠 Prédire le risque de défaut (faible ou élevé) via un modèle Machine Learning.
- ⚙️ Filtrer dynamiquement les données par secteur, taille, classification, COVID...
- 📊 Visualiser les taux de défaut par catégorie et la probabilité prédite pour un profil personnalisé.

---

## 🧰 Technologies utilisées

| Outil | Description |
|-------|-------------|
| `Python 3.11` | Langage de programmation principal |
| `pandas`, `matplotlib`, `seaborn` | Analyse et visualisation de données |
| `scikit-learn` | Modélisation (Random Forest) |
| `Streamlit` | Interface utilisateur interactive |
| `Git + GitHub` | Gestion de versions & hébergement |

---

## ⚡ Fonctionnalités

- Interface utilisateur avec filtres dynamiques (secteur, effectifs, etc.)
- Calcul automatique des proportions d’entreprises à risque faible et élevé
- Affichage clair de la **probabilité prédite de défaut**
- Visualisations intégrées (histogrammes, heatmap de corrélation, camembert, etc.)
- Application entièrement interactive avec Streamlit

---

## 📂 Structure du projet


---

## 🚀 Lancer le projet en local

1. Clone le projet :
   ```bash
   git clone https://github.com/lensariyacoub/defaillance-prediction.git
   cd defaillance-prediction
