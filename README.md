# Cancer Risk Level Predictor
### Projet Machine Learning End-to-End

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-98.31%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## Table des Matières

- [Contexte](#-contexte)
- [Dataset](#-dataset)
- [Structure du Projet](#-structure-du-projet)
- [Démarche ML](#-démarche-ml)
- [Résultats](#-résultats)
- [Application Streamlit](#-application-streamlit)
- [Installation](#-installation)
- [Auteur](#-auteur)

---

## Contexte

Ce projet s'inscrit dans le cadre du cours **Machine Learning** et vise à développer
un pipeline complet de Data Science — de l'exploration des données jusqu'au déploiement
d'une application web interactive.

**Problématique :** Prédire automatiquement le **niveau de risque de cancer** d'un patient
(Low / Medium / High) à partir de ses symptômes et facteurs de risque environnementaux.

**Type de problème :** Classification supervisée multi-classe (3 classes)

| Classe | Description |
|--------|-------------|
| 🟢 Low | Risque faible |
| 🟡 Medium | Risque modéré |
| 🔴 High | Risque élevé |

---

##  Dataset

| Propriété | Valeur |
|-----------|--------|
| **Source** | [Kaggle — Cancer Patient Data](https://www.kaggle.com) |
| **Lignes totales** | 1 100 patients |
| **Lignes valides** | 885 patients (après fix critique) |
| **Features** | 23 variables médicales |
| **Target** | `Level` — Low / Medium / High |
| **Type de données** | Numérique (échelle 1–9) |

### Variables principales

```
Age, Gender, Air Pollution, Alcohol use, Dust Allergy,
Occupational Hazards, Genetic Risk, Chronic Lung Disease,
Balanced Diet, Obesity, Smoking, Passive Smoker,
Chest Pain, Coughing of Blood, Fatigue, Weight Loss,
Shortness of Breath, Wheezing, Swallowing Difficulty,
Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring
```

###  Découverte Critique

> **215 lignes** présentaient une valeur manquante dans la colonne `Level` (TARGET).
> L'imputation par la médiane créait artificiellement un **4ème label fictif**,
> faisant chuter l'accuracy de **98 % → 64–79 %**.
>
> **Fix appliqué :**
> ```python
> df = df.dropna(subset=['Level'])
> ```

---

## Structure du Projet

```
cancer-risk-predictor/
│
├── 📂 data/
│   └── cancer_patient_dirty_pro.xlsx      # Dataset brut
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb                       # Analyse Exploratoire
│   └── 02_Modeling.ipynb                  # Préprocessing & Modélisation
│
├── 📂 src/
│   ├── preprocessing.py                   # Pipeline de nettoyage
│   └── train.py                           # Entraînement du modèle
│
├── app.py                                 # Application Streamlit
├── model.pkl                              # Modèle entraîné (Random Forest)
├── label_encoder.pkl                      # Encodeur des labels
├── requirements.txt                       # Dépendances Python
├── projet_ml_final.qmd                    # Rapport Quarto académique
├── Presentation.pptx                      # Slides de soutenance
└── README.md                              # Ce fichier
```

---

## 🔬 Démarche ML

### 1. Analyse Exploratoire (EDA)
- Statistiques descriptives et types de variables
- Distribution de la variable cible (dataset équilibré ✅)
- Matrice de corrélation (Smoking ↔ Lung Disease : corrélation forte)
- Détection des outliers via boxplot

### 2. Prétraitement des Données

| Étape | Action | Impact |
|-------|--------|--------|
| 1 | `dropna(subset=['Level'])` | 1100 → 885 lignes |
| 2 | `fillna(median)` | Imputation features |
| 3 | `LabelEncoder` | Encodage target |
| 4 | `drop('Patient Id')` | Suppression ID |
| 5 | `clip(upper=8)` | Correction outliers |
| 6 | `train_test_split(0.2)` | 80% train / 20% test |
| 7 | `StandardScaler` | Normalisation (SVM, KNN, LR) |

### 3. Modèles Testés

| Modèle | Accuracy |
|--------|----------|
| KNN | 80.23% |
| Logistic Regression | 85.31% |
| SVM (Linear) | 88.14% |
| Decision Tree | 91.53% |
| **Random Forest** | **98.31% 🏆** |

### 4. Optimisation — GridSearchCV

```python
param_grid = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [None, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}
```

**Meilleurs paramètres :** `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`

---

## 📈 Résultats

| Métrique | Valeur |
|----------|--------|
| **Accuracy (Test Set)** | **98.31%** |
| **5-Fold CV (Moyenne)** | **99.43%** |
| **5-Fold CV (Écart-type)** | ± 0.23% |

### Rapport de Classification

```
              precision    recall  f1-score   support

        High       0.96      1.00      0.98        64
         Low       1.00      0.97      0.98        58
      Medium       1.00      0.98      0.99        55

    accuracy                           0.98       177
   macro avg       0.99      0.98      0.98       177
```

### Feature Importance (Top 5)

```
1. Coughing of Blood     0.073
2. Passive Smoker        0.059
3. Dust Allergy          0.057
4. Air Pollution         0.054
5. Fatigue               0.052
```

---

## 🌐 Application Streamlit

Une application web interactive a été développée et déployée pour permettre
la prédiction en temps réel du niveau de risque d'un patient.

🔗 **Lien de l'application :** [cancer-risk-predictor.streamlit.app](https://streamlit.io)

### Fonctionnalités
- Saisie des symptômes via des sliders interactifs
- Prédiction instantanée du niveau de risque
- Affichage des probabilités par classe
- Interface responsive et intuitive

---

##  Installation

### Prérequis
- Python 3.10+
- pip

### Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/RaniaDK/cancer-risk-predictor.git
cd cancer-risk-predictor

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application Streamlit
streamlit run app.py

# 4. Ouvrir les notebooks
jupyter notebook notebooks/
```

### Dépendances (`requirements.txt`)

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
openpyxl
jupyter
```

---

## 👤 Auteur

**Dorra Mani & Rania Dabbek**
Ecole polytechnique Sousse — 2025/2026

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/RaniaDK)


