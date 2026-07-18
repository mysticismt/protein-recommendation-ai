# 🧬 Protein Recommendation AI: Personalized Nutrition & Physiological Score Predictor

An end-to-end Machine Learning pipeline for estimating personalized daily protein requirements using real-world NHANES (National Health and Nutrition Examination Survey) population data combined with engineered physiological and biological features.

The project integrates anthropometric measurements, lifestyle factors, biochemical indicators, and mathematically derived personalized scores to build a predictive model for individualized protein requirement estimation.

---

## 🎯 Project Overview
Protein requirements vary significantly between individuals depending on factors such as:
* Body composition
* Age
* Physical activity level
* Metabolic condition
* Physiological efficiency

This project explores how machine learning can model these relationships by combining real epidemiological data with feature engineering techniques inspired by nutritional and physiological research. The final model predicts estimated daily protein requirements (grams/day) based on multiple personal and biological parameters.

---

## 🚀 Key Features

### 🧬 Advanced Feature Engineering
The pipeline creates personalized physiological indicators from available NHANES variables, including:
* Genetic-Inspired Score (GIS)
* Physical Activity Score (PAS)
* Protein Utilization Efficiency Score
* Metabolic Adjustment Factors

> **Note:** These scores are not randomly generated data; they are mathematically derived features calculated from real measurements and domain-inspired formulas.

### 🧪 Biochemical & Physiological Integration
The model incorporates important biological variables, including:
* Albumin level
* Blood Urea Nitrogen (BUN)
* Body Mass Index (BMI)
* Body weight
* Age
* Physical activity indicators
* Lifestyle factors

These features allow the model to capture relationships between physiological conditions and protein requirements.

---

## 🤖 Machine Learning Pipeline
The project follows a modular ML workflow:

```text
Raw NHANES Data
       |
       ↓
Data Cleaning & Preprocessing
       |
       ↓
Physiological Feature Engineering
       |
       ↓
Feature Selection
       |
       ↓
Regression Model Training
       |
       ↓
Model Evaluation
       |
       ↓
Protein Requirement Prediction

```

---

## 📊 Model Performance & Explainable AI (SHAP) Report

This report outlines the performance comparison between our baseline and advanced machine learning models, followed by an in-depth interpretability analysis using SHAP values.

---

### 🤖 Model Performance Comparison

To predict the exact daily protein requirement (`Protein_Requirement_g`), we evaluated multiple architectures. The advanced ensemble tree model significantly outperformed the linear baseline by capturing non-linear biological interactions.

| Model Architecture | R² Score | Mean Absolute Error (MAE) | Status |
| :--- | :---: | :---: | :---: |
| **Multiple Linear Regression** | 0.9815 | 1.506 g | Baseline |
| **Random Forest Regressor** | **0.9953** | **0.398 g** | **Top Performer** |
| **XGBoost Regressor** | 0.9951 | 0.412 g | Production Ready |

### Key Insight:
The **Random Forest** and **XGBoost** models achieved an exceptionally low MAE (~0.4g), meaning the system can predict an individual's protein needs with sub-gram accuracy based purely on their physiological profile.

---

### 🔍 SHAP Explainability & Global Feature Importance

Using Explainable AI (XAI) via **SHAP (SHapley Additive exPlanations)**, we opened the "black box" of our top-performing model. The global feature importance ranking discovered by the model is as follows:

1. **`Lean_Mass_kg` (66.2%)**: The absolute driving factor. The model accurately locked onto the fact that active muscle tissue is the primary consumer of dietary amino acids.
2. **`Activity_Score` (27.2%)**: Metabolic rate acceleration due to physical activity acts as the second major booster for protein requirements.
3. **`Weight_kg` (6.0%)**: Represents the overall biological mass baseline.
4. **Other Features (`BMI`, `Genetic_Score`, `Age` < 1%)**: Serve as micro-adjusters to fine-tune the final recommendation based on age-related decline or specific metabolic profiles.

*The generated plot `reports/shap/shap_summary.png` visually demonstrates how higher feature values (e.g., high Lean Mass) push the SHAP values in the positive direction.*

---

### 🧬 Smart Supplement Recommendation Engine
To evolve this project from a regression model into an **AI Product**, we introduced the `Recommended_Supplement_Type` feature. This maps individuals to specific protein types (`Whey_Isolate`, `Whey_Concentrate`, `Casein`, `Plant_Protein`) based on their BMI, body fat percentage, and activity levels. This column will serve as the target for our upcoming recommendation agent.

The regression model was evaluated on a separated test dataset.

| Metric | Score |
| --- | --- |
| **R² Score** | **0.9815** |
| **MAE** | **1.5034 grams** |
| **RMSE** | **1.9422 grams** |

### Performance Interpretation

The high R² score reflects the strong relationship between engineered physiological variables and the target protein requirement estimation formula. Because part of the target relationship is based on scientifically motivated mathematical modeling, the model demonstrates its ability to approximate complex physiological relationships captured in the engineered dataset. Future versions will validate performance against independent clinical nutrition datasets.

---

## 📂 Dataset Methodology

### Real Data Source

This project uses publicly available NHANES datasets containing:

* Demographic information
* Anthropometric measurements
* Physical activity information
* Dietary information
* Laboratory measurements

*Source:* [NHANES Official Website](https://www.cdc.gov/nchs/nhanes/index.htm)

### Feature Engineering Strategy

Some biological parameters required for personalized prediction are not directly available in NHANES. Instead of generating random synthetic samples, these variables were derived through mathematical transformations and domain-based formulas using available real measurements.

* **Protein Efficiency Score** = $f(\text{body composition}, \text{activity level}, \text{metabolic indicators})$
* **Physical Activity Score** = $f(\text{activity frequency}, \text{intensity}, \text{lifestyle variables})$

This approach preserves relationships between real-world measurements while expanding the predictive feature space.

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                 # Original NHANES datasets
│   ├── interim/             # Primary NHANES datasets
│   └── processed/           # Cleaned and engineered datasets
├── notebooks/
│   ├── Data collection and preparation.ipynb
│   ├── Multiple Linear Regression.ipynb
│   ├── model_comparison.ipynb                     # Linear Regression , Ridge Regression , Random Forest , XGBoost , CatBoost
│   ├──                                            Polynomial Regression , Support Vector Regression , K-Nearest Neighbors (KNN)
│   ├── Neural Networks and Deep Learning.ipynb
│   ├── Explainable AI (SHAP).ipynb
│   └── All In One.ipynb
├── reports/figures/
│   ├── 01_target_distribution.png
│   ├── 02_weight_vs_protein.png
│   ├── 03_correlation_matrix.png
│   ├── 04_model_comparison.png
│   ├── 05_nn_loss_curve.png
│   └── eda_analysis.ipynb
├── reports/shap/
│   ├── shap_summary.png
│   └── shap_waterfall.png
├── src/
│   ├── linear_regression.ipynb
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── train.py
├── models/
│   ├── 04_model_comparison.png
│   └── trained_model.pkl
├── requirements.txt
└── README.md

```

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/mysticismt/protein-recommendation-ai.git](https://github.com/mysticismt/protein-recommendation-ai.git)
cd protein-recommendation-ai

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## ▶️ Usage

Here is a functional syntax example of the pipeline deployment:

```python
# Pseudo-code example for inference pipeline
prediction = model.predict([
    age,
    weight,
    bmi,
    activity_score,
    metabolic_score
])

print(f"Estimated daily protein requirement: {prediction} grams")

```

---

## 🔬 Technologies Used

* **Programming:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Dataset:** NHANES

---

## 🔮 Future Improvements

* Integration with real genomic variants (SNP-based features)
* Connection with biological databases
* RAG-based nutrition assistant using scientific literature
* Explainable AI using **SHAP**
* Comparison with advanced baseline models: **XGBoost**, **Random Forest**, **Gradient Boosting**, and **Neural Networks**

---

## ⚠️ Limitations

* This project is a machine learning research prototype and should not be considered a medical recommendation system.
* The predicted protein requirements are based on statistical modeling and engineered physiological relationships.
* Clinical validation with controlled nutrition studies would be required before real-world healthcare usage.

---

## 📌 Conclusion

This project demonstrates how machine learning can combine population-level health data, physiological feature engineering, and predictive modeling to estimate personalized nutrition requirements. It provides a foundation for future integration of genomic information and AI-powered personalized nutrition systems.
