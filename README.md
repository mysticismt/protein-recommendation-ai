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

## 📊 Model Performance

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
│   ├── 01_data_collection_preprocessing.ipynb
│   └── 02_machine_learning_regression.ipynb
├── reports/figures/
│   ├── 01_target_distribution.png
│   ├── 02_weight_vs_protein.png
│   ├── 03_correlation_matrix.png
│   └── eda_analysis.ipynb
├── src/
│   ├── linear_regression.ipynb
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── train.py
├── models/
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
