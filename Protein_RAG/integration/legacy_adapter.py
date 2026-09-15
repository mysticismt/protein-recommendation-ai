from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd


class LegacyProjectAdapter:
    """Read-only adapter around the existing legacy dataset and trained models."""
    REGRESSOR_FEATURES = ["Age", "Is_Male", "Weight_kg", "Height_cm", "BMI", "Body_Fat_Percent", "Lean_Mass_kg", "Activity_Score", "Daily_Protein_Intake_g", "Genetic_Score"]
    CLASSIFIER_FEATURES = ["Age", "Gender_Encoded", "Weight_kg", "Height_cm", "BMI", "Body_Fat_Percent", "Lean_Mass_kg", "Activity_Score", "Genetic_Score"]

    def __init__(self, project_root: Optional[str | Path] = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        self.data_path = self.project_root / "data" / "processed" / "NHANES_Master_Dataset.csv"
        self.regressor = joblib.load(self.project_root / "models" / "random_forest_protein_model.pkl")
        self.classifier = joblib.load(self.project_root / "models" / "trained_supplement_classifier.pkl")
        self.df = pd.read_csv(self.data_path)
        self.df["Is_Male"] = self.df["Gender"].map({"Male": 1, "Female": 0})

    def build_real_case(self, row_id: float, shap_top_k: int = 10) -> Dict[str, Any]:
        mask = np.isclose(self.df["ID"].astype(float), float(row_id))
        if not mask.any():
            raise KeyError(f"No NHANES row found with ID={row_id!r}")
        row = self.df.loc[mask].iloc[0]
        X = pd.DataFrame([{k: float(row[k]) for k in self.REGRESSOR_FEATURES}], columns=self.REGRESSOR_FEATURES)
        XC = pd.DataFrame([{k: float(row["Is_Male"]) if k == "Gender_Encoded" else float(row[k]) for k in self.CLASSIFIER_FEATURES}], columns=self.CLASSIFIER_FEATURES)
        pred = float(np.asarray(self.regressor.predict(X)).reshape(-1)[0])
        supplement = str(self.classifier.predict(XC)[0])
        intake = float(row["Daily_Protein_Intake_g"]); weight = float(row["Weight_kg"])
        ml = {"model": "RandomForestRegressor", "Daily_Protein_Requirement_g": pred, "Daily_Protein_Requirement_g_per_kg": pred / weight, "Daily_Protein_Intake_g": intake, "Daily_Protein_Intake_g_per_kg": intake / weight, "Intake_Gap_g": pred - intake, "Recommended_Supplement": supplement, "dataset_target_Protein_Requirement_g": float(row["Protein_Requirement_g"])}
        if hasattr(self.classifier, "predict_proba"):
            ml["Supplement_Probabilities"] = {str(c): float(p) for c, p in zip(self.classifier.classes_, self.classifier.predict_proba(XC)[0])}
        profile = {k: row[k] for k in ["ID", "Age", "Gender", "Weight_kg", "Height_cm", "BMI", "Body_Fat_Percent", "Lean_Mass_kg", "Activity_Level", "Activity_Score", "Daily_Protein_Intake_g", "Genetic_Score"] if k in row.index}
        profile["Is_Male"] = int(row["Is_Male"])
        return {"profile": profile, "ml_results": ml, "shap_summary": {}}
