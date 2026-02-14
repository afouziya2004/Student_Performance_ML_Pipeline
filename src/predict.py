# predict.py
import pandas as pd
import joblib
import os

# ==============================
# LOAD MODEL
# ==============================

model_path = os.path.join(os.path.dirname(__file__), "final_model.pkl")
print(f"Loading model from: {model_path}")

model = joblib.load(model_path)
print("Model loaded successfully!")

# ==============================
# CREATE SAMPLE INPUT
# ==============================

sample_student = pd.DataFrame([{
    "StudyHours": 6,
    "Attendance": 85,
    "AssignmentsCompleted": 7,
    "FamilyIncome": 40000,
    "SleepHours": 7,
    "Study_Attendance": 510,
    "StudyHours_sq": 36
}])

print("\nOriginal Input:")
print(sample_student)

# ==============================
# AUTO ALIGN FEATURES WITH MODEL
# ==============================

expected_features = model.feature_names_in_

print("\nModel expects features:")
print(expected_features)

# keep only expected columns
sample_student = sample_student.reindex(columns=expected_features)

print("\nAligned Input:")
print(sample_student)

# ==============================
# MAKE PREDICTION
# ==============================

prediction = model.predict(sample_student)

print("\n Predicted Test Score:", round(prediction[0], 2))
