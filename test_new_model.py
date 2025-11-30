"""Quick test to verify new model and preprocessor load correctly."""

import sys
import pandas as pd
from models.model_loader import load_model_components
from config.settings import MODEL_JSON_PATH, PREPROCESSOR_PATH, THRESHOLD_PATH

print("Testing new model and preprocessor...")
print(f"Model path: {MODEL_JSON_PATH}")
print(f"Preprocessor path: {PREPROCESSOR_PATH}")
print(f"Threshold path: {THRESHOLD_PATH}")
print()

# Load components
model, preprocessor, threshold = load_model_components(
    MODEL_JSON_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH
)

if model is None or preprocessor is None:
    print("[FAIL] Failed to load model components")
    sys.exit(1)

print("[OK] Model loaded successfully")
print("[OK] Preprocessor loaded successfully")
print(f"[OK] Threshold: {threshold}")
print()

# Test with sample data
test_data = pd.DataFrame([{
    'GenHlth': 3.0,
    'HighBP': 1.0,
    'HighChol': 1.0,
    'Age': 9.0,
    'BMI': 28.5,
    'DiffWalk': 0.0,
    'Income': 6.0,
    'PhysHlth': 5.0,
    'HeartDiseaseorAttack': 0.0,
    'Education': 5.0,
    'PhysActivity': 1.0
}])

print("Testing prediction with sample data...")
X_processed = preprocessor.transform(test_data)
print(f"Processed shape: {X_processed.shape}")
print(f"Processed features: {X_processed[0][:5]}... (first 5)")

proba = model.predict_proba(X_processed)
print(f"Prediction probability: {proba[0][1] * 100:.2f}%")
print()
print("[OK] All tests passed!")
