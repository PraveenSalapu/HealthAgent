"""
Diagnostic script to examine what's inside your pickle file.
This will help identify why the model isn't loading correctly.

Usage:
    python diagnose_model.py
"""

import joblib
import pickle
from pathlib import Path

MODEL_PATH = "model_output/ml_xgboost_smoteenn_pipeline.pkl"

def diagnose():
    print("=" * 70)
    print("DIAGNOSING MODEL FILE")
    print("=" * 70)
    
    # Check if file exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ File not found: {MODEL_PATH}")
        return
    
    print(f"✅ File found: {MODEL_PATH}")
    print(f"   File size: {Path(MODEL_PATH).stat().st_size:,} bytes")
    
    # Try loading with joblib
    print("\n1. Attempting to load with joblib...")
    try:
        obj = joblib.load(MODEL_PATH)
        print(f"✅ Loaded successfully with joblib")
    except Exception as e:
        print(f"❌ joblib failed: {e}")
        print("\n2. Attempting to load with pickle...")
        try:
            with open(MODEL_PATH, 'rb') as f:
                obj = pickle.load(f)
            print(f"✅ Loaded successfully with pickle")
        except Exception as e2:
            print(f"❌ pickle also failed: {e2}")
            return
    
    # Analyze the loaded object
    print("\n3. Object Analysis:")
    print(f"   Type: {type(obj)}")
    print(f"   Type name: {type(obj).__name__}")
    print(f"   Module: {type(obj).__module__}")
    
    # Check if it's a dict
    if isinstance(obj, dict):
        print("\n   ⚠️ Object is a DICTIONARY")
        print("   Keys:", list(obj.keys()))
        print("\n   Examining each key:")
        for key, value in obj.items():
            print(f"     '{key}': {type(value).__name__}")
            if hasattr(value, 'predict'):
                print(f"       ✅ This has a 'predict' method - might be the model!")
    
    # Check if it's a list/tuple
    elif isinstance(obj, (list, tuple)):
        print(f"\n   ⚠️ Object is a {type(obj).__name__} with {len(obj)} items")
        for i, item in enumerate(obj):
            print(f"     [{i}]: {type(item).__name__}")
            if hasattr(item, 'predict'):
                print(f"       ✅ This has a 'predict' method!")
    
    # Check attributes
    print(f"\n4. Object Attributes:")
    attrs = dir(obj)
    important_attrs = [a for a in attrs if not a.startswith('_')]
    print(f"   Total attributes: {len(important_attrs)}")
    
    # Check for key methods
    has_predict = hasattr(obj, 'predict')
    has_predict_proba = hasattr(obj, 'predict_proba')
    has_fit = hasattr(obj, 'fit')
    has_transform = hasattr(obj, 'transform')
    has_steps = hasattr(obj, 'steps')
    
    print(f"   Has 'predict' method: {has_predict}")
    print(f"   Has 'predict_proba' method: {has_predict_proba}")
    print(f"   Has 'fit' method: {has_fit}")
    print(f"   Has 'transform' method: {has_transform}")
    print(f"   Has 'steps' attribute: {has_steps}")
    
    # If it's a pipeline, show steps
    if has_steps:
        print(f"\n5. Pipeline Steps:")
        try:
            for i, (name, step) in enumerate(obj.steps):
                print(f"   {i+1}. {name}: {type(step).__name__}")
        except Exception as e:
            print(f"   ❌ Could not read steps: {e}")
    
    # Try a test prediction if possible
    if has_predict:
        print(f"\n6. Testing Prediction:")
        try:
            import pandas as pd
            import numpy as np
            
            # Create test data
            test_data = pd.DataFrame([{
                "GenHlth": 3, "HighBP": 1, "DiffWalk": 0, "BMI": 28.0,
                "HighChol": 1, "Age": 9, "HeartDiseaseorAttack": 0,
                "PhysHlth": 5, "Income": 4, "Education": 5, "PhysActivity": 1
            }])
            
            print(f"   Test data shape: {test_data.shape}")
            print(f"   Test data columns: {list(test_data.columns)}")
            
            pred = obj.predict(test_data)
            print(f"   ✅ Prediction successful!")
            print(f"   Prediction: {pred}")
            
            if has_predict_proba:
                proba = obj.predict_proba(test_data)
                print(f"   Probabilities: {proba}")
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            import traceback
            print("\n   Full traceback:")
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    if isinstance(obj, dict):
        print("   ⚠️ Your pickle contains a dictionary, not a model")
        print("   → Check which key contains the actual pipeline")
        print("   → You may need to modify how the model was saved")
    elif not has_predict:
        print("   ⚠️ The object doesn't have a 'predict' method")
        print("   → This is not a trained model")
        print("   → Check your model training/saving code")
    else:
        print("   ✅ Object looks like a valid model!")
        print("   → If predictions still fail, check feature order/names")

if __name__ == "__main__":
    diagnose()