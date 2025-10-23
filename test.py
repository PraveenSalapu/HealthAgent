"""
Test script to verify how data flows through your trained pipeline.
This helps understand the preprocessing steps.

Usage:
    python test.py
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = "model_output/ml_xgboost_smoteenn_pipeline.pkl"

# Test input (same order as training)
test_input = {
    "GenHlth": 3,           # Good health
    "HighBP": 1,            # Has high BP
    "DiffWalk": 0,          # No difficulty walking
    "BMI": 28.0,            # Slightly overweight
    "HighChol": 1,          # Has high cholesterol
    "Age": 9,               # Age category 9 (60-64 years)
    "HeartDiseaseorAttack": 0,  # No heart disease
    "PhysHlth": 5,          # 5 days of poor health
    "Income": 4,            # Middle income
    "Education": 5,         # Some college
    "PhysActivity": 1       # Physically active
}

def main():
    print("="*70)
    print("TESTING PIPELINE DATA FLOW")
    print("="*70)
    
    # Load model
    print(f"\n1. Loading model from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Show pipeline structure
    print("\n2. Pipeline Structure:")
    for i, (name, step) in enumerate(model.steps):
        print(f"   Step {i+1}: {name} ({type(step).__name__})")
    
    # Create input dataframe
    print("\n3. Creating input DataFrame...")
    input_df = pd.DataFrame([test_input])
    print("\n   Raw Input:")
    print(input_df.T)
    
    # Step-by-step transformation
    print("\n4. Step-by-Step Transformation:")
    
    # Step 1: Preprocessor (ColumnTransformer)
    preprocessor = model.named_steps['preprocessor']
    transformed = preprocessor.transform(input_df)
    print(f"\n   After Preprocessor (ColumnTransformer):")
    print(f"   Shape: {transformed.shape}")
    print(f"   Values: {transformed[0]}")
    
    # Show which features are scaled
    print("\n   Feature Processing:")
    print("   - Numeric features (SCALED): BMI, GenHlth, PhysHlth, Age, Education, Income")
    print("   - Binary features (PASSTHROUGH): HighBP, DiffWalk, HighChol, HeartDiseaseorAttack, PhysActivity")
    
    # Step 2: Sampler (only active during training)
    print("\n   SMOTEENN Sampler: SKIPPED (only active during .fit(), not .predict())")
    
    # Step 3: Make prediction
    print("\n5. Making Prediction...")
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        
        print(f"\n   Prediction: {prediction} ({'Diabetes' if prediction == 1 else 'No Diabetes'})")
        print(f"   Probabilities:")
        print(f"     - No Diabetes: {proba[0]:.4f} ({proba[0]*100:.2f}%)")
        print(f"     - Diabetes:    {proba[1]:.4f} ({proba[1]*100:.2f}%)")
        
        # Risk interpretation
        diabetes_prob = proba[1] * 100
        if diabetes_prob < 30:
            risk = "LOW"
        elif diabetes_prob < 60:
            risk = "MODERATE"
        else:
            risk = "HIGH"
        
        print(f"\n   Risk Level: {risk}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with different inputs
    print("\n" + "="*70)
    print("6. Testing with Multiple Scenarios:")
    print("="*70)
    
    scenarios = [
        {
            "name": "Low Risk Profile",
            "data": {"GenHlth": 1, "HighBP": 0, "DiffWalk": 0, "BMI": 22.0, 
                    "HighChol": 0, "Age": 3, "HeartDiseaseorAttack": 0, 
                    "PhysHlth": 0, "Income": 7, "Education": 6, "PhysActivity": 1}
        },
        {
            "name": "High Risk Profile",
            "data": {"GenHlth": 5, "HighBP": 1, "DiffWalk": 1, "BMI": 35.0, 
                    "HighChol": 1, "Age": 13, "HeartDiseaseorAttack": 1, 
                    "PhysHlth": 20, "Income": 2, "Education": 2, "PhysActivity": 0}
        }
    ]
    
    for scenario in scenarios:
        df_scenario = pd.DataFrame([scenario["data"]])
        pred = model.predict(df_scenario)[0]
        prob = model.predict_proba(df_scenario)[0][1]
        
        print(f"\n{scenario['name']}:")
        print(f"  Diabetes Risk: {prob*100:.1f}%")
        print(f"  Prediction: {'Diabetes' if pred == 1 else 'No Diabetes'}")
    
    print("\n" + "="*70)
    print("✅ Pipeline test complete!")
    print("="*70)

if __name__ == "__main__":
    main()