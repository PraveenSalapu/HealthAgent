import json

# Load your file
with open('model_output/diabetic_averages.json', 'r') as f:
    averages = json.load(f)

# Check all required features
required = ["GenHlth", "HighBP", "DiffWalk", "BMI", "HighChol", 
            "Age", "HeartDiseaseorAttack", "PhysHlth", "Income", 
            "Education", "PhysActivity"]

print("Checking averages file:")
for feature in required:
    if feature in averages:
        print(f"✅ {feature}: {averages[feature]}")
    else:
        print(f"❌ MISSING: {feature}")