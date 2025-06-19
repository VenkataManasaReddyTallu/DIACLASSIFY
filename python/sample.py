import joblib
import numpy as np

# Load the saved Random Forest model
model = joblib.load('random_forest_model.pkl')  # Update path if needed

# Define a sample input: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
sample_input = np.array([[2, 130, 70, 22, 94, 28.5, 0.45, 35]])  # Replace with actual test values

# Make prediction
prediction = model.predict(sample_input)
probability = model.predict_proba(sample_input)[0][1]  # Probability of being diabetic (class 1)

# Display results
print("🔍 Sample Input:", sample_input.tolist()[0])
print("📊 Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
print(f"🎯 Probability of being diabetic: {probability:.2f}")
