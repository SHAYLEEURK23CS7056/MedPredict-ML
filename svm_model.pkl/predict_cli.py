import pickle
import numpy as np

# Load the trained model
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature order (same as in dataset)
features = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

print("\n Diabetes Risk Prediction")
print("Please enter the following medical details:")

# Take input from user
input_data = []
for feature in features:
    value = float(input(f"{feature}: "))
    input_data.append(value)

# Convert to numpy array and reshape for prediction
input_array = np.array(input_data).reshape(1, -1)

# Make prediction
prediction = model.predict(input_array)


# Show result
if prediction[0] == 1:
    print("\n Risk Detected: The patient is likely to have diabetes.")
else:
    print("\n No Risk Detected: The patient is not likely to have diabetes.")
