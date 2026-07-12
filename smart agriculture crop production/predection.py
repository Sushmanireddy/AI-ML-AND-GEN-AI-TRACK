import pickle
import numpy as np

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

# Sample input
sample = np.array([[90, 42, 43, 20.87, 82.00, 6.50, 202.94]])

prediction = model.predict(sample)

print("Predicted Crop:", prediction[0])