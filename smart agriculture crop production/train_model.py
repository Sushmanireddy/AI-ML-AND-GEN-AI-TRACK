import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load cleaned dataset
data = pd.read_csv("dataset/cleaned_crop_data.csv")

# Features and target
X = data.drop("label", axis=1)
y = data["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")