import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove duplicate rows
data = data.drop_duplicates()

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Save the cleaned dataset
data.to_csv("dataset/cleaned_crop_data.csv", index=False)

print("\nData preprocessing completed successfully!")