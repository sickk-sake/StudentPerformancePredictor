import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Number of rows for the dataset
n_rows = 100

# Generate data
hours_studied = np.round(np.random.uniform(1, 10, n_rows), 1)  # Study hours between 1 and 10
attendance = np.random.uniform(50, 100, n_rows)  # Attendance percentage between 50% and 100%
health_status = np.random.choice([0, 1], size=n_rows, p=[0.3, 0.7])  # 70% healthy, 30% not healthy
extracurricular = np.random.choice([0, 1], size=n_rows, p=[0.5, 0.5])  # 50% participation
stress_level = np.random.randint(1, 11, n_rows)  # Stress level between 1 and 10
sleep_duration = np.round(np.random.uniform(4, 10, n_rows), 1)  # Sleep hours between 4 and 10
internet_availability = np.random.choice([0, 1], size=n_rows, p=[0.2, 0.8])  # 80% have internet

# Define weights for marks (target variable)
weights = {
    "hours_studied": 7.5,
    "attendance": 0.1,
    "health_status": 5,
    "extracurricular": 3,
    "stress_level": -2,
    "sleep_duration": 2,
    "internet_availability": 5
}

# Calculate marks based on a linear combination of factors with some random noise
marks = (
    weights["hours_studied"] * hours_studied +
    weights["attendance"] * attendance +
    weights["health_status"] * health_status +
    weights["extracurricular"] * extracurricular +
    weights["stress_level"] * stress_level +
    weights["sleep_duration"] * sleep_duration +
    weights["internet_availability"] * internet_availability +
    np.random.normal(0, 5, n_rows)  # Add random noise
)

# Clip marks to be between 0 and 100
marks = np.clip(marks, 0, 100)

# Create the dataset
dataset = pd.DataFrame({
    "Hours_Studied": hours_studied,
    "Attendance_Percentage": np.round(attendance, 1),
    "Health_Status": health_status,
    "Extracurricular": extracurricular,
    "Stress_Level": stress_level,
    "Sleep_Duration": sleep_duration,
    "Internet_Availability": internet_availability,
    "Marks": np.round(marks, 1)
})

# Save the dataset to a CSV file
dataset.to_csv("Student_Study_Hours_Dataset.csv", index=False)

print("Dataset created and saved as 'Student_Study_Hours_Dataset.csv'.")
