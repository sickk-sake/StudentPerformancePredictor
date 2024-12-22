import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "Student_Study_Hours_Dataset.csv"  # Ensure this is the correct file path
dataset = pd.read_csv(file_path)

# Select the independent variables (features) and dependent variable (Marks)
X = dataset[['Hours_Studied', 'Attendance_Percentage', 'Health_Status', 'Extracurricular', 
             'Stress_Level', 'Sleep_Duration', 'Internet_Availability']]  # Multiple features
y = dataset['Marks']  # Target variable

# Scaling the features (important for better performance of models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% for training and 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance on Test Data:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

# Function to get user input and make a prediction using linear regression and if-else logic
def predict_marks_using_combined_approach():
    print("Enter the following details to predict Marks:")
    
    # Collect inputs from the user
    hours_studied = float(input("Hours Studied: "))
    attendance_percentage = float(input("Attendance Percentage (1-100): "))  # Now input is from 1-100
    health_status = int(input("Health Status (1: Poor, 2: Average, 3: Good): "))
    extracurricular = int(input("Extracurricular Activities (1: None, 2: Some, 3: Active): "))
    stress_level = int(input("Stress Level (1-10): "))
    sleep_duration = float(input("Sleep Duration (in hours): "))
    internet_availability = int(input("Internet Availability (1: Poor, 2: Average, 3: Good): "))

    # Create a feature array based on the user's input
    user_input = pd.DataFrame([[hours_studied, attendance_percentage, health_status, extracurricular, 
                                stress_level, sleep_duration, internet_availability]], 
                              columns=['Hours_Studied', 'Attendance_Percentage', 'Health_Status', 'Extracurricular', 
                                       'Stress_Level', 'Sleep_Duration', 'Internet_Availability'])

    # Scale the input using the same scaler
    user_input_scaled = scaler.transform(user_input)

    # Predict the Marks using linear regression
    predicted_marks = model.predict(user_input_scaled)

    # Apply if-else logic to adjust the predicted marks

    # Extracurricular activities impact
    if extracurricular == 3:  # If the student is active in extracurricular activities
        predicted_marks += 5  # Increase marks by 5
    elif extracurricular == 2:  # If the student participates in some extracurricular activities
        predicted_marks += 2  # Increase marks by 2

    # Health status impact
    if health_status == 3:  # If health is good
        predicted_marks += 5  # Increase marks by 5
    elif health_status == 2:  # If health is average
        predicted_marks += 2  # Increase marks by 2

    # Stress level impact
    if stress_level <= 3:  # If stress level is low
        predicted_marks += 5  # Increase marks by 5
    elif stress_level <= 6:  # If stress level is moderate
        predicted_marks += 2  # Increase marks by 2
    else:  # If stress level is high
        predicted_marks -= 3  # Decrease marks by 3

    # Sleep duration impact
    if sleep_duration >= 7:  # If sleep is sufficient
        predicted_marks += 3  # Increase marks by 3
    elif sleep_duration >= 5:  # If sleep is average
        predicted_marks += 1  # Increase marks by 1
    else:  # If sleep is poor
        predicted_marks -= 2  # Decrease marks by 2

    # Ensure that marks don't exceed 100 and are at least 0
    predicted_marks = min(max(predicted_marks[0], 0), 100)

    print(f"Predicted Marks: {predicted_marks:.2f}")

# Call the function to make a prediction
predict_marks_using_combined_approach()
