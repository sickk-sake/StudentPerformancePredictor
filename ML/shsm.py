import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#dataset
file_path = "Student_Study_Hours_Dataset.csv"  # Ensure this is the correct file path
dataset = pd.read_csv(file_path)

X = dataset[['Hours_Studied', 'Attendance_Percentage', 'Health_Status', 'Extracurricular', 
             'Stress_Level', 'Sleep_Duration', 'Internet_Availability']] 
y = dataset['Marks'] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)

#prediction
def predict_marks():
    print("Enter the following details to predict Marks:")
    
    #userinput
    hours_studied = float(input("Hours Studied: "))
    attendance_percentage = float(input("Attendance Percentage: "))
    health_status = float(input("Health Status (1: Poor, 2: Average, 3: Good): "))
    extracurricular = float(input("Extracurricular Activities (1: None, 2: Some, 3: Active): "))
    stress_level = float(input("Stress Level (1-10): "))
    sleep_duration = float(input("Sleep Duration (in hours): "))
    internet_availability = float(input("Internet Availability (1: Poor, 2: Average, 3: Good): "))

    # Create a feature array based on the user's input
    user_input = pd.DataFrame([[hours_studied, attendance_percentage, health_status, extracurricular, 
                                stress_level, sleep_duration, internet_availability]], 
                              columns=['Hours_Studied', 'Attendance_Percentage', 'Health_Status', 'Extracurricular', 
                                       'Stress_Level', 'Sleep_Duration', 'Internet_Availability'])

    user_input_scaled = scaler.transform(user_input)

    #usermarks
    predicted_marks = model.predict(user_input_scaled)

    #maxmarks = 100
    predicted_marks = min(predicted_marks[0], 100)

    print(f"Predicted Marks: {predicted_marks:.2f}")

# Call the function
predict_marks()
