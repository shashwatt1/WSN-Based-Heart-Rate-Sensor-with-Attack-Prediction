import random
import pandas as pd

# Helper function to generate random data
def generate_random_data(num_records):
    # Possible values for each column
    gender = ['Male', 'Female']
    exercise_freq = [0, 1, 2, 3, 4, 5, 6]  # Times per week
    physical_activity_type = ['Cardio', 'Strength Training', 'Yoga', 'None']
    smoking = ['Yes', 'No']
    alcohol = ['Yes', 'No']
    previous_conditions = ['Yes', 'No']
    family_history = ['Yes', 'No']
    cholesterol = ['High', 'Normal']
    blood_pressure = ['Normal', 'High']
    stress_levels = ['Low', 'Medium', 'High']
    diet = ['Healthy', 'Unhealthy']
    diabetes_history = ['Yes', 'No']
    heart_attack_risk = [0, 1]  # 0 for no, 1 for yes

    # List to hold all generated rows
    data = []

    # Generate random data
    for _ in range(num_records):
        age = random.randint(20, 80)
        height = random.randint(150, 190)  # cm
        weight = random.randint(50, 100)  # kg
        heart_rate = random.randint(60, 100)  # bpm
        exercise = random.choice(exercise_freq)
        activity_type = random.choice(physical_activity_type)
        smokes = random.choice(smoking)
        drinks = random.choice(alcohol)
        prev_conditions = random.choice(previous_conditions)
        family_disease = random.choice(family_history)
        cholesterol_level = random.choice(cholesterol)
        hdl = random.uniform(40, 60)  # HDL cholesterol (good cholesterol)
        ldl = random.uniform(100, 160)  # LDL cholesterol (bad cholesterol)
        bp = random.choice(blood_pressure)
        stress = random.choice(stress_levels)
        diet_choice = random.choice(diet)
        diabetes = random.choice(diabetes_history)
        sleep_hours = random.uniform(5, 9)  # Sleep duration in hours
        work_hours = random.randint(30, 60)  # Weekly work hours
        risk = random.choice(heart_attack_risk)

        # Create a row of data
        row = [
            age, random.choice(gender), height, weight, heart_rate, exercise, activity_type,
            smokes, drinks, prev_conditions, family_disease, cholesterol_level, hdl, ldl, bp, stress,
            diet_choice, diabetes, sleep_hours, work_hours, risk
        ]
        data.append(row)

    # Column names
    columns = [
        'Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'Heart Rate (bpm)', 'Exercise Frequency (times/week)',
        'Physical Activity Type', 'Smoking', 'Alcohol Consumption', 'Previous Heart Conditions',
        'Family History of Heart Disease', 'Cholesterol Levels', 'HDL Cholesterol', 'LDL Cholesterol',
        'Blood Pressure', 'Stress Levels', 'Diet', 'Diabetes History', 'Sleep Duration (hours)',
        'Work Hours Per Week', 'Heart Attack Risk'
    ]

    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=columns)

    return df

# Generate a dataset with 600 records
df = generate_random_data(600)

# Save to a CSV file
df.to_csv('heart_rate_prediction_dataset.csv', index=False)

print("CSV file 'heart_rate_prediction_dataset.csv' has been created!")
