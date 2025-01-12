import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df):
    """
    This function preprocesses the data by handling missing values,
    encoding categorical features, and scaling numerical features if needed.
    """
    # Handle missing values (if any)
    df.fillna(method='ffill', inplace=True)

    # Encoding categorical variables
    label_encoder = LabelEncoder()

    # Encode Gender (binary column)
    df['Gender'] = label_encoder.fit_transform(df['Gender'])

    # Apply Label Encoding for binary columns
    binary_columns = ['Smoking', 'Alcohol Consumption', 'Previous Heart Conditions',
                      'Family History of Heart Disease', 'Diabetes History', 'Cholesterol Levels', 'Blood Pressure']
    for col in binary_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # One-Hot Encoding for multi-class categorical variables
    one_hot_columns = ['Physical Activity Type', 'Stress Levels', 'Diet']
    df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)

    # Normalize or scale numeric features if needed (optional)
    # Example: Scaling age, height, weight, etc.
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # numeric_columns = ['Age', 'Height (cm)', 'Weight (kg)', 'Heart Rate (bpm)', 'Sleep Duration (hours)', 'Work Hours Per Week']
    # df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df
