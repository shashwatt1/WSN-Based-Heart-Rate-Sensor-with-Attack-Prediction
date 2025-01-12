import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


# Function to train the model
def train_model(df):
    """
    This function trains a RandomForestClassifier using the preprocessed dataset.
    It returns the trained model, accuracy, classification report, and confusion matrix.
    """
    # Define features (X) and target variable (y)
    X = df.drop('Heart Attack Risk', axis=1)  # Assuming 'Heart Attack Risk' is the target variable
    y = df['Heart Attack Risk']  # Target variable
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get the classification report (precision, recall, f1-score)
    report = classification_report(y_test, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, report, cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    """
    This function takes the confusion matrix as input and plots it using seaborn heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)  # Use st.pyplot() to display the plot in Streamlit
   

# Function to get feature importance
def get_feature_importance(model, df):
    # Define feature names (exclude target column 'Heart Attack Risk')
    feature_names = df.drop('Heart Attack Risk', axis=1).columns  # Exclude the target variable
    
    importances = None

    # If the model has feature_importances_ (e.g., RandomForest or GradientBoosting)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For models like LogisticRegression or LinearSVC
        importances = model.coef_[0]  # For binary classification

    if importances is None:
        raise ValueError("Model does not have 'feature_importances_' or 'coef_' attribute.")

    # Debugging: Print feature names and importance lengths
    print(f"Feature Names Length: {len(feature_names)}")
    print(f"Importance Length: {len(importances)}")

    # Ensure the number of features matches the length of importance values
    if len(feature_names) != len(importances):
        # Debugging: print the feature names and importances for comparison
        print(f"Feature Names: {feature_names}")
        print(f"Importances: {importances}")
        raise ValueError("Number of features does not match the length of importance values.")

    # Create a DataFrame with features and their importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    return importance_df
