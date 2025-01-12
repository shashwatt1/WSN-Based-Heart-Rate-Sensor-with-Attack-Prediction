import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessor import preprocess_data
from helper import train_model, get_feature_importance

# Function to plot confusion matrix with a more professional layout
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['True Negative', 'True Positive'], 
                annot_kws={"size": 16})
    ax.set_title("Confusion Matrix", fontsize=18)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    st.pyplot(fig)

# Function to plot Feature Importance
def plot_feature_importance(feature_importance_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    ax.set_title("Feature Importance", fontsize=18)
    ax.set_xlabel("Importance", fontsize=14)
    ax.set_ylabel("Features", fontsize=14)
    st.pyplot(fig)

# Load the dataset
@st.cache_data  # Cache the dataframe to speed up the app
def load_data():
    df = pd.read_csv('heart_rate_prediction_dataset.csv')
    return df

# App layout
def main():
    st.title("Heart Attack Prediction Dashboard")

    # Load data
    df = load_data()

    # Show dataset in the app
    st.write("### Heart Attack Dataset", df.head())

    # Display some descriptive statistics
    st.write("### Descriptive Statistics", df.describe())

    # Data Preprocessing
    st.write("### Data Preprocessing")
    df_processed = preprocess_data(df)

    # Debugging: Print columns of the processed dataframe
    print(f"Processed feature names: {df_processed.columns}")
    st.write("Processed Data", df_processed.head())

    # Train the model
    model, accuracy, report, cm = train_model(df_processed)

    # Display model accuracy and classification report
    st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
    st.text_area("Classification Report", report, height=200)

    # Display confusion matrix
    plot_confusion_matrix(cm)

    # Display Feature Importance
    st.write("### Feature Importance")
    
    # Debugging: Print feature names and the importance lengths
    feature_df = get_feature_importance(model, df_processed)
    
    # Debugging: Print feature importance DataFrame
    print(f"Feature importance DataFrame: \n{feature_df}")
    
    plot_feature_importance(feature_df)

if __name__ == "__main__":
    main()
