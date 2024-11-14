import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('lightgbm_churn_model.pkl')  # Ensure this file is in the same directory as app.py

# Set up SHAP explainer
explainer = shap.TreeExplainer(model)

# Define the Streamlit app
st.title("Churn Prediction App")
st.write("Enter the customer's information to predict the probability of churn.")

# Define input fields
customer_age = st.number_input("Customer Age:", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender:", ["Male", "Female"])
dependent_count = st.number_input("Dependent Count:", min_value=0, max_value=10, value=1)
education_level = st.selectbox("Education Level:", ["Uneducated", "High School", "Graduate", "Post-Graduate", "Doctorate", "Unknown"])
marital_status = st.selectbox("Marital Status:", ["Single", "Married", "Divorced", "Unknown"])
income_category = st.selectbox("Income Category:", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])

# Create a dictionary of features
features = {
    "Customer_Age": customer_age,
    "Gender": 1 if gender == "Male" else 0,
    "Dependent_count": dependent_count,
    "Education_Level_" + education_level: 1,
    "Marital_Status_" + marital_status: 1,
    "Income_Category_" + income_category: 1
}

# Function to prepare input data with consistent feature columns
def prepare_input_data(features, feature_columns):
    # Convert the features dictionary to a DataFrame
    input_data = pd.DataFrame([features])

    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing column with 0s

    # Reorder columns to match the model's expected input order
    input_data = input_data[feature_columns]
    return input_data

# Define the exact columns that the model expects (match with the trained model's feature names)
feature_columns = [
    "Customer_Age", "Gender", "Dependent_count", "Months_on_book", 
    "Contacts_Count_12_mon", "Total_Relationship_Count", "Months_Inactive_12_mon", 
    "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Trans_Amt", 
    "Total_Trans_Ct", "Total_Amt_Chng_Q4_Q1", "Total_Ct_Chng_Q4_Q1", 
    "Marital_Status_Single", "Marital_Status_Unknown", "Income_Category_$40K - $60K", 
    "Income_Category_$60K - $80K", "Income_Category_$80K - $120K", 
    "Income_Category_Less than $40K", "Income_Category_Unknown", 
    "Education_Level_High School", "Education_Level_Doctorate", 
    "Education_Level_Graduate", "Education_Level_Uneducated"
    # Add all other columns expected by the model here
]

# Prepare input data
input_data_prepared = prepare_input_data(features, feature_columns)

# Predict churn probability
if st.button("Predict Churn Probability"):
    try:
        # Calculate churn probability
        churn_probability = model.predict_proba(input_data_prepared)[0][1]
        st.write(f"Probability of Churning: {churn_probability:.2%}")

        # SHAP explanation
        shap_values = explainer.shap_values(input_data_prepared)
        st.write("SHAP Explanation:")
        shap.initjs()
        st_shap = shap.force_plot(explainer.expected_value[1], shap_values[1], input_data_prepared, matplotlib=True)
        st.pyplot(st_shap)

    except ValueError as e:
        st.error(f"An error occurred: {str(e)}")
