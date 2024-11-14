import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and expected features
model = joblib.load('lightgbm_churn_model.pkl')  # Ensure this file is in the same directory
expected_features = model.booster_.feature_name()

# Define the StandardScaler used during training
scaler = StandardScaler()
scaler.mean_ = [45.1, 0.45, 2.5, 30.5, 3.0, 2.3, 7500, 1500, 5000, 0.5, 50, 0.3, 10000, 0.25]  # example mean
scaler.scale_ = [10, 0.5, 1.5, 15, 1.5, 0.8, 2000, 500, 1000, 0.1, 15, 0.1, 2000, 0.15]  # example scale

# Streamlit app setup
st.title("Customer Churn Prediction App")
st.write("Enter the customer's information to predict churn probability.")

# Input fields
customer_age = st.number_input("Customer Age:", min_value=18, max_value=100, value=45)
gender = st.selectbox("Gender:", ["Male", "Female"])
dependent_count = st.number_input("Dependent Count:", min_value=0, max_value=10, value=2)
education_level = st.selectbox("Education Level:", ["Uneducated", "High School", "Graduate", "Post-Graduate", "Doctorate", "Unknown"])
marital_status = st.selectbox("Marital Status:", ["Single", "Married", "Divorced", "Unknown"])
income_category = st.selectbox("Income Category:", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])
total_trans_amt = st.number_input("Total Transaction Amount:", min_value=0, value=5000)
total_amt_chng_q4_q1 = st.number_input("Transaction Amount Change Q4 to Q1:", min_value=0.0, max_value=5.0, value=1.5)
total_trans_ct = st.number_input("Total Transaction Count:", min_value=0, value=50)
total_ct_chng_q4_q1 = st.number_input("Transaction Count Change Q4 to Q1:", min_value=0.0, max_value=2.0, value=0.7)
total_revolving_bal = st.number_input("Total Revolving Balance:", min_value=0, value=1500)
total_relationship_count = st.number_input("Total Relationship Count:", min_value=0, max_value=6, value=3)
credit_limit = st.number_input("Credit Limit:", min_value=0, value=10000)
avg_utilization_ratio = st.number_input("Average Utilization Ratio:", min_value=0.0, max_value=1.0, value=0.25)

# Dictionary of features
features = {
    "Customer_Age": customer_age,
    "Gender": 1 if gender == "Male" else 0,
    "Dependent_count": dependent_count,
    f"Education_Level_{education_level}": 1,
    f"Marital_Status_{marital_status}": 1,
    f"Income_Category_{income_category}": 1,
    "Total_Trans_Amt": total_trans_amt,
    "Total_Amt_Chng_Q4_Q1": total_amt_chng_q4_q1,
    "Total_Trans_Ct": total_trans_ct,
    "Total_Ct_Chng_Q4_Q1": total_ct_chng_q4_q1,
    "Total_Revolving_Bal": total_revolving_bal,
    "Total_Relationship_Count": total_relationship_count,
    "Credit_Limit": credit_limit,
    "Avg_Utilization_Ratio": avg_utilization_ratio
}

# Function to prepare input data with consistent feature columns
def prepare_input_data(features, expected_features, scaler):
    # Convert features to DataFrame
    input_data = pd.DataFrame([features])

    # Add missing columns
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0  # Fill missing column with 0

    # Reorder to match model's expected features
    input_data = input_data[expected_features]

    # Apply scaling to numerical columns
    numerical_features = ["Customer_Age", "Dependent_count", "Total_Trans_Amt", 
                          "Total_Revolving_Bal", "Credit_Limit", "Avg_Utilization_Ratio",
                          "Total_Amt_Chng_Q4_Q1", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1"]
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    return input_data

# Prepare input data
input_data_prepared = prepare_input_data(features, expected_features, scaler)

# Predict churn probability
if st.button("Predict Churn Probability"):
    churn_probability = model.predict_proba(input_data_prepared)[0][1]
    st.write(f"Probability of Churning: {churn_probability:.2%}")

    # Display input data for verification
    st.write("Debugging Information:")
    st.write("Prepared Input Data:", input_data_prepared)
