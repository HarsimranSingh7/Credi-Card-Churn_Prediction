import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('lightgbm_churn_model.pkl')

# Define the feature scaling and encoding steps as per the notebook
scaler = StandardScaler()

# Set up Streamlit app
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict the probability of churn.")

# Input fields for customer details
age = st.number_input("Customer Age", min_value=18, max_value=100)
gender = st.selectbox("Gender", options=["Male", "Female"])
dependent_count = st.number_input("Dependent Count", min_value=0, max_value=10)
months_on_book = st.number_input("Months on Book", min_value=0, max_value=120)
total_relationship_count = st.number_input("Total Relationship Count", min_value=0, max_value=10)
months_inactive_12_mon = st.number_input("Months Inactive (Last 12 Months)", min_value=0, max_value=12)
contacts_count_12_mon = st.number_input("Contacts Count (Last 12 Months)", min_value=0, max_value=20)
credit_limit = st.number_input("Credit Limit", min_value=0.0)
total_revolving_bal = st.number_input("Total Revolving Balance", min_value=0.0)
avg_open_to_buy = st.number_input("Average Open to Buy", min_value=0.0)
total_amt_chng_q4_q1 = st.number_input("Total Amount Change Q4-Q1", min_value=0.0)
total_trans_amt = st.number_input("Total Transaction Amount", min_value=0.0)
total_trans_ct = st.number_input("Total Transaction Count", min_value=0)
total_ct_chng_q4_q1 = st.number_input("Total Transaction Count Change Q4-Q1", min_value=0.0)
avg_utilization_ratio = st.number_input("Average Utilization Ratio", min_value=0.0)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Customer_Age': [age],
    'Gender': [1 if gender == "Male" else 0],
    'Dependent_count': [dependent_count],
    'Months_on_book': [months_on_book],
    'Total_Relationship_Count': [total_relationship_count],
    'Months_Inactive_12_mon': [months_inactive_12_mon],
    'Contacts_Count_12_mon': [contacts_count_12_mon],
    'Credit_Limit': [credit_limit],
    'Total_Revolving_Bal': [total_revolving_bal],
    'Avg_Open_To_Buy': [avg_open_to_buy],
    'Total_Amt_Chng_Q4_Q1': [total_amt_chng_q4_q1],
    'Total_Trans_Amt': [total_trans_amt],
    'Total_Trans_Ct': [total_trans_ct],
    'Total_Ct_Chng_Q4_Q1': [total_ct_chng_q4_q1],
    'Avg_Utilization_Ratio': [avg_utilization_ratio]
})

# Scale numerical features
input_data = scaler.transform(input_data)

# Make prediction
if st.button("Predict Churn Probability"):
    probability = model.predict_proba(input_data)[:, 1][0]
    st.write(f"The predicted probability of churn is: {probability:.2%}")
