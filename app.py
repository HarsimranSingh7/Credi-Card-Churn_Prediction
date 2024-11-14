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

# Define input fields for all features
customer_age = st.number_input("Customer Age:", min_value=18, max_value=100, value=30)
total_trans_amt = st.number_input("Total Transaction Amount:", min_value=0, value=1000)
total_amt_chng_q4_q1 = st.number_input("Transaction Amount Change Q4 to Q1:", min_value=0.0, value=1.2)
total_trans_ct = st.number_input("Total Transaction Count:", min_value=0, value=50)
total_ct_chng_q4_q1 = st.number_input("Transaction Count Change Q4 to Q1:", min_value=0.0, value=0.8)
total_revolving_bal = st.number_input("Total Revolving Balance:", min_value=0, value=500)
total_relationship_count = st.number_input("Total Relationship Count:", min_value=0, max_value=10, value=3)
credit_limit = st.number_input("Credit Limit:", min_value=0, value=15000)
avg_open_to_buy = st.number_input("Average Open to Buy:", min_value=0, value=12000)
contacts_count_12_mon = st.number_input("Contacts Count in 12 Months:", min_value=0, max_value=12, value=3)
months_on_book = st.number_input("Months on Book:", min_value=0, max_value=60, value=24)
months_inactive_12_mon = st.number_input("Months Inactive in 12 Months:", min_value=0, max_value=12, value=2)
avg_utilization_ratio = st.number_input("Average Utilization Ratio:", min_value=0.0, max_value=1.0, value=0.25)
dependent_count = st.number_input("Dependent Count:", min_value=0, max_value=10, value=1)
gender = st.selectbox("Gender:", ["Male", "Female"])
marital_status = st.selectbox("Marital Status:", ["Single", "Married", "Divorced", "Unknown"])
education_level = st.selectbox("Education Level:", ["Uneducated", "High School", "Graduate", "Post-Graduate", "Doctorate", "Unknown"])
income_category = st.selectbox("Income Category:", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])
card_category = st.selectbox("Card Category:", ["Blue", "Silver", "Gold", "Platinum"])

# Encode categorical variables as they were during model training
gender = 1 if gender == "Male" else 0
marital_status_map = {"Single": 0, "Married": 1, "Divorced": 2, "Unknown": -1}
education_level_map = {"Uneducated": 0, "High School": 1, "Graduate": 2, "Post-Graduate": 3, "Doctorate": 4, "Unknown": -1}
income_category_map = {"Less than $40K": 0, "$40K - $60K": 1, "$60K - $80K": 2, "$80K - $120K": 3, "$120K +": 4, "Unknown": -1}
card_category_map = {"Blue": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

# Create a dictionary of features
features = {
    "Customer_Age": customer_age,
    "Total_Trans_Amt": total_trans_amt,
    "Total_Amt_Chng_Q4_Q1": total_amt_chng_q4_q1,
    "Total_Trans_Ct": total_trans_ct,
    "Total_Ct_Chng_Q4_Q1": total_ct_chng_q4_q1,
    "Total_Revolving_Bal": total_revolving_bal,
    "Total_Relationship_Count": total_relationship_count,
    "Credit_Limit": credit_limit,
    "Avg_Open_To_Buy": avg_open_to_buy,
    "Contacts_Count_12_mon": contacts_count_12_mon,
    "Months_on_book": months_on_book,
    "Months_Inactive_12_mon": months_inactive_12_mon,
    "Avg_Utilization_Ratio": avg_utilization_ratio,
    "Dependent_count": dependent_count,
    "Gender": gender,
    "Marital_Status": marital_status_map[marital_status],
    "Education_Level": education_level_map[education_level],
    "Income_Category": income_category_map[income_category],
    "Card_Category": card_category_map[card_category]
}

# Predict churn probability
if st.button("Predict Churn Probability"):
    # Convert features into a DataFrame with the exact order of columns
    input_data = pd.DataFrame([features])
    
    try:
        # Calculate churn probability
        churn_probability = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of Churning: {churn_probability:.2%}")
        
        # SHAP explanation
        shap_values = explainer.shap_values(input_data)
        st.write("SHAP Explanation:")
        
        # Use matplotlib to display the SHAP force plot
        shap.force_plot(explainer.expected_value[1], shap_values[1], input_data, matplotlib=True)
        st.pyplot()
        
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        st.write("Please check that all input fields match the format expected by the model.")
