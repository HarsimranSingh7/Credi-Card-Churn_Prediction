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
    "Education_Level": education_level,
    "Marital_Status": marital_status,
    "Income_Category": income_category
}

# Predict churn probability
if st.button("Predict Churn Probability"):
    # Convert features into a DataFrame
    input_data = pd.DataFrame([features])
    
    # Calculate churn probability
    churn_probability = model.predict_proba(input_data)[0][1]
    st.write(f"Probability of Churning: {churn_probability:.2%}")
    
    # SHAP explanation
    shap_values = explainer.shap_values(input_data)
    st.write("SHAP Explanation:")
    shap.initjs()
    st_shap = shap.force_plot(explainer.expected_value[1], shap_values[1], input_data, matplotlib=True)
    st.pyplot(st_shap)
