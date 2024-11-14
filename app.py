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

# Encode categorical variables as they were during model training
education_map = {"Uneducated": 0, "High School": 1, "Graduate": 2, "Post-Graduate": 3, "Doctorate": 4, "Unknown": -1}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Unknown": -1}
income_map = {"Less than $40K": 0, "$40K - $60K": 1, "$60K - $80K": 2, "$80K - $120K": 3, "$120K +": 4, "Unknown": -1}

# Create a dictionary of features
features = {
    "Customer_Age": customer_age,
    "Gender": 1 if gender == "Male" else 0,
    "Dependent_count": dependent_count,
    "Education_Level": education_map[education_level],
    "Marital_Status": marital_map[marital_status],
    "Income_Category": income_map[income_category]
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
    
    # Use matplotlib to display the SHAP force plot
    shap.force_plot(explainer.expected_value[1], shap_values[1], input_data, matplotlib=True)
    st.pyplot()
