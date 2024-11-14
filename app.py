import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('lightgbm_churn_model.pkl')

# Set up SHAP explainer
explainer = shap.TreeExplainer(model)

# Define the Streamlit app
st.title("Churn Prediction App")
st.write("Enter the customer's information to predict the probability of churn.")

# Define input fields
customer_age = st.number_input("Customer Age:", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender:", ["Male", "Female"])
dependent_count = st.number_input("Dependent Count:", min_value=0, max_value=10, value=1)
months_on_book = st.number_input("Months on Book:", min_value=0, max_value=100, value=24)
total_relationship_count = st.number_input("Total Relationship Count:", min_value=0, max_value=10, value=3)
months_inactive_12_mon = st.number_input("Months Inactive in 12 Months:", min_value=0, max_value=12, value=2)
contacts_count_12_mon = st.number_input("Contacts Count in 12 Months:", min_value=0, max_value=20, value=3)
credit_limit = st.number_input("Credit Limit:", min_value=0.0, max_value=50000.0, value=10000.0)
total_revolving_bal = st.number_input("Total Revolving Balance:", min_value=0, max_value=50000, value=2000)
avg_open_to_buy = st.number_input("Average Open to Buy:", min_value=0.0, max_value=50000.0, value=3000.0)

# Dropdowns for categorical features with one-hot encoding
education_level = st.selectbox("Education Level:", ["Uneducated", "High School", "Graduate", "Post-Graduate", "Doctorate", "Unknown"])
marital_status = st.selectbox("Marital Status:", ["Single", "Married", "Divorced", "Unknown"])
income_category = st.selectbox("Income Category:", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])
card_category = st.selectbox("Card Category:", ["Blue", "Silver", "Gold", "Platinum"])

# Create a dictionary for input features
features = {
    "Customer_Age": customer_age,
    "Gender": 1 if gender == "Male" else 0,
    "Dependent_count": dependent_count,
    "Months_on_book": months_on_book,
    "Total_Relationship_Count": total_relationship_count,
    "Months_Inactive_12_mon": months_inactive_12_mon,
    "Contacts_Count_12_mon": contacts_count_12_mon,
    "Credit_Limit": credit_limit,
    "Total_Revolving_Bal": total_revolving_bal,
    "Avg_Open_To_Buy": avg_open_to_buy,
    "Education_Level_" + education_level: 1,
    "Marital_Status_" + marital_status: 1,
    "Income_Category_" + income_category: 1,
    "Card_Category_" + card_category: 1
}

# Convert features to a DataFrame and ensure all required columns are present
input_data = pd.DataFrame([features])

# Add missing columns as 0 to match model's expected input
for col in model.booster_.feature_name():
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match the model's expected input order
input_data = input_data[model.booster_.feature_name()]

# Predict churn probability
if st.button("Predict Churn Probability"):
    try:
        churn_probability = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of Churning: {churn_probability:.2%}")

        # SHAP explanation
        shap_values = explainer.shap_values(input_data)
        st.write("SHAP Explanation:")
        shap.initjs()
        st_shap = shap.force_plot(explainer.expected_value[1], shap_values[1], input_data, matplotlib=True)
        st.pyplot(st_shap)

    except ValueError as e:
        st.error(f"An error occurred: {str(e)}")
