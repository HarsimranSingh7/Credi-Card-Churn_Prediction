import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('lightgbm_churn_model.pkl')  # Ensure this file is in the same directory as app.py

# Get the model's expected features
expected_features = model.booster_.feature_name()

# Set up SHAP explainer
explainer = shap.TreeExplainer(model)

# Define the Streamlit app
st.title("Churn Prediction App")
st.write("Enter the customer's information to predict the probability of churn.")

# Define input fields with realistic default values
customer_age = st.number_input("Customer Age:", min_value=18, max_value=100, value=45)
gender = st.selectbox("Gender:", ["Male", "Female"])
dependent_count = st.number_input("Dependent Count:", min_value=0, max_value=10, value=2)
education_level = st.selectbox("Education Level:", ["Uneducated", "High School", "Graduate", "Post-Graduate", "Doctorate", "Unknown"])
marital_status = st.selectbox("Marital Status:", ["Single", "Married", "Divorced", "Unknown"])
income_category = st.selectbox("Income Category:", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])

# Additional input fields with typical default values
total_trans_amt = st.number_input("Total Transaction Amount:", min_value=0, value=5000)
total_amt_chng_q4_q1 = st.number_input("Transaction Amount Change Q4 to Q1:", min_value=0.0, max_value=5.0, value=1.5)
total_trans_ct = st.number_input("Total Transaction Count:", min_value=0, value=50)
total_ct_chng_q4_q1 = st.number_input("Transaction Count Change Q4 to Q1:", min_value=0.0, max_value=2.0, value=0.7)
total_revolving_bal = st.number_input("Total Revolving Balance:", min_value=0, value=1500)
total_relationship_count = st.number_input("Total Relationship Count:", min_value=0, max_value=6, value=3)
credit_limit = st.number_input("Credit Limit:", min_value=0, value=10000)
avg_utilization_ratio = st.number_input("Average Utilization Ratio:", min_value=0.0, max_value=1.0, value=0.25)

# Create a dictionary of features
features = {
    "Customer_Age": customer_age,
    "Gender": 1 if gender == "Male" else 0,
    "Dependent_count": dependent_count,
    "Education_Level_" + education_level: 1,
    "Marital_Status_" + marital_status: 1,
    "Income_Category_" + income_category: 1,
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
def prepare_input_data(features, expected_features):
    # Convert the features dictionary to a DataFrame
    input_data = pd.DataFrame([features])

    # Ensure all expected feature columns are present
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing column with 0s

    # Reorder columns to match the model's expected input order
    input_data = input_data[expected_features]
    return input_data

# Prepare input data
input_data_prepared = prepare_input_data(features, expected_features)

# Function to display SHAP plot in Streamlit
def st_shap(plot, height=None):
    from streamlit.components.v1 import html
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    html(shap_html, height=height)

# Predict churn probability
if st.button("Predict Churn Probability"):
    try:
        # Calculate churn probability
        churn_probability = model.predict_proba(input_data_prepared)[0][1]
        st.write(f"Probability of Churning: {churn_probability:.2%}")

        # SHAP explanation
        shap_values = explainer.shap_values(input_data_prepared)
        st.write("SHAP Explanation:")

        # Display the SHAP force plot using Streamlit
        if isinstance(explainer.expected_value, list):
            expected_value = explainer.expected_value[1]  # binary classification
            shap_plot = shap.force_plot(expected_value, shap_values[1], input_data_prepared)
        else:
            expected_value = explainer.expected_value
            shap_plot = shap.force_plot(expected_value, shap_values, input_data_prepared)
        
        st_shap(shap_plot, height=300)

    except ValueError as e:
        st.error(f"An error occurred: {str(e)}")
    except KeyError as e:
        st.error(f"A key error occurred, please check input columns: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
