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

        # For binary classification, use shap_values[0] as SHAP outputs only one set of SHAP values for binary classes
        st.write("SHAP Explanation:")

        # Display the SHAP force plot using Streamlit
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_data_prepared), height=300)

    except ValueError as e:
        st.error(f"An error occurred: {str(e)}")
    except KeyError as e:
        st.error(f"A key error occurred, please check input columns: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
