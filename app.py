import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler 

# Load the trained model and scaler
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Bank Customer Churn Prediction')

st.sidebar.header('Input Customer Data')

# Function to get user input
def get_user_input():
    credit_score = st.sidebar.slider('Credit Score', 350, 850, 650)
    country = st.sidebar.selectbox('Country', ['France', 'Germany', 'Spain'])
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    age = st.sidebar.slider('Age', 18, 92, 37)
    tenure = st.sidebar.slider('Tenure (years)', 0, 10, 5)
    balance = st.sidebar.slider('Balance', 0.0, 250000.0, 97000.0)
    products_number = st.sidebar.slider('Number of Products', 1, 4, 1)
    credit_card = st.sidebar.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    active_member = st.sidebar.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    estimated_salary = st.sidebar.slider('Estimated Salary', 0.0, 200000.0, 100000.0)

    user_data = {'credit_score': credit_score,
                 'country': country,
                 'gender': gender,
                 'age': age,
                 'tenure': tenure,
                 'balance': balance,
                 'products_number': products_number,
                 'credit_card': credit_card,
                 'active_member': active_member,
                 'estimated_salary': estimated_salary}

    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input_df = get_user_input()

# Display user input
st.subheader('User Input:')
st.write(user_input_df)

# Preprocess the user input
# One-hot encode categorical features - ensure consistent column order as training data
user_input_df = pd.get_dummies(user_input_df, columns=['country', 'gender'], drop_first=True)

# Align columns with training data - add missing columns with default value 0
train_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary', 'country_Germany', 'country_Spain', 'gender_Male']
for col in train_cols:
    if col not in user_input_df.columns:
        user_input_df[col] = 0

user_input_df = user_input_df[train_cols]


# Scale the user input
user_input_scaled = scaler.transform(user_input_df)

#StandardScaler
# Make prediction
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

st.subheader('Prediction:')
if prediction[0] == 1:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is unlikely to churn.')

st.subheader('Prediction Probability:')
st.write(f"Churn Probability: {prediction_proba[0][1]:.2f}")
st.write(f"No Churn Probability: {prediction_proba[0][0]:.2f}")
