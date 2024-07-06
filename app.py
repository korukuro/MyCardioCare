import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText

# Load environment variables from .env file
load_dotenv()

# Function to send an email
def send_email(subject, body, to):
    gmail_user = 'mycardiocareindia@gmail.com'
    gmail_password = os.getenv('MAIL_PASSWORD')

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = gmail_user
    msg['To'] = to

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(gmail_user, gmail_password)
    server.send_message(msg)
    server.quit()

# Load the data
dataframe = pd.read_csv("static/myheart.csv")
df = pd.DataFrame(dataframe)
df = df.astype(str)

# Encode categorical features
columns_to_encode = ['General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 'Sex', 'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'Smoking_History', 'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']

# Create a dictionary to store the label encoders for each column
label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Separate features and target variable
X = df.iloc[:, 0:17].values
Y = df.iloc[:, 17].values

# Encode the target variable
target_encoder = LabelEncoder()
Y = target_encoder.fit_transform(Y)

# Initialize the random forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, Y)

# Streamlit app
st.title("My CardioCare App")
st.write("Enter your details to predict the possibility of heart disease.")

# Collect user inputs
customer_name = st.text_input("Customer Name")
customer_email = st.text_input("Customer Email")
general_health = st.selectbox("General Health", df['General_Health'].unique())
checkup = st.selectbox("Checkup", df['Checkup'].unique())
exercise = st.selectbox("Exercise", df['Exercise'].unique())
skin_cancer = st.selectbox("Skin Cancer", df['Skin_Cancer'].unique())
other_cancer = st.selectbox("Other Cancer", df['Other_Cancer'].unique())
depression = st.selectbox("Depression", df['Depression'].unique())
diabetes = st.selectbox("Diabetes", df['Diabetes'].unique())
arthritis = st.selectbox("Arthritis", df['Arthritis'].unique())
sex = st.selectbox("Sex", df['Sex'].unique())
age = st.number_input("Age", min_value=0, max_value=120, step=1)
height = st.number_input("Height (cm)", min_value=0.0, max_value=300.0, step=0.1)
weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, step=0.1)
smoking = st.selectbox("Smoking History", df['Smoking_History'].unique())
alcohol = st.selectbox("Alcohol Consumption", df['Alcohol_Consumption'].unique())
fruit = st.selectbox("Fruit Consumption", df['Fruit_Consumption'].unique())
vegetable = st.selectbox("Green Vegetables Consumption", df['Green_Vegetables_Consumption'].unique())
potato = st.selectbox("Fried Potato Consumption", df['FriedPotato_Consumption'].unique())

def age_cat(age):
    if 18 <= age <= 24:
        return "18-24"
    elif 25 <= age <= 29:
        return "25-29"
    elif 30 <= age <= 34:
        return "30-34"
    elif 35 <= age <= 39:
        return "35-39"
    elif 40 <= age <= 44:
        return "40-44"
    elif 45 <= age <= 49:
        return "45-49"
    elif 50 <= age <= 54:
        return "50-54"
    elif 55 <= age <= 59:
        return "55-59"
    elif 60 <= age <= 64:
        return "60-64"
    elif 65 <= age <= 69:
        return "65-69"
    elif 70 <= age <= 74:
        return "70-74"
    elif 75 <= age <= 79:
        return "75-79"
    elif age >= 80:
        return "80+"
    else:
        return "Invalid age"

def categorize_weight(weight_kg):
    if weight_kg >= 100:
        return "100+"
    elif 85 <= weight_kg < 100:
        return "85-100"
    elif 76.5 <= weight_kg < 85:
        return "76.5-85"
    elif 70 <= weight_kg < 76.5:
        return "70-76.5"
    elif 64 <= weight_kg < 70:
        return "64-70"
    elif 59 <= weight_kg < 64:
        return "59-64"
    elif 55 <= weight_kg < 59:
        return "55-59"
    elif 52 <= weight_kg < 55:
        return "52-55"
    elif 48 <= weight_kg < 52:
        return "48-52"
    elif weight_kg < 48:
        return "0-48"

def categorize_height(height_cm):
    if height_cm >= 210:
        return "210 cm and above"
    elif 200 <= height_cm < 210:
        return "200-209 cm"
    elif 190 <= height_cm < 200:
        return "190-199 cm"
    elif 180 <= height_cm < 190:
        return "180-189 cm"
    elif 170 <= height_cm < 180:
        return "170-179 cm"
    elif 160 <= height_cm < 170:
        return "160-169 cm"
    elif 150 <= height_cm < 160:
        return "150-159 cm"
    elif 140 <= height_cm < 150:
        return "140-149 cm"
    elif height_cm < 140:
        return "Up to 139 cm"

# Predict button
if st.button("Predict"):
    try:
        age_category = age_cat(age)
        
        input_data = {
            'General_Health': general_health.capitalize(),
            'Checkup': checkup,
            'Exercise': exercise.capitalize(),
            'Skin_Cancer': skin_cancer.capitalize(),
            'Other_Cancer': other_cancer.capitalize(),
            'Depression': depression.capitalize(),
            'Diabetes': diabetes.capitalize(),
            'Arthritis': arthritis.capitalize(),
            'Sex': sex.capitalize(),
            'Age_Category': age_category.capitalize(),
            'Height_(cm)': categorize_height(height),
            'Weight_(kg)': categorize_weight(weight),
            'Smoking_History': smoking.capitalize(),
            'Alcohol_Consumption': alcohol.capitalize(),
            'Fruit_Consumption': fruit.capitalize(),
            'Green_Vegetables_Consumption': vegetable.capitalize(),
            'FriedPotato_Consumption': potato.capitalize(),
        }
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df.astype(str)

        # Transform categorical columns using label_encoders
        for column in columns_to_encode:
            input_df[column] = label_encoders[column].transform(input_df[column])

        # Make prediction using the trained model
        prediction = rf_classifier.predict(input_df.values)[0]
        predicted_label = target_encoder.inverse_transform([prediction])[0]
        
        # Prepare the message to send
        if predicted_label == 'Yes':
            message = f"Dear {customer_name},\n\nBased on the information you provided, our analysis suggests a potential concern regarding heart disease. However, please note that this is a prediction and not a definitive diagnosis.\n\nWe strongly recommend consulting with your healthcare provider for further evaluation and guidance.\n\nBest regards,\nTeam myCardioCare"
        else:
            message = f"Dear {customer_name},\n\nBased on the information you provided, our analysis does not indicate a possibility of heart disease. However, it is important to consult with your healthcare provider for a comprehensive evaluation.\n\nBest regards,\nTeam myCardioCare"

        # Send email to the customer
        send_email('Disease Prediction Result', message, customer_email)
        
        st.success('Prediction: ' + predicted_label)
        st.write(message)
    
    except Exception as e:
        st.error('An error occurred: ' + str(e))
