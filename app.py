# app.py
from flask import Flask, request, render_template
from flask_mail import Mail, Message
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'mycardiocareindia@gmail.com'
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_DEFAULT_SENDER'] = 'mycardiocare@gmail.com'

mail = Mail(app)

# Load the model and encoders
rf_classifier = joblib.load('rf_classifier_compressed.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get customer name and email
        customer_name = request.form['customer_name']
        customer_email = request.form['customer_email']
        
        # Get symptoms data
        general_health = request.form['general_health'].lower()
        checkup = request.form['checkup'].lower()
        exercise = request.form['exercise'].lower()
        skin_cancer = request.form['skin_cancer'].lower()
        other_cancer = request.form['other_cancer'].lower()
        depression = request.form['depression'].lower()
        diabetes = request.form['diabetes'].lower()
        arthritis = request.form['arthritis'].lower()
        sex = request.form['sex'].lower()
        age = int(request.form['age'])
        age_category = age_cat(age)
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        smoking = request.form['smoking'].lower()
        alcohol = request.form['alcohol'].lower()
        fruit = request.form['fruit'].lower()
        vegetable = request.form['vegetable'].lower()
        potato = request.form['potato'].lower()
        
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
        for column in label_encoders:
            input_df[column] = label_encoders[column].transform(input_df[column])

        # Make prediction using the trained model
        prediction = rf_classifier.predict(input_df.values)[0]
        predicted_label = target_encoder.inverse_transform([prediction])[0]
        
        # Prepare the message to send
        if (predicted_label == 'Yes'):
            message = f"Dear {customer_name},\n\nBased on the information you provided, our analysis suggests a potential concern regarding heart disease. However, please note that this is a prediction and not a definitive diagnosis.\n\nWe strongly recommend consulting with your healthcare provider for further evaluation and guidance.\n\nBest regards,\nTeam myCardioCare"
        else:
            message = f"Dear {customer_name},\n\nBased on the information you provided, our analysis does not indicate a possibility of heart disease. However, it is important to consult with your healthcare provider for a comprehensive evaluation.\n\nBest regards,\nTeam myCardioCare"

        # Send email to the customer
        msg = Message('Disease Prediction Result', recipients=[customer_email])
        msg.body = message
        mail.send(msg)
        
        # Print prediction to console for verification
        print(message)
        
        return render_template('result.html', prediction_text=f'The possibility of heart disease is: {predicted_label}', message=message)
    
    except Exception as e:
        return render_template('result.html', prediction_text=f'An error occurred: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
