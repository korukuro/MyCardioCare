import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, render_template

app = Flask(__name__)

# Create the improved dataset
data = {
    'Fever': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No'],
    'Cough': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No'],
    'Fatigue': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'Headache': ['Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
    'Sore Throat': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Yes'],
    'Runny Nose': ['No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes'],
    'Muscle Pain': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No'],
    'Disease': ['Influenza', 'Influenza', 'Common Cold', 'Common Cold', 'Dengue', 'Dengue', 'Migraine', 'Migraine', 'Covid-19', 'Covid-19', 'Strep Throat', 'Allergies']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Fever', 'Cough', 'Fatigue', 'Headache', 'Sore Throat', 'Runny Nose', 'Muscle Pain'])

# Split the data into features and target
X = df_encoded.drop('Disease', axis=1)
y = df_encoded['Disease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    fever = request.form['fever']
    cough = request.form['cough']
    fatigue = request.form['fatigue']
    headache = request.form['headache']
    sore_throat = request.form['sore_throat']
    runny_nose = request.form['runny_nose']
    muscle_pain = request.form['muscle_pain']
    
    input_data = {
        'Fever_Yes': [1 if fever == 'Yes' else 0],
        'Fever_No': [1 if fever == 'No' else 0],
        'Cough_Yes': [1 if cough == 'Yes' else 0],
        'Cough_No': [1 if cough == 'No' else 0],
        'Fatigue_Yes': [1 if fatigue == 'Yes' else 0],
        'Fatigue_No': [1 if fatigue == 'No' else 0],
        'Headache_Yes': [1 if headache == 'Yes' else 0],
        'Headache_No': [1 if headache == 'No' else 0],
        'Sore Throat_Yes': [1 if sore_throat == 'Yes' else 0],
        'Sore Throat_No': [1 if sore_throat == 'No' else 0],
        'Runny Nose_Yes': [1 if runny_nose == 'Yes' else 0],
        'Runny Nose_No': [1 if runny_nose == 'No' else 0],
        'Muscle Pain_Yes': [1 if muscle_pain == 'Yes' else 0],
        'Muscle Pain_No': [1 if muscle_pain == 'No' else 0],
    }
    
    input_df = pd.DataFrame(input_data)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    
    prediction = clf.predict(input_df)
    
    return render_template('index.html', prediction_text=f'The predicted disease is: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
    