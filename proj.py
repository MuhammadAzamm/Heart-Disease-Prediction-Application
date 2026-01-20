import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://gist.githubusercontent.com/notreallyme2/8d20f6b5bcc3606e0541cbdf2ee0a7a6/raw/heart.csv"
df = pd.read_csv(url)
print(df.head())
print(df.isnull().sum())
sns.countplot(x='target', data=df)
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
sns.histplot(data=df, x='age', hue='target', kde=True)
plt.show()
sns.countplot(x='cp', hue='target', data=df)
plt.show()
# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Example patient data
new_patient = np.array([[55,1,2,140,220,0,1,150,0,2.3,3,0,2]])  # Example values
prediction = model.predict(new_patient)
print("Heart Disease" if prediction[0]==1 else "No Heart Disease")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load dataset and train model (or load a pre-trained model)
df = pd.read_csv("heart.csv")

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("Heart Disease Prediction App")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Yes","No"])
restecg = st.number_input("Resting ECG Result (0-2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina?", ["Yes","No"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)", min_value=1, max_value=3, value=2)

# Button to predict
if st.button("Predict"):
    # Convert inputs to numbers
    sex_val = 1 if sex=="Male" else 0
    fbs_val = 1 if fbs=="Yes" else 0
    exang_val = 1 if exang=="Yes" else 0

    patient_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val,
                              restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(patient_data)
    
    if prediction[0] == 1:
        st.error("The patient may have Heart Disease!")
    else:
        st.success("The patient is likely healthy.")
