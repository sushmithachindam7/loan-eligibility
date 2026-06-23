import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("loan_data.csv")

# Encode categorical columns
categorical_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col].astype(str))

# Target and features
y = data['Loan_Status'].map({'Y':1,'N':0})
X = data.drop('Loan_Status', axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("📊 Loan Eligibility Predictor")

st.write("Enter applicant details below:")

gender = st.selectbox("Gender", ["Male","Female"])
married = st.selectbox("Married", ["Yes","No"])
dependents = st.selectbox("Dependents", ["0","1","2","3"])
education = st.selectbox("Education", ["Graduate","Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes","No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Amount Term", [360,180,120,60])
credit_history = st.selectbox("Credit History", [1,0])
property_area = st.selectbox("Property Area", ["Urban","Rural","Semiurban"])

if st.button("Predict Eligibility"):
    applicant_data = {
        'Gender':gender,'Married':married,'Dependents':dependents,'Education':education,
        'Self_Employed':self_employed,'ApplicantIncome':applicant_income,
        'CoapplicantIncome':coapplicant_income,'LoanAmount':loan_amount,
        'Loan_Amount_Term':loan_term,'Credit_History':credit_history,
        'Property_Area':property_area
    }
    df = pd.DataFrame([applicant_data])
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    result = "✅ Eligible" if prediction == 1 else "❌ Not Eligible"
    st.subheader(f"Prediction: {result}")
