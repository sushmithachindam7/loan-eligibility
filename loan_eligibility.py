# loan_eligibility.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Example dataset structure (replace with your CSV file)
# Columns: ['Gender','Married','Dependents','Education','Self_Employed',
#           'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
#           'Credit_History','Property_Area','Loan_Status']

data = pd.read_csv("loan_data.csv")

# -----------------------------
# 2. Preprocess Data
# -----------------------------
# Handle categorical variables
categorical_cols = ['Gender','Married','Dependents','Education',
                    'Self_Employed','Property_Area']

encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col].astype(str))

# Target variable
y = data['Loan_Status'].map({'Y':1,'N':0})   # Eligible=1, Not Eligible=0
X = data.drop('Loan_Status', axis=1)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Model Training
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 6. Prediction Function
# -----------------------------
def predict_eligibility(applicant_data):
    """
    applicant_data: dict with keys matching dataset columns (except Loan_Status)
    Example:
    {
      'Gender':'Male','Married':'Yes','Dependents':'0','Education':'Graduate',
      'Self_Employed':'No','ApplicantIncome':5000,'CoapplicantIncome':2000,
      'LoanAmount':150,'Loan_Amount_Term':360,'Credit_History':1,
      'Property_Area':'Urban'
    }
    """
    df = pd.DataFrame([applicant_data])
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    return "Eligible" if prediction == 1 else "Not Eligible"

# -----------------------------
# 7. Example Usage
# -----------------------------
sample_applicant = {
    'Gender':'Male','Married':'Yes','Dependents':'0','Education':'Graduate',
    'Self_Employed':'No','ApplicantIncome':5000,'CoapplicantIncome':2000,
    'LoanAmount':150,'Loan_Amount_Term':360,'Credit_History':1,
    'Property_Area':'Urban'
}

print("Prediction for sample applicant:", predict_eligibility(sample_applicant))
