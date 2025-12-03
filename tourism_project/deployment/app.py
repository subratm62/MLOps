import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Package Purchase Predictor", layout="centered")

# ------------------------------
# Load Model from Hugging Face Hub
# ------------------------------

REPO_ID = "subratm62/tourism-project"

# Download model pipeline
model_path = hf_hub_download(repo_id=REPO_ID, filename="best_tourism_model.joblib")
model = joblib.load(model_path)

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Wellness Tourism Package Purchase Predictor")
st.write("Predict whether a customer is likely to purchase the **Wellness Tourism Package** based on their details.")

st.markdown("---")

# ------------------------------
# Create Input Form
# ------------------------------

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=1, max_value=100, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
    Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Business", "Free Lancer"])

with col2:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=1, max_value=10, value=2)
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Queen"])

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("Number of Trips per year", min_value=0, max_value=50, value=2)
    Passport = st.selectbox("Passport Available?", ["Yes", "No"])
    
with col4:
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP"])

MonthlyIncome = st.number_input("Monthly Income (₹)", min_value=0.0, value=25000.0)

st.markdown("---")

# ------------------------------
# Prepare input for model
# ------------------------------

input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,

    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,

    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0
}])

# Set the classification threshold
classification_threshold = 0.45

# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict Purchase Likelihood"):
    proba = model.predict_proba(input_data)[0, 1]
    prediction = 1 if proba >= classification_threshold else 0

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f" The customer is **LIKELY to buy** the Wellness Tourism Package.")
    else:
        st.error(f"❗ The customer is **NOT likely to buy** the package.")

    st.write(f"**Predicted Probability:** {proba:.4f}")
    st.write(f"**Decision Threshold:** {classification_threshold:.2f}")
