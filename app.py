
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Risk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "diabetes_risk_model.pkl"
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #171b21;
    color: #d6dde5;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #e4eaf0 !important;
    font-family: 'Manrope', sans-serif;
}

p, label, .stMarkdown {
    color: #c0cad5;
}

.hero {
    background: #222a33;
    border: 1px solid #35414d;
    border-left: 4px solid #6baba3;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 24px;
}

.hero h1 {
    color: #edf2f7 !important;
    font-size: 2.15rem;
    margin-bottom: 10px;
}

.hero p {
    color: #aebbc8;
    font-size: 1rem;
    margin-bottom: 0;
}

.section-card {
    background: #222830;
    border: 1px solid #343e49;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #20262e;
    color: #e0e7ee;
    border-color: #414c58;
    border-radius: 9px;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #6baba3;
    box-shadow: 0 0 0 1px #6baba3;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #477f79;
    color: #ffffff;
    border: 1px solid #5a928b;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    min-height: 46px;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #568e86;
    border-color: #75aaa3;
    color: white;
}

[data-testid="stSidebar"] {
    background: #1d232b;
    border-right: 1px solid #343e49;
}

[data-testid="stSidebar"] * {
    color: #cbd5df;
}

div[data-testid="stMetric"] {
    background: #222830;
    border: 1px solid #343e49;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: #343e49;
}

.small-note {
    color: #9daab7;
    font-size: 0.88rem;
}

.result-card {
    background: #222830;
    border: 1px solid #394650;
    border-left: 4px solid #6baba3;
    border-radius: 14px;
    padding: 22px;
    margin-top: 18px;
}

</style>
""", unsafe_allow_html=True)



# ---------------- LOAD MODEL / DATA ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🩺 Diabetes Risk AI")
    st.caption("Machine Learning • Health Awareness")
    st.divider()

    st.markdown("### About this project")
    st.write(
        "An educational machine-learning application "
        "that predicts a risk category from entered dataset features."
    )

    st.markdown("### Project workflow")
    st.write("1. Enter the input details")
    st.write("2. Submit the prediction form")
    st.write("3. Review the model's predicted category")

    st.divider()
    st.caption("Built with Python, Scikit-learn & Streamlit")


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>Understand Your Health Data</h1>
    <p>
        A machine-learning demonstration for diabetes risk awareness.
        Enter the details below to generate a dataset-based prediction.
    </p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational demonstration only — this tool is not a medical "
    "diagnosis and has not been established as a clinically validated assessment."
)

# Check files before showing the form
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please run train_model.py first "
        "and confirm model/diabetes_risk_model.pkl exists."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

dataset = load_dataset()


# ---------------- CATEGORY OPTIONS ----------------
def get_options(column, fallback):
    if dataset is not None and column in dataset.columns:
        values = dataset[column].dropna().astype(str).unique().tolist()
        if values:
            return sorted(values)
    return fallback


gender_options = get_options(
    "gender", ["Female", "Male", "Other"]
)
family_options = get_options(
    "family_history_diabetes", ["No", "Yes"]
)
activity_options = get_options(
    "physical_activity_level", ["Low", "Moderate", "High"]
)
diet_options = get_options(
    "diet_type", ["Healthy", "Mixed", "Unhealthy"]
)
smoking_options = get_options(
    "smoking_status", ["Never", "Former", "Current", "Unknown"]
)
alcohol_options = get_options(
    "alcohol_consumption", ["None", "Low", "Moderate", "High", "Unknown"]
)
income_options = get_options(
    "income_bracket",
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High", "Unknown"]
)


# ---------------- INPUT FORM ----------------
st.markdown("## Patient Data")
st.caption("Complete the fields and submit to generate a prediction.")

with st.form("prediction_form"):
    st.markdown("### 👤 Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=30
        )
    with col2:
        gender = st.selectbox("Gender", gender_options)
    with col3:
        city_default = "Delhi"
        if dataset is not None and "city" in dataset.columns:
            cities = dataset["city"].dropna().astype(str)
            if not cities.empty:
                city_default = cities.mode().iloc[0]
        city = st.text_input("City", value=city_default)

    st.divider()
    st.markdown("### 🥗 Lifestyle & Family History")

    col1, col2 = st.columns(2)

    with col1:
        family_history_diabetes = st.selectbox(
            "Family history of diabetes", family_options
        )
        physical_activity_level = st.selectbox(
            "Physical activity level", activity_options
        )
        diet_type = st.selectbox("Diet type", diet_options)
        smoking_status = st.selectbox(
            "Smoking status", smoking_options
        )

    with col2:
        alcohol_consumption = st.selectbox(
            "Alcohol consumption", alcohol_options
        )
        hours_sleep_per_night = st.number_input(
            "Hours of sleep per night",
            min_value=0.0, max_value=24.0, value=7.0, step=0.5
        )
        stress_level = st.number_input(
            "Stress level (dataset scale)",
            min_value=0, max_value=10, value=5
        )
        income_bracket = st.selectbox(
            "Income bracket", income_options
        )

    st.divider()
    st.markdown("### 🧪 Health Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=100.0,
            value=25.0, step=0.1
        )
        fasting_blood_sugar = st.number_input(
            "Fasting blood sugar", min_value=0.0,
            value=100.0, step=1.0
        )
        hba1c_level = st.number_input(
            "HbA1c level", min_value=0.0,
            value=5.5, step=0.1
        )

    with col2:
        blood_pressure_systolic = st.number_input(
            "Systolic blood pressure",
            min_value=0, value=120
        )
        blood_pressure_diastolic = st.number_input(
            "Diastolic blood pressure",
            min_value=0, value=80
        )

    with col3:
        waist_circumference_cm = st.number_input(
            "Waist circumference (cm)",
            min_value=0.0, value=85.0, step=0.5
        )

    submitted = st.form_submit_button(
        "✨  Predict Risk Category"
    )


# ---------------- PREDICTION ----------------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "city": city,
        "bmi": bmi,
        "family_history_diabetes": family_history_diabetes,
        "physical_activity_level": physical_activity_level,
        "diet_type": diet_type,
        "smoking_status": smoking_status,
        "alcohol_consumption": alcohol_consumption,
        "hours_sleep_per_night": hours_sleep_per_night,
        "stress_level": stress_level,
        "fasting_blood_sugar": fasting_blood_sugar,
        "hba1c_level": hba1c_level,
        "blood_pressure_systolic": blood_pressure_systolic,
        "blood_pressure_diastolic": blood_pressure_diastolic,
        "waist_circumference_cm": waist_circumference_cm,
        "income_bracket": income_bracket,
    }])

    try:
        prediction = model.predict(input_data)[0]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.markdown("---")
    st.markdown("## Prediction Result")

    label = str(prediction).strip().lower()

    if label == "high":
        st.error(f"### Predicted category: {prediction}")
    elif label == "moderate":
        st.warning(f"### Predicted category: {prediction}")
    else:
        st.success(f"### Predicted category: {prediction}")

    st.markdown(
        '<div class="result-card">'
        '<strong>Prediction generated successfully</strong><br>'
        '<span class="small-note">The result is the model\'s predicted '
        'dataset label. It should not be used to make healthcare decisions.</span>'
        '</div>',
        unsafe_allow_html=True
    )

    with st.expander("Review the values submitted"):
        st.dataframe(
            input_data.T.rename(columns={0: "Entered Value"}),
            use_container_width=True
        )

st.markdown("---")
st.markdown(
    '<p class="small-note">Note: This project is intended for learning '
    'and demonstration. For personal health concerns, consult a qualified '
    'healthcare professional.</p>',
    unsafe_allow_html=True
)
