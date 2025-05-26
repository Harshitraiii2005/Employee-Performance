import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/traditional/stackingregressor.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    /* Full screen black/purple gradient background */
    body, html, [class*="css"] {
        margin: 0; padding: 0; height: 100%;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #0a0028, #36005a, #000000);
        color: #eee;
        overflow-x: hidden;
    }

    /* Animated subtle purple particle gif background behind container */
    .bg-gif {
        position: fixed;
        top: 0; left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
        opacity: 0.15;
        pointer-events: none;
        background: url('https://i.gifer.com/embedded/download/3o7qDP8L8M5a6xwEj2.gif') no-repeat center center / cover;
        filter: brightness(0.7);
        animation: bgFade 12s ease-in-out infinite alternate;
    }
    @keyframes bgFade {
        0% {opacity: 0.1;}
        50% {opacity: 0.2;}
        100% {opacity: 0.1;}
    }

    /* 3D floating container with higher z-index */
    .main-container {
        max-width: 850px;
        margin: 4rem auto 5rem auto;
        background: linear-gradient(145deg, #292929, #1e1e1e);
        border-radius: 22px;
        box-shadow:
            6px 6px 15px #121212,
            -6px -6px 20px #3a3a3a;
        padding: 2.5rem 3rem 3rem 3rem;
        transform-style: preserve-3d;
        animation: float3d 6s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }

    @keyframes float3d {
        0%, 100% {
            transform: translateZ(0) translateY(0);
            box-shadow:
                6px 6px 15px #121212,
                -6px -6px 20px #3a3a3a;
        }
        50% {
            transform: translateZ(15px) translateY(-10px);
            box-shadow:
                10px 10px 20px #0a0a0a,
                -10px -10px 30px #444444;
        }
    }

    /* Title with subtle 3D gradient */
    .title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #8854d0, #a259ff, #c299ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        letter-spacing: 0.06em;
        user-select: none;
        text-shadow: 0 0 5px #a259ff88;
    }

    /* Smaller 3D style button */
    .stButton>button {
        max-width: 240px;
        margin: 1rem auto 0 auto;
        display: block;
        background: linear-gradient(145deg, #7a3eda, #a259ff);
        color: #fff;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.45rem 0;
        border-radius: 18px;
        border: none;
        cursor: pointer;
        box-shadow:
            4px 4px 10px #5d2f9c,
            -4px -4px 10px #c199ff;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        user-select: none;
        text-align: center;
    }

    .stButton>button:hover {
        box-shadow:
            6px 6px 18px #5d2f9c,
            -6px -6px 20px #d9baff;
        transform: translateY(-3px);
        background: linear-gradient(145deg, #b687ff, #d8b3ff);
        color: #fafafa;
    }

    .stButton>button:active {
        transform: translateY(-1px);
        box-shadow:
            3px 3px 7px #5d2f9c,
            -3px -3px 8px #c199ff;
    }

    /* Numeric inputs and selects with 3D "neumorphism"-style */
    input[type=number], select {
        width: 100%;
        background: #2a2a2a;
        border-radius: 14px;
        padding: 12px 16px;
        font-size: 1rem;
        color: #eee;
        font-family: 'Poppins', sans-serif;
        border: none;
        outline: none;
        box-shadow:
            inset 4px 4px 7px #212121,
            inset -4px -4px 7px #393939;
        transition: box-shadow 0.3s ease;
        user-select: text;
    }

    input[type=number]:focus, select:focus {
        box-shadow:
            inset 6px 6px 10px #191919,
            inset -6px -6px 10px #464646;
        color: #fff;
    }

    /* Streamlit expander style overrides */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        color: #b89aff !important;
        user-select: none !important;
        text-shadow: 0 0 5px #a259ff88 !important;
        border-left: 4px solid #8c53e2 !important;
        padding-left: 0.7rem !important;
        margin-bottom: 1.5rem !important;
    }
    .streamlit-expanderContent {
        background: #232323 !important;
        border-radius: 16px !important;
        padding: 1rem 1.5rem !important;
        box-shadow:
            6px 6px 15px #121212,
            -6px -6px 20px #3a3a3a !important;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-container {
            margin: 2rem 1rem 3rem 1rem;
            padding: 2rem 1.5rem 2.5rem 1.5rem;
        }
    }

    /* Hide Streamlit footer and header */
    footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Background GIF overlay
st.markdown('<div class="bg-gif"></div>', unsafe_allow_html=True)

# Main container with 3D floating effect
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">🎯 Employee Satisfaction Predictor</h1>', unsafe_allow_html=True)

# Use Streamlit expanders for interactive headings
with st.expander("Input Your Work Metrics 📊", expanded=True):
    numeric_features = [
        "Age", "Years_At_Company", "Performance_Score", "Monthly_Salary",
        "Work_Hours_Per_Week", "Projects_Handled", "Overtime_Hours",
        "Sick_Days", "Remote_Work_Frequency", "Team_Size",
        "Training_Hours", "Promotions"
    ]

    default_values = {
        "Age": 30,
        "Years_At_Company": 3,
        "Performance_Score": 3,
        "Monthly_Salary": 5000,
        "Work_Hours_Per_Week": 40,
        "Projects_Handled": 2,
        "Overtime_Hours": 5,
        "Sick_Days": 2,
        "Remote_Work_Frequency": 1,
        "Team_Size": 5,
        "Training_Hours": 10,
        "Promotions": 0,
    }

    cols = st.columns(3)
    inputs = {}

    for i, feature in enumerate(numeric_features):
        col = cols[i % 3]
        max_val = {
            "Monthly_Salary": 20000,
            "Age": 100,
            "Years_At_Company": 50,
            "Performance_Score": 5,
            "Work_Hours_Per_Week": 168,
            "Projects_Handled": 50,
            "Overtime_Hours": 80,
            "Sick_Days": 365,
            "Remote_Work_Frequency": 7,
            "Team_Size": 100,
            "Training_Hours": 500,
            "Promotions": 10,
        }.get(feature, 1000)

        inputs[feature] = col.number_input(
            feature.replace('_', ' '),
            min_value=0,
            max_value=max_val,
            value=default_values.get(feature, 0),
            step=1,
            key=feature
        )

with st.expander("Select Role Attributes 🧩", expanded=True):
    departments = [
        "Customer Support","Engineering","Finance","HR","IT","Legal",
        "Marketing","Operations","Sales"
    ]

    job_titles = [
        "Analyst","Consultant","Developer","Engineer","Manager",
        "Specialist","Technician"
    ]

    education_levels = [
        "Bachelor","High School","Master","PhD"
    ]

    dept = st.selectbox("Department", departments, index=departments.index("Engineering"))
    job = st.selectbox("Job Title", job_titles, index=job_titles.index("Developer"))
    edu = st.selectbox("Education Level", education_levels, index=education_levels.index("Bachelor"))

# Prepare input for model
def prepare_input():
    data = {}
    for f in numeric_features:
        data[f] = inputs[f]

    for d in departments:
        data[f"Department_{d}"] = 1 if d == dept else 0
    for j in job_titles:
        data[f"Job_Title_{j}"] = 1 if j == job else 0
    for e in education_levels:
        data[f"Education_Level_{e}"] = 1 if e == edu else 0

    return pd.DataFrame([data])

input_df = prepare_input()

# Predict button
if st.button("Predict Employee Satisfaction 🎉"):
    with st.spinner("Analyzing performance and predicting satisfaction..."):
        pred = model.predict(input_df)[0]
        st.success(f"🧠 Predicted Satisfaction Score: **{pred:.2f}**")

st.markdown('</div>', unsafe_allow_html=True)
