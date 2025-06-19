import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')  # Adjust path if needed

# Page setup
st.set_page_config(page_title="DiaClassify: Diabetes Prediction", layout="wide")

# Apply custom CSS for background and styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(to bottom right, #e6f0ff, #f9f9f9);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stNumberInput, .stSlider {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Heading
st.markdown("<h1 style='text-align: center; color: #004080;'>🩺 DiaClassify - Diabetes Prediction</h1>", unsafe_allow_html=True)

# Layout: 2 input columns + spacer + 1 result column
input_col1, input_col2, spacer_col, result_col = st.columns([1, 1, 0.5, 2])

with input_col1:
    st.subheader("👤 Input Features")
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.slider("🩸 Glucose", 0, 200, 120)
    blood_pressure = st.slider("💓 Blood Pressure", 0, 150, 70)
    skin_thickness = st.slider("🧪 Skin Thickness", 0, 100, 20)

with input_col2:
    st.subheader(" ")
    insulin = st.slider("💉 Insulin", 0, 900, 80)
    bmi = st.slider("📏 BMI", 0.0, 70.0, 24.0)
    dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("🎂 Age", 10, 100, 30)

# Empty spacer column

with result_col:
    st.subheader("🔍 Prediction Panel")
    st.write("Click below to get the prediction based on the input values.")

    if st.button("🚀 Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])

        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.error("⚠️ **The person is likely diabetic!**")
        else:
            st.success("✅ **The person is not likely diabetic!**")

        st.markdown("---")
        st.markdown("### 🧾 Prediction Outcome")
        
        if prediction[0] == 1:
            st.markdown("<h2 style='color:red;'>Diabetic</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>Not Diabetic</h2>", unsafe_allow_html=True)

# Footer
st.markdown("---")
#st.markdown("<center><small>Made with ❤️ using Streamlit</small></center>", unsafe_allow_html=True)
