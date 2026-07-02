import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Fertilizer Advisor", page_icon="🌾", layout="centered")

# Load Model
@st.cache_resource
def load_model():
    model = pickle.load(open('fertilizer_model.pkl', 'rb'))
    scaler = pickle.load(open('fertilizer_scaler.pkl', 'rb'))
    le_soil = pickle.load(open('soil_encoder.pkl', 'rb'))
    le_crop = pickle.load(open('crop_encoder.pkl', 'rb'))
    return model, scaler, le_soil, le_crop

model, scaler, le_soil, le_crop = load_model()

st.title("🌾 Fertilizer Recommendation System")
st.subheader("Helping Farmers Choose the Right Fertilizer")

st.markdown("---")

st.write("### Enter Your Farm Details")

col1, col2 = st.columns(2)

with col1:
    N = st.slider("Nitrogen Level (N)", 0, 200, 70)
    P = st.slider("Phosphorus Level (P)", 0, 200, 40)
    K = st.slider("Potassium Level (K)", 0, 200, 40)
    temp = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)

with col2:
    humidity = st.slider("Humidity (%)", 30.0, 100.0, 75.0)
    soil = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy"])
    crop = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato", "Tomato"])

if st.button("🌱 Get Fertilizer Recommendation", type="primary", use_container_width=True):
    soil_enc = le_soil.transform([soil])[0]
    crop_enc = le_crop.transform([crop])[0]
    
    input_data = np.array([[N, P, K, temp, humidity, soil_enc, crop_enc]])
    input_scaled = scaler.transform(input_data)
    
    pred = model.predict(input_scaled)[0]
    
    st.success(f"**Recommended Fertilizer: {pred}**")
    st.balloons()

    st.info(f"""
    **Recommendation Summary:**
    - Crop: **{crop}**
    - Soil: **{soil}**
    - Temperature: **{temp}°C**
    - Humidity: **{humidity}%**
    """)

st.markdown("---")
st.caption("Agriculture Project | Naive Bayes Algorithm")