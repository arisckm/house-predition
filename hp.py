import streamlit as st
import numpy as np
import pickle

# 🌟 Load the model and scaler
with open("rf_model_small.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="California Housing Price Predictor", layout="centered")

st.title("🏡 California Housing Price Predictor")
st.markdown("Predict median house value based on neighborhood features.")

# 🧾 User Input
MedInc = st.number_input("Median Income (10k USD)", min_value=0.0, step=0.1, value=3.5)
HouseAge = st.slider("House Age (years)", 1, 52, 20)
AveRooms = st.number_input("Average Number of Rooms", min_value=0.1, step=0.1, value=5.0)
AveBedrms = st.number_input("Average Number of Bedrooms", min_value=0.1, step=0.1, value=1.0)
Population = st.number_input("Population", min_value=1, step=1, value=1000)
AveOccup = st.number_input("Average Occupancy", min_value=0.1, step=0.1, value=3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -119.0)

if st.button("Predict House Price 💰"):
    # 🛠 Engineered features
    LogPopulation = np.log1p(Population)
    PeoplePerRoom = Population / AveRooms if AveRooms != 0 else 0

    # 📦 Prepare input array
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude,
                            LogPopulation, PeoplePerRoom]])

    # 📏 Scale the input
    input_scaled = scaler.transform(input_data)

    # 🔮 Predict
    prediction = model.predict(input_scaled)[0]
    prediction_in_usd = round(prediction * 100000, 2)

    st.success(f"🏠 Estimated Median House Value: **${prediction_in_usd:,}**")

