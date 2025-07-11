import streamlit as st
import numpy as np
import pickle

# Load model dan preprocessor
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le_neigh = pickle.load(open("le_neigh.pkl", "rb"))
le_room = pickle.load(open("le_room.pkl", "rb"))

st.title("🏠 Prediksi Harga Sewa AirBnB")

# Input pengguna
neigh_input = st.selectbox("Lokasi (Neighbourhood)", le_neigh.classes_)
room_input = st.selectbox("Tipe Kamar", le_room.classes_)
guests = st.slider("Jumlah Tamu", 1, 10, 2)
nights = st.slider("Lama Inap (malam)", 1, 30, 2)

# Encode input
neigh_encoded = le_neigh.transform([neigh_input])[0]
room_encoded = le_room.transform([room_input])[0]

# Prediksi
input_data = np.array([[neigh_encoded, room_encoded, guests, nights]])
input_scaled = scaler.transform(input_data)
predicted_price = model.predict(input_scaled)[0]

st.subheader(f"💰 Estimasi Harga: ${predicted_price:.2f} per malam")
    
