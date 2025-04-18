import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("🚗 Car Evaluation Classifier")
st.markdown("Enter the details of the car to predict its evaluation class.")

buying = st.selectbox("Buying Price", ["vhigh", "high", "med", "low"])
maint = st.selectbox("Maintenance Cost", ["vhigh", "high", "med", "low"])
doors = st.selectbox("Number of Doors", ["2", "3", "4", "5more"])
persons = st.selectbox("Capacity (Persons)", ["2", "4", "more"])
lug_boot = st.selectbox("Luggage Boot Size", ["small", "med", "big"])
safety = st.selectbox("Safety", ["low", "med", "high"])

if st.button("Predict"):
    input_df = pd.DataFrame([[buying, maint, doors, persons, lug_boot, safety]],
                            columns=["buying", "maint", "doors", "persons", "lug_boot", "safety"])
    input_encoded = onehot_encoder.transform(input_df)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)
    class_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"The predicted car evaluation class is: **{class_label.upper()}** 🚘")
