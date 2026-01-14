import streamlit as st
import pandas as pd
import pickle as pk

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title(" ō͡≡o˞̶Car Price Prediction ML App")
st.caption("Made by Aman Kumar Mahto")

# ----------------------------
# Load model
# ----------------------------
loaded = pk.load(open("model.pkl","rb"))

# ✅ agar dict payload hai
if isinstance(loaded, dict):
    model = loaded["model"]
    brand_map = loaded["brand_map"]
    owner_map = loaded["owner_map"]
    fuel_map = loaded["fuel_map"]
    seller_map = loaded["seller_map"]
    trans_map = loaded["trans_map"]
else:
    # ✅ agar purana model.pkl hai (sirf model)
    model = loaded


# ----------------------------
# Load dataset for dropdown values
# ----------------------------
cars_data = pd.read_csv("Cardetails.csv")

# Brand निकालना (same as training)
cars_data["name"] = cars_data["name"].apply(lambda x: str(x).split(" ")[0])

# Clean string columns
for col in ["fuel", "seller_type", "transmission", "owner"]:
    cars_data[col] = cars_data[col].astype(str).str.strip()

# ----------------------------
# Same mappings as train_model.py
# ----------------------------
owner_map = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 5
}

fuel_map = {
    "Diesel": 1,
    "Petrol": 2,
    "LPG": 3,
    "CNG": 4,
    "Electric": 5
}

seller_map = {
    "Individual": 1,
    "Dealer": 2,
    "Trustmark Dealer": 3
}

trans_map = {
    "Manual": 1,
    "Automatic": 2
}

# ✅ Brand map auto create (same as training logic)
brand_list = cars_data["name"].unique()
brand_map = {brand: i + 1 for i, brand in enumerate(brand_list)}

# ----------------------------
# UI Inputs
# ----------------------------
brand = st.selectbox("Select Car Brand", sorted(brand_list))
year = st.slider("Car Manufactured Year", 1994, 2025, 2015)
km_driven = st.slider("No. of KMs Driven", 0, 300000, 50000)

fuel = st.selectbox("Fuel type", sorted(cars_data["fuel"].unique()))
seller_type = st.selectbox("Seller type", sorted(cars_data["seller_type"].unique()))
transmission = st.selectbox("Transmission type", sorted(cars_data["transmission"].unique()))
owner = st.selectbox("Owner type", sorted(cars_data["owner"].unique()))

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict"):
    # Encode inputs
    brand_encoded = brand_map.get(brand)
    fuel_encoded = fuel_map.get(fuel)
    seller_encoded = seller_map.get(seller_type)
    trans_encoded = trans_map.get(transmission)
    owner_encoded = owner_map.get(owner)

    # ✅ Check unsupported values (to avoid NaN crash)
    if None in [brand_encoded, fuel_encoded, seller_encoded, trans_encoded, owner_encoded]:
        st.error("⚠️ Selected value mapping not found. Please change the selection.")
        st.stop()

    input_df = pd.DataFrame([[
        brand_encoded,
        year,
        km_driven,
        fuel_encoded,
        seller_encoded,
        trans_encoded,
        owner_encoded
    ]], columns=["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"])

    # Predict
    pred = model.predict(input_df)[0]
    st.success(f"✅ Predicted Car Price: ₹ {int(pred):,}")



