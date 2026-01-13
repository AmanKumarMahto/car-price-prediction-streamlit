import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ----------------------------
# Load dataset
# ----------------------------
cars_data = pd.read_csv("Cardetails.csv")

# ----------------------------
# Basic cleaning
# ----------------------------
cars_data = cars_data.dropna()  # simple clean to avoid NaN issues

# Brand extract (same as app.py)
cars_data["name"] = cars_data["name"].apply(lambda x: str(x).split(" ")[0])

# Strip text
for col in ["fuel", "seller_type", "transmission", "owner"]:
    cars_data[col] = cars_data[col].astype(str).str.strip()

# ----------------------------
# Mappings (same as app.py)
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

# ✅ Brand map auto generate
brand_list = cars_data["name"].unique()
brand_map = {brand: i + 1 for i, brand in enumerate(brand_list)}

# ----------------------------
# Apply encoding
# ----------------------------
cars_data["name"] = cars_data["name"].map(brand_map)
cars_data["owner"] = cars_data["owner"].map(owner_map)
cars_data["fuel"] = cars_data["fuel"].map(fuel_map)
cars_data["seller_type"] = cars_data["seller_type"].map(seller_map)
cars_data["transmission"] = cars_data["transmission"].map(trans_map)

# Drop rows where mapping failed (safety)
cars_data = cars_data.dropna()

# ----------------------------
# Features + Target
# ----------------------------
X = cars_data[["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]]
y = cars_data["selling_price"]

# ----------------------------
# Train/Test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Save model + mappings together ✅
# ----------------------------
payload = {
    "model": model,
    "brand_map": brand_map,
    "owner_map": owner_map,
    "fuel_map": fuel_map,
    "seller_map": seller_map,
    "trans_map": trans_map
}

pk.dump(payload, open("model.pkl", "wb"))

print("✅ model.pkl created successfully with mappings!")
