"""
Train a house price prediction model for Egypt
and save it as a .pkl file
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ── Synthetic Egyptian real-estate data ──────────────────────────────────────
np.random.seed(42)
n = 1200

areas       = ["Cairo", "Giza", "Alexandria", "New Cairo", "6th October",
               "Heliopolis", "Maadi", "Zamalek", "Nasr City", "Shorouk"]
prop_types  = ["Apartment", "Villa", "Duplex", "Studio", "Penthouse"]

area_multiplier = {
    "Zamalek": 2.8, "Maadi": 2.2, "Heliopolis": 2.0, "New Cairo": 1.9,
    "Cairo": 1.5, "Giza": 1.4, "Alexandria": 1.3, "6th October": 1.2,
    "Nasr City": 1.35, "Shorouk": 1.1
}
type_multiplier = {
    "Penthouse": 2.5, "Villa": 2.2, "Duplex": 1.7,
    "Apartment": 1.0, "Studio": 0.65
}

data = []
for _ in range(n):
    area      = np.random.choice(areas)
    ptype     = np.random.choice(prop_types)
    size_m2   = int(np.random.normal(150, 60))
    size_m2   = max(40, min(size_m2, 600))
    bedrooms  = np.random.randint(1, 6)
    bathrooms = np.random.randint(1, 4)
    floor     = np.random.randint(0, 20)
    age_years = np.random.randint(0, 30)
    has_garage   = np.random.choice([0, 1])
    has_elevator = np.random.choice([0, 1])
    has_pool     = np.random.choice([0, 1], p=[0.85, 0.15])

    base_price = (
        size_m2 * 18_000
        * area_multiplier[area]
        * type_multiplier[ptype]
        * (1 + bedrooms * 0.05)
        * (1 - age_years * 0.008)
        * (1 + has_garage * 0.06)
        * (1 + has_elevator * 0.04)
        * (1 + has_pool * 0.12)
    )
    price = int(base_price * np.random.uniform(0.88, 1.12))

    data.append([area, ptype, size_m2, bedrooms, bathrooms, floor,
                 age_years, has_garage, has_elevator, has_pool, price])

df = pd.DataFrame(data, columns=[
    "area", "property_type", "size_m2", "bedrooms", "bathrooms",
    "floor", "age_years", "has_garage", "has_elevator", "has_pool", "price_egp"
])

# ── Encode categoricals ───────────────────────────────────────────────────────
le_area = LabelEncoder()
le_type = LabelEncoder()
df["area_enc"]  = le_area.fit_transform(df["area"])
df["ptype_enc"] = le_type.fit_transform(df["property_type"])

features = ["area_enc", "ptype_enc", "size_m2", "bedrooms", "bathrooms",
            "floor", "age_years", "has_garage", "has_elevator", "has_pool"]
X = df[features]
y = df["price_egp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train ─────────────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
print(f"MAE : {mae:,.0f} EGP")
print(f"R²  : {r2:.4f}")

# ── Save artefacts ────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/encoders.pkl", "wb") as f:
    pickle.dump({"area": le_area, "property_type": le_type}, f)
with open("model/features.pkl", "wb") as f:
    pickle.dump(features, f)

print("Model saved to model/model.pkl")
