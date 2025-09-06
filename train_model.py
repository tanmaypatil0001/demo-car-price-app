import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle

print("🔄 Loading dataset...")
df = pd.read_excel("car_database.xlsx")
print("➡️ Original columns:", df.columns.tolist())

# Rename columns to standard
df = df.rename(columns={
    "company": "brand",
    "name": "model",
    "year": "Year",
    "kms_driven": "Kms_Driven",
    "fuel_type": "fuel_type",
    "Price": "Price"
})

# Add dummy columns for city & owner (since Excel doesn't have them)
df["city"] = "Unknown"
df["owner"] = "1"

# --- Clean values ---
def clean_price(x):
    """Convert price, skip 'Ask For Price'"""
    s = str(x).strip().replace(",", "")
    if "ask" in s.lower():
        return None
    try:
        return float(s)
    except:
        return None

def clean_kms(x):
    """Convert '45,000 kms' -> 45000"""
    s = re.sub(r"[^0-9]", "", str(x))
    return int(s) if s else None

df["Price"] = df["Price"].apply(clean_price)
df["Kms_Driven"] = df["Kms_Driven"].apply(clean_kms)

# Drop rows with missing values
before = len(df)
df = df.dropna(subset=["Price", "Year", "Kms_Driven"])
after = len(df)
print(f"✅ Dropped {before - after} rows with missing/invalid Price, Year, or Kms")

# Define X and y
X = df[["brand", "model", "Year", "Kms_Driven", "fuel_type", "city", "owner"]]
y = df["Price"]

# Preprocessing
categorical = ["brand", "model", "fuel_type", "city", "owner"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough"
)

# Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🔧 Training model...")
model.fit(X_train, y_train)
print("📊 Train R²:", model.score(X_train, y_train))
print("📊 Test R²:", model.score(X_test, y_test))

# Save
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved as car_price_model.pkl")
