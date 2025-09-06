from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained ML model
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form inputs
        brand = request.form.get("brand")
        model_name = request.form.get("model")
        year = int(request.form.get("Year"))
        kms_driven = int(request.form.get("Kms_Driven"))
        fuel_type = request.form.get("fuel_type")
        city = request.form.get("city")
        owner = request.form.get("owner")

        # Prepare DataFrame for ML model
        input_data = pd.DataFrame([{
            "brand": brand,
            "model": model_name,
            "Year": year,
            "Kms_Driven": kms_driven,
            "fuel_type": fuel_type,
            "city": city,
            "owner": owner
        }])

        # Predict price
        predicted_price = model.predict(input_data)[0]

        # Send all data + result to template
        return render_template("result.html",
                               price=round(predicted_price, 2),
                               brand=brand,
                               model=model_name,
                               year=year,
                               kms=kms_driven,
                               fuel=fuel_type,
                               city=city,
                               owner=owner)

    except Exception as e:
        return f"<h3 style='color:red;'>❌ Error: {e}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
