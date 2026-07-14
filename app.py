from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# ==========================
# Load Model and Files
# ==========================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

industry_encoder = joblib.load("industry_encoder.pkl")
ethnicity_encoder = joblib.load("ethnicity_encoder.pkl")
citizen_encoder = joblib.load("citizen_encoder.pkl")


# ==========================
# Home Page
# ==========================

@app.route("/")
def home():
    return render_template("home.html")


# ==========================
# About Page
# ==========================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================
# Prediction Page
# ==========================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # Encode categorical values
        industry = industry_encoder.transform(
            [request.form["Industry"]]
        )[0]

        ethnicity = ethnicity_encoder.transform(
            [request.form["Ethnicity"]]
        )[0]

        citizen = citizen_encoder.transform(
            [request.form["Citizen"]]
        )[0]

        # Prepare input data
        data = [[
            float(request.form["Gender"]),
            float(request.form["Age"]),
            float(request.form["Debt"]),
            float(request.form["Married"]),
            float(request.form["BankCustomer"]),
            industry,
            ethnicity,
            float(request.form["YearsEmployed"]),
            float(request.form["PriorDefault"]),
            float(request.form["Employed"]),
            float(request.form["CreditScore"]),
            float(request.form["DriversLicense"]),
            citizen,
            float(request.form["ZipCode"]),
            float(request.form["Income"])
        ]]

        # Scale the data
        data = scaler.transform(data)

        # Prediction
        prediction = model.predict(data)[0]

        # Prediction Probability
        probability = model.predict_proba(data)[0]

        if prediction == 1:
            result = "Approved"
            confidence = round(probability[1] * 100, 2)
        else:
            result = "Rejected"
            confidence = round(probability[0] * 100, 2)

        return render_template(
            "result.html",
            result=result,
            confidence=confidence
        )

    return render_template("predict.html")


# ==========================
# Run Flask
# ==========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)