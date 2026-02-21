from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

model = None
scaler = None
columns = None
accuracy = None

@app.route("/", methods=["GET", "POST"])
def index():
    global model, scaler, columns, accuracy

    if request.method == "POST":
        file = request.files["file"]

        if file:
            df = pd.read_csv(file)

            if "Exited" not in df.columns:
                return "Dataset must contain 'Exited' column"

            # Drop unwanted columns
            df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

            # Encode categorical columns
            le = LabelEncoder()
            df["Geography"] = le.fit_transform(df["Geography"])
            df["Gender"] = le.fit_transform(df["Gender"])

            X = df.drop("Exited", axis=1)
            y = df["Exited"]
            columns = X.columns

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Model training
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            return render_template("index.html",
                                   columns=columns,
                                   accuracy=round(accuracy, 2))

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model, scaler, columns

    values = [float(request.form[col]) for col in columns]
    input_data = np.array(values).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    result = "Customer is likely to CHURN" if prediction[0] == 1 else "Customer is NOT likely to churn"
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
