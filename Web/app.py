from flask import Flask, request, render_template
import numpy as np
import joblib
from keras.models import load_model

app = Flask(__name__)

# Load saved things
model = load_model("aapl_lstm.h5")
scaler = joblib.load("scaler.pkl")
with open("window_size.txt") as f:
    WINDOW_SIZE = int(f.read())

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Read textarea input
            raw_text = request.form["prices"]

            # convert to list of floats
            prices = [float(x.strip()) for x in raw_text.split(",") if x.strip()]

            if len(prices) != WINDOW_SIZE:
                error = f"Please enter exactly {WINDOW_SIZE} prices. You entered {len(prices)}."
            else:
                prices_np = np.array(prices).reshape(-1, 1)

                # scale
                scaled = scaler.transform(prices_np)

                # reshape for LSTM (1, window, 1)
                X = scaled.reshape(1, WINDOW_SIZE, 1)

                # predict
                pred_scaled = model.predict(X)[0][0]

                # inverse scale
                pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]

                prediction = pred_real

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error, window_size=WINDOW_SIZE)


if __name__ == "__main__":
    app.run(debug=True)
