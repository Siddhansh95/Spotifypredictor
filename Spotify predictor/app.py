from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder="templates")

# Path to your uploaded model
MODEL_PATH = "random_forest_model-2.pickle"

# Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    # Renders the frontend form (AJAX uses /api/predict)
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Expects JSON:
    {
      "track_duration_ms": <float>,
      "album_total_tracks": <int>,
      "artist_popularity": <float>,
      "artist_followers": <int>
    }
    Returns JSON:
    { "prediction": <float> }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        # Extract features in the exact order your model expects
        features = [
            float(data["track_duration_ms"]),
            float(data["album_total_tracks"]),
            float(data["artist_popularity"]),
            float(data["artist_followers"]),
        ]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except ValueError:
        return jsonify({"error": "Feature values must be numbers"}), 400

    X = np.array([features])

    try:
        pred = model.predict(X)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    # If model outputs class or array-like, convert to float
    # If the model outputs a regression numeric value, use it directly
    if isinstance(pred, (list, tuple, np.ndarray)):
        raw = float(pred[0])
    else:
        raw = float(pred)

    # Optionally clip and round (assume popularity range 0-100)
    popularity = max(0.0, min(100.0, raw))
    popularity = round(popularity, 2)

    return jsonify({"prediction": popularity})

@app.route("/predict", methods=["POST"])
def form_predict():
    """
    This endpoint allows a classic form POST (if you prefer).
    It extracts HTML form fields and returns the template with the result.
    """
    try:
        form = request.form
        features = [
            float(form.get("track_duration_ms")),
            float(form.get("album_total_tracks")),
            float(form.get("artist_popularity")),
            float(form.get("artist_followers")),
        ]
    except Exception:
        return render_template("index.html", error="Invalid form input. Please enter numbers.")

    X = np.array([features])
    try:
        pred = model.predict(X)
    except Exception as e:
        return render_template("index.html", error=f"Model prediction failed: {e}")

    raw = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
    popularity = max(0.0, min(100.0, raw))
    popularity = round(popularity, 2)

    return render_template("index.html", prediction_text=f"Predicted Popularity: {popularity}")

if __name__ == "__main__":
    # debug True only for development
    app.run(debug=True, host="0.0.0.0", port=5000)

