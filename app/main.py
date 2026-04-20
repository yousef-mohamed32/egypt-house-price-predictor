import pickle, time, os
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load artefacts once at startup ────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "..", "model")

with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)
with open(os.path.join(MODEL_DIR, "features.pkl"), "rb") as f:
    features = pickle.load(f)

AREAS = list(encoders["area"].classes_)
TYPES = list(encoders["property_type"].classes_)

# ── Simple in-memory counters for /metrics ────────────────────────────────────
stats = {"total_requests": 0, "successful_predictions": 0, "errors": 0}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({
        "service": "Egypt House Price Predictor",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "endpoints": {
            "POST /predict": "Predict house price",
            "GET  /health":  "Health check",
            "GET  /metrics": "Prometheus-style metrics",
            "GET  /info":    "Model info & valid values"
        }
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/info")
def info():
    return jsonify({
        "valid_areas": AREAS,
        "valid_property_types": TYPES,
        "features": features,
        "sample_request": {
            "area": "New Cairo",
            "property_type": "Apartment",
            "size_m2": 150,
            "bedrooms": 3,
            "bathrooms": 2,
            "floor": 5,
            "age_years": 3,
            "has_garage": 1,
            "has_elevator": 1,
            "has_pool": 0
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    stats["total_requests"] += 1
    start = time.time()

    try:
        body = request.get_json(force=True)

        # ── Validate required fields ──────────────────────────────────────────
        required = ["area", "property_type", "size_m2", "bedrooms",
                    "bathrooms", "floor", "age_years"]
        missing = [k for k in required if k not in body]
        if missing:
            stats["errors"] += 1
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        area  = body["area"]
        ptype = body["property_type"]
        if area not in AREAS:
            stats["errors"] += 1
            return jsonify({"error": f"Unknown area '{area}'. Valid: {AREAS}"}), 400
        if ptype not in TYPES:
            stats["errors"] += 1
            return jsonify({"error": f"Unknown type '{ptype}'. Valid: {TYPES}"}), 400

        # ── Build feature vector ──────────────────────────────────────────────
        row = [[
            encoders["area"].transform([area])[0],
            encoders["property_type"].transform([ptype])[0],
            int(body["size_m2"]),
            int(body["bedrooms"]),
            int(body["bathrooms"]),
            int(body["floor"]),
            int(body["age_years"]),
            int(body.get("has_garage", 0)),
            int(body.get("has_elevator", 0)),
            int(body.get("has_pool", 0)),
        ]]

        price = int(model.predict(row)[0])
        elapsed = round((time.time() - start) * 1000, 2)
        stats["successful_predictions"] += 1

        return jsonify({
            "price_egp":      price,
            "price_formatted": f"{price:,} EGP",
            "price_usd_approx": int(price / 48.5),
            "input_summary": {
                "area": area, "type": ptype,
                "size_m2": body["size_m2"], "bedrooms": body["bedrooms"]
            },
            "latency_ms": elapsed
        })

    except Exception as e:
        stats["errors"] += 1
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    """Prometheus text format"""
    lines = [
        "# HELP predictor_requests_total Total prediction requests",
        "# TYPE predictor_requests_total counter",
        f"predictor_requests_total {stats['total_requests']}",
        "# HELP predictor_successes_total Successful predictions",
        "# TYPE predictor_successes_total counter",
        f"predictor_successes_total {stats['successful_predictions']}",
        "# HELP predictor_errors_total Prediction errors",
        "# TYPE predictor_errors_total counter",
        f"predictor_errors_total {stats['errors']}",
    ]
    return "\n".join(lines), 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
