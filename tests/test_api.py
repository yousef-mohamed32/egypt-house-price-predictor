import json
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from main import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

VALID_PAYLOAD = {
    "area": "New Cairo", "property_type": "Apartment",
    "size_m2": 150, "bedrooms": 3, "bathrooms": 2,
    "floor": 5, "age_years": 3, "has_garage": 1,
    "has_elevator": 1, "has_pool": 0
}

def post_json(client, payload):
    return client.post("/predict",
                       data=json.dumps(payload),
                       content_type="application/json")

# ── /health ───────────────────────────────────────────────────────────────────
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "healthy"

def test_health_model_loaded(client):
    r = client.get("/health")
    assert r.get_json()["model_loaded"] is True

# ── /predict happy path ───────────────────────────────────────────────────────
def test_predict_returns_price(client):
    r = post_json(client, VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.get_json()
    assert "price_egp" in data
    assert data["price_egp"] > 0

def test_predict_price_is_reasonable(client):
    r = post_json(client, VALID_PAYLOAD)
    price = r.get_json()["price_egp"]
    # 150m² New Cairo apartment should be between 1M and 20M EGP
    assert 1_000_000 < price < 20_000_000

def test_predict_includes_usd(client):
    r = post_json(client, VALID_PAYLOAD)
    assert "price_usd_approx" in r.get_json()

def test_predict_includes_latency(client):
    r = post_json(client, VALID_PAYLOAD)
    assert "latency_ms" in r.get_json()

# ── /predict error cases ──────────────────────────────────────────────────────
def test_predict_missing_field_returns_400(client):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "area"}
    r = post_json(client, bad)
    assert r.status_code == 400
    assert "Missing fields" in r.get_json()["error"]

def test_predict_invalid_area_returns_400(client):
    bad = {**VALID_PAYLOAD, "area": "Mars"}
    r = post_json(client, bad)
    assert r.status_code == 400

def test_predict_invalid_type_returns_400(client):
    bad = {**VALID_PAYLOAD, "property_type": "Castle"}
    r = post_json(client, bad)
    assert r.status_code == 400

# ── /metrics ──────────────────────────────────────────────────────────────────
def test_metrics_returns_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.data.decode()
    assert "predictor_requests_total" in text
    assert "predictor_successes_total" in text

# ── /info ─────────────────────────────────────────────────────────────────────
def test_info_lists_areas(client):
    r = client.get("/info")
    data = r.get_json()
    assert "valid_areas" in data
    assert "New Cairo" in data["valid_areas"]

def test_info_lists_property_types(client):
    r = client.get("/info")
    assert "Apartment" in r.get_json()["valid_property_types"]
