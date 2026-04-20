# Egypt House Price Predictor 🏠

An end-to-end MLOps project — a trained Machine Learning model served via a REST API, fully containerized with Docker, monitored with Prometheus & Grafana, and deployed through a CI/CD pipeline on GitHub Actions.

![CI/CD](https://github.com/yousef-mohamed32/egypt-house-price-predictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

---

## 🧠 What it does

Predicts apartment/villa prices (in EGP) across 10 Egyptian cities based on:
- Area (Cairo, New Cairo, Maadi, Zamalek, etc.)
- Property type (Apartment, Villa, Duplex, Studio, Penthouse)
- Size, bedrooms, bathrooms, floor, age
- Amenities (garage, elevator, pool)

**Model:** Random Forest Regressor | **R² score:** 0.856

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                 GitHub Actions                  │
│   push → test (pytest) → build → smoke test     │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼────────────────────────┐
          │         Docker Compose              │
          │                                     │
          │  ┌───────────────────────────────┐  │
          │  │   price-predictor  :5000      │  │
          │  │   POST /predict               │  │
          │  │   GET  /health                │  │
          │  │   GET  /metrics  ─────────────┼──┼──► Prometheus :9090
          │  └───────────────────────────────┘  │              │
          │                                     │              ▼
          │                                     │        Grafana :3000
          └─────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/yousef-mohamed32/egypt-house-price-predictor.git
cd egypt-house-price-predictor

docker compose up --build
```

| Service    | URL                      |
|------------|--------------------------|
| API        | http://localhost:5000     |
| Prometheus | http://localhost:9090     |
| Grafana    | http://localhost:3000     |

---

## 📡 API Usage

### `GET /health`
```json
{ "status": "healthy", "model_loaded": true }
```

### `GET /info`
Returns all valid areas, property types, and a sample request body.

### `POST /predict`

**Request:**
```json
{
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
```

**Response:**
```json
{
  "price_egp": 4600266,
  "price_formatted": "4,600,266 EGP",
  "price_usd_approx": 94850,
  "latency_ms": 11.2,
  "input_summary": {
    "area": "New Cairo",
    "type": "Apartment",
    "size_m2": 150,
    "bedrooms": 3
  }
}
```

---

## 📁 Project Structure

```
egypt-house-price-predictor/
├── app/
│   └── main.py                  # Flask API (predict, health, metrics)
├── model/
│   ├── train.py                 # Model training script
│   ├── model.pkl                # Trained RandomForest model
│   └── encoders.pkl             # Label encoders
├── tests/
│   └── test_api.py              # 12 pytest tests
├── monitoring/
│   ├── prometheus.yml           # Scrape config
│   └── grafana/provisioning/    # Auto-provision Prometheus datasource
├── .github/workflows/
│   └── ci.yml                   # test → build → smoke test → lint
├── Dockerfile                   # Multi-stage: train + serve
├── docker-compose.yml           # App + Prometheus + Grafana
└── requirements.txt
```

---

## 🔁 CI/CD Pipeline

Every push to `main` runs 3 parallel jobs:

| Job | Steps |
|-----|-------|
| **Test** | Install deps → train model → run 12 pytest tests |
| **Build** | Build Docker image → start container → health check → smoke test prediction |
| **Lint** | Hadolint static analysis on Dockerfile |

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| ML Model | scikit-learn (Random Forest) |
| API | Python + Flask + Gunicorn |
| Containerization | Docker (multi-stage build) |
| Orchestration | Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Testing | pytest (12 tests) |

---

## 📬 Author

**Yousef Mohamed** — [GitHub](https://github.com/yousef-mohamed32) · [LinkedIn](https://linkedin.com/in/yousef-mohamed)
