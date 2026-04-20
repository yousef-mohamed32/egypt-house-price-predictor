# ── Stage 1: Train the model ──────────────────────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /train

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/train.py ./model/train.py
RUN python model/train.py


# ── Stage 2: Run the API ──────────────────────────────────────────────────────
FROM python:3.11-slim AS api

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained model artefacts from stage 1
COPY --from=trainer /train/model/*.pkl ./model/

# Copy app source
COPY app/ ./app/

ENV APP_VERSION=1.0.0 \
    PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app.main:app"]
