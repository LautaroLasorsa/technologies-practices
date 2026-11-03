# Python Internal Microservice

This microservice provides operational KPIs, slow machine learning forecasting, and scenario mutation APIs.

## Authentication
This service is intentionally designed as an internal microservice not meant for direct browser consumption. Every request (except `/health`) must provide the header:
```
X-Internal-Service-Key: super-secret-internal-key
```

If a browser client attempts to query it directly without this secret or across unauthenticated networks, it receives an HTTP 403 Forbidden error. The Next.js BFF layer holds this secret in server environment variables and acts as the secure intermediary.

## Setup & Running
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\pip install -e .
.venv\Scripts\uvicorn main:app --port 8117 --reload

# On Linux/macOS:
.venv/bin/pip install -e .
.venv/bin/uvicorn main:app --port 8117 --reload
```
