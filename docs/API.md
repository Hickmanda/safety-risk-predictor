# API Documentation

## 1. Overview

The Construction Safety Risk Predictor exposes a production REST API built
with FastAPI.

The API serves the trained XGBoost surrogate model and provides endpoints
for:

- service health checks
- model metadata
- single-worker safety prediction
- lightweight runtime monitoring

Production API:

```text
https://safety-risk-predictor-api.onrender.com
```

Interactive Swagger documentation:

```text
https://safety-risk-predictor-api.onrender.com/docs
```

OpenAPI specification:

```text
https://safety-risk-predictor-api.onrender.com/openapi.json
```

---

## 2. Base URLs

### Production

```text
https://safety-risk-predictor-api.onrender.com
```

### Local development

```text
http://127.0.0.1:8000
```

Local API startup command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 3. Endpoints

The API currently exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service information |
| `GET` | `/health` | Health and model availability |
| `GET` | `/model-info` | Production model metadata |
| `POST` | `/predict` | Predict safe or unsafe behavior |
| `GET` | `/metrics` | Runtime monitoring metrics |

---

# 4. GET /

Returns basic service information.

## Request

```http
GET /
```

No request body is required.

## Example response

```json
{
  "service": "Construction Safety Risk Predictor",
  "status": "running",
  "documentation": "/docs",
  "health": "/health"
}
```

## cURL

```bash
curl https://safety-risk-predictor-api.onrender.com/
```

## PowerShell

```powershell
Invoke-RestMethod `
    -Uri "https://safety-risk-predictor-api.onrender.com/" `
    -Method Get
```

---

# 5. GET /health

Checks whether the API process is running and whether the production model
artifact was loaded successfully.

## Request

```http
GET /health
```

No request body is required.

## Successful response

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "xgboost",
  "detail": null
}
```

## Response fields

| Field | Type | Description |
|---|---|---|
| `status` | string | Overall API health |
| `model_loaded` | boolean | Whether the production model is available |
| `model_name` | string/null | Loaded model name |
| `detail` | string/null | Additional diagnostic information |

## cURL

```bash
curl https://safety-risk-predictor-api.onrender.com/health
```

## PowerShell

```powershell
Invoke-RestMethod `
    -Uri "https://safety-risk-predictor-api.onrender.com/health" `
    -Method Get
```

---

# 6. GET /model-info

Returns metadata describing the model currently used for production
inference.

## Request

```http
GET /model-info
```

## Example response

```json
{
  "model_name": "xgboost",
  "selection_metric": "f1_macro",
  "validation_score": 0.612631918777677,
  "decision_threshold": 0.64,
  "feature_count": 11,
  "features": [
    "SA",
    "SK",
    "SN",
    "BA",
    "PBC",
    "reference_point",
    "alpha",
    "beta",
    "lam",
    "intention",
    "day"
  ],
  "target": "behavior",
  "threshold_objective": "macro F1"
}
```

## Important fields

### `model_name`

Current production model:

```text
xgboost
```

### `selection_metric`

Model-selection metric:

```text
f1_macro
```

### `decision_threshold`

Production threshold applied to `P(safe)`:

```text
0.64
```

### `feature_count`

Number of model features:

```text
11
```

## cURL

```bash
curl https://safety-risk-predictor-api.onrender.com/model-info
```

## PowerShell

```powershell
Invoke-RestMethod `
    -Uri "https://safety-risk-predictor-api.onrender.com/model-info" `
    -Method Get
```

---

# 7. POST /predict

Predicts whether a worker behavior decision will be classified as safe or
unsafe.

The endpoint accepts worker-state and CPT features.

The API calculates `intention` internally as:

```text
intention = (SN + BA + PBC) / 3
```

Clients therefore do not need to submit `intention`.

---

## 7.1 Input Schema

The request contains 10 input variables.

| Field | Type | Range | Description |
|---|---|---:|---|
| `SA` | float | 0.0-1.0 | Situational Awareness |
| `SK` | float | 0.0-1.0 | Safety Knowledge |
| `SN` | float | 0.0-1.0 | Subjective Norm |
| `BA` | float | 0.0-1.0 | Behavior Attitude |
| `PBC` | float | 0.0-1.0 | Perceived Behavior Control |
| `reference_point` | float | 0.0-1.0 | CPT reference point |
| `alpha` | float | model range | CPT gain sensitivity |
| `beta` | float | model range | CPT loss sensitivity |
| `lam` | float | model range | CPT loss aversion |
| `day` | integer | 0-99 | Simulation day |

---

## 7.2 Example Request

```json
{
  "SA": 0.70,
  "SK": 0.75,
  "SN": 0.68,
  "BA": 0.72,
  "PBC": 0.70,
  "reference_point": 0.60,
  "alpha": 0.88,
  "beta": 0.88,
  "lam": 1.18,
  "day": 50
}
```

The server derives:

```text
intention = (0.68 + 0.72 + 0.70) / 3
          = 0.70
```

---

## 7.3 Example Response

A representative response is:

```json
{
  "behavior": 1,
  "label": "safe",
  "safe_probability": 0.8418137431144714,
  "unsafe_probability": 0.15818625688552856,
  "decision_threshold": 0.64,
  "latency_ms": 4.34
}
```

Exact probability and latency values may vary slightly by environment and
software version.

---

## 7.4 Response Fields

| Field | Type | Description |
|---|---|---|
| `behavior` | integer | `1` for safe, `0` for unsafe |
| `label` | string | Human-readable prediction |
| `safe_probability` | float | Model probability for safe behavior |
| `unsafe_probability` | float | Model probability for unsafe behavior |
| `decision_threshold` | float | Production `P(safe)` threshold |
| `latency_ms` | float | Server-side prediction latency |

---

## 7.5 Decision Rule

The production threshold is:

```text
0.64
```

Classification is performed as:

```python
if safe_probability >= 0.64:
    behavior = 1
    label = "safe"
else:
    behavior = 0
    label = "unsafe"
```

The threshold was selected on validation simulations using Macro F1.

The test set was not used to choose the threshold.

---

# 8. Prediction with cURL

## Linux/macOS

```bash
curl -X POST \
  "https://safety-risk-predictor-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "SA": 0.70,
    "SK": 0.75,
    "SN": 0.68,
    "BA": 0.72,
    "PBC": 0.70,
    "reference_point": 0.60,
    "alpha": 0.88,
    "beta": 0.88,
    "lam": 1.18,
    "day": 50
  }'
```

---

# 9. Prediction with PowerShell

Create the request body:

```powershell
$body = @{
    SA = 0.70
    SK = 0.75
    SN = 0.68
    BA = 0.72
    PBC = 0.70
    reference_point = 0.60
    alpha = 0.88
    beta = 0.88
    lam = 1.18
    day = 50
} | ConvertTo-Json
```

Send the request:

```powershell
Invoke-RestMethod `
    -Uri "https://safety-risk-predictor-api.onrender.com/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

---

# 10. Prediction with Python

Example using `requests`:

```python
import requests

API_URL = "https://safety-risk-predictor-api.onrender.com"

payload = {
    "SA": 0.70,
    "SK": 0.75,
    "SN": 0.68,
    "BA": 0.72,
    "PBC": 0.70,
    "reference_point": 0.60,
    "alpha": 0.88,
    "beta": 0.88,
    "lam": 1.18,
    "day": 50,
}

response = requests.post(
    f"{API_URL}/predict",
    json=payload,
    timeout=60,
)

response.raise_for_status()

prediction = response.json()

print(prediction)
```

---

# 11. Input Validation

FastAPI and Pydantic validate request data before inference.

Examples of invalid input include:

- missing required fields
- non-numeric feature values
- values outside permitted feature ranges
- invalid simulation days

Example invalid request:

```json
{
  "SA": 1.5,
  "SK": 0.75,
  "SN": 0.68,
  "BA": 0.72,
  "PBC": 0.70,
  "reference_point": 0.60,
  "alpha": 0.88,
  "beta": 0.88,
  "lam": 1.18,
  "day": 50
}
```

`SA = 1.5` is invalid because Situational Awareness must remain within its
allowed range.

---

# 12. Validation Error

Invalid input normally returns:

```text
HTTP 422 Unprocessable Entity
```

Example structure:

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": [
        "body",
        "SA"
      ],
      "msg": "Input should be less than or equal to 1",
      "input": 1.5
    }
  ]
}
```

Exact Pydantic error text may vary between dependency versions.

---

# 13. Missing Field Example

Request:

```json
{
  "SA": 0.70,
  "SK": 0.75
}
```

Because required fields are missing, the API returns:

```text
HTTP 422
```

This prevents incomplete feature vectors from reaching the model.

---

# 14. GET /metrics

Returns lightweight runtime monitoring information.

## Request

```http
GET /metrics
```

## Example response

```json
{
  "total_predictions": 2,
  "safe_predictions": 2,
  "unsafe_predictions": 0,
  "average_latency_ms": 7.54,
  "uptime_seconds": 386.54
}
```

Values change while the service is running.

---

## 14.1 Metrics Fields

| Field | Type | Description |
|---|---|---|
| `total_predictions` | integer | Number of prediction requests |
| `safe_predictions` | integer | Number classified as safe |
| `unsafe_predictions` | integer | Number classified as unsafe |
| `average_latency_ms` | float | Mean prediction latency |
| `uptime_seconds` | float | API process uptime |

---

## 14.2 cURL

```bash
curl https://safety-risk-predictor-api.onrender.com/metrics
```

## PowerShell

```powershell
Invoke-RestMethod `
    -Uri "https://safety-risk-predictor-api.onrender.com/metrics" `
    -Method Get
```

---

# 15. Monitoring Limitations

The current monitoring implementation is intentionally lightweight.

Metrics are stored in application memory.

This means they are reset when:

- the container restarts
- Render redeploys the service
- the free instance spins down
- a new process replaces the previous process

The endpoint is intended to demonstrate basic inference monitoring rather
than provide persistent observability.

A more advanced deployment could use:

- Prometheus
- Grafana
- OpenTelemetry
- external logging
- persistent metrics storage

---

# 16. HTTP Status Codes

Important status codes include:

| Status | Meaning |
|---:|---|
| `200` | Request completed successfully |
| `422` | Request body failed validation |
| `500` | Unexpected server-side failure |
| `503` | Model or service dependency unavailable |

---

# 17. Model Availability Errors

Prediction requires a valid production model artifact.

If the model cannot be loaded, the API should not silently generate a
prediction.

The health endpoint exposes model availability through:

```json
{
  "model_loaded": false
}
```

Prediction requests may return a service error until the artifact becomes
available.

This fail-fast behavior is preferable to returning misleading predictions.

---

# 18. Feature Order

The model was trained with the following exact feature order:

```text
SA
SK
SN
BA
PBC
reference_point
alpha
beta
lam
intention
day
```

The API reconstructs the inference DataFrame in this order.

This is important because tree-based model artifacts expect the same feature
schema used during training.

---

# 19. Why run_id, seed, and worker_id Are Not Accepted

The generated dataset also contains:

```text
run_id
seed
worker_id
```

These fields are intentionally excluded from the production prediction API.

They are simulation identifiers, not meaningful predictive safety
characteristics.

Allowing them into the model could introduce simulation-specific leakage.

---

# 20. Why behavior Is Not Accepted

`behavior` is the target variable.

It is what the model predicts.

Therefore it must never be part of the inference input.

---

# 21. Why Intention Is Not Required from the Client

The training dataset contains:

```text
intention
```

but the public API calculates it automatically.

The relationship is:

```text
intention = (SN + BA + PBC) / 3
```

This provides two advantages:

1. the public request format is simpler
2. inconsistent combinations cannot be submitted

For example, a client cannot submit:

```text
SN = 0.80
BA = 0.80
PBC = 0.80
intention = 0.10
```

because intention is derived server-side.

---

# 22. API Latency

Warm model inference is designed to remain well below:

```text
100 ms
```

Local and deployed tests typically show inference times of only a few
milliseconds once the API process is active.

The returned:

```text
latency_ms
```

field measures inference/API processing inside the service.

It does not represent complete browser-to-server network latency.

---

# 23. Render Cold Starts

The backend uses the Render free tier.

Free instances may spin down after inactivity.

The first request after a period of inactivity may therefore take tens of
seconds.

This delay can include:

```text
Render container startup
+
Python process startup
+
model loading
+
request processing
```

Subsequent requests are normally much faster.

A cold start should not be interpreted as slow XGBoost inference.

---

# 24. Swagger UI

FastAPI automatically generates interactive documentation.

Production Swagger UI:

```text
https://safety-risk-predictor-api.onrender.com/docs
```

The interface allows users to:

- inspect schemas
- inspect endpoint descriptions
- enter request values
- execute live production requests
- inspect response bodies
- inspect validation errors

This also acts as a convenient public demonstration of the backend.

---

# 25. OpenAPI

The application exposes a machine-readable OpenAPI schema:

```text
https://safety-risk-predictor-api.onrender.com/openapi.json
```

This schema can be used by:

- API clients
- testing tools
- documentation generators
- SDK generators
- external frontend applications

---

# 26. Streamlit Integration

The Streamlit frontend communicates with this API over HTTP.

Production flow:

```text
Browser
   |
   v
Streamlit Community Cloud
   |
   v
FastAPI on Render
   |
   v
XGBoost production model
```

The Streamlit application does not load `best_model.pkl`.

This keeps production inference centralized in the API.

---

# 27. Batch Predictions

The frontend supports CSV batch predictions.

Batch processing currently works by sending multiple prediction requests
through the production API.

Each row must contain:

```text
SA
SK
SN
BA
PBC
reference_point
alpha
beta
lam
day
```

`intention` is derived automatically for every row.

---

# 28. Example Batch Input

Example CSV:

```csv
SA,SK,SN,BA,PBC,reference_point,alpha,beta,lam,day
0.70,0.75,0.68,0.72,0.70,0.60,0.88,0.88,1.18,50
0.40,0.45,0.55,0.58,0.52,0.60,0.88,0.88,1.18,20
0.85,0.90,0.80,0.82,0.78,0.65,0.95,0.95,1.20,75
```

A reproducible example is stored in:

```text
data/sample_batch.csv
```

---

# 29. Example Batch Output

The frontend adds fields such as:

```text
prediction
safe_probability
unsafe_probability
decision_threshold
```

A resulting table may look conceptually like:

| Row | Prediction | P(safe) | P(unsafe) | Threshold |
|---:|---|---:|---:|---:|
| 1 | safe | 0.8418 | 0.1582 | 0.64 |
| 2 | unsafe | 0.0019 | 0.9981 | 0.64 |
| 3 | safe | 0.9844 | 0.0156 | 0.64 |

---

# 30. Testing

API behavior is tested using `pytest` and FastAPI's testing utilities.

Tests cover areas such as:

- root endpoint
- health endpoint
- model metadata
- valid prediction requests
- invalid prediction requests
- feature validation
- threshold behavior
- response schema
- metrics endpoint

Run tests locally with:

```bash
python -m pytest -v
```

---

# 31. Docker Testing

The production Docker image can be built with:

```bash
docker build -t safety-risk-predictor:local .
```

Start the container:

```bash
docker run --rm -p 8000:8000 safety-risk-predictor:local
```

Then check:

```text
http://127.0.0.1:8000/health
```

and:

```text
http://127.0.0.1:8000/docs
```

---

# 32. CI Smoke Test

GitHub Actions performs an API smoke test after building the Docker image.

The CI process verifies:

```text
Container starts
       |
       v
/health returns successfully
       |
       v
/model-info is available
       |
       v
/predict accepts a valid request
```

This verifies the actual deployable artifact rather than only testing Python
modules independently.

---

# 33. Security Scope

This project is an educational and research-oriented demonstration.

The API currently does not implement:

- authentication
- user accounts
- API keys
- rate limiting
- persistent request logs
- private model endpoints

The service accepts only numerical simulation features and does not require
personal information.

For a real production safety system, additional controls would be required.

---

# 34. Intended Use

The API is intended to demonstrate:

- machine-learning inference
- model serving
- input validation
- REST API design
- containerization
- monitoring
- CI/CD
- cloud deployment

It is not intended to make real occupational safety decisions.

---

# 35. Important Limitation

The model was trained entirely on synthetic data generated by an
agent-based simulation.

Therefore:

```text
prediction != validated real-world construction safety assessment
```

The output should be interpreted as:

> A prediction of the behavior generated by the source simulation under a
> similar state.

It should not be interpreted as a certified assessment of a real worker.

---

# 36. Production Links

## API

```text
https://safety-risk-predictor-api.onrender.com
```

## Swagger

```text
https://safety-risk-predictor-api.onrender.com/docs
```

## Health

```text
https://safety-risk-predictor-api.onrender.com/health
```

## Model metadata

```text
https://safety-risk-predictor-api.onrender.com/model-info
```

## Runtime metrics

```text
https://safety-risk-predictor-api.onrender.com/metrics
```

## Streamlit UI

```text
https://safety-risk-predictor.streamlit.app
```

## GitHub

```text
https://github.com/Hickmanda/safety-risk-predictor
```

---

# 37. Minimal API Workflow

A complete client workflow can be summarized as:

```text
GET /health
    |
    v
Confirm model_loaded = true
    |
    v
POST /predict
    |
    v
Receive probabilities
    |
    v
Interpret decision using threshold
```

For debugging or inspection:

```text
GET /model-info
GET /metrics
```

---

# 38. Summary

The REST API provides a lightweight production interface around the
construction safety ML surrogate.

It combines:

```text
FastAPI
+
Pydantic validation
+
XGBoost inference
+
model metadata
+
threshold-aware classification
+
runtime monitoring
+
Docker deployment
+
automated CI testing
```

The same API serves both the public Streamlit application and direct
external clients.
