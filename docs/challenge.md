# LATAM Airlines ML & AI Engineer Challenge

## 1. Problem Overview

The goal of this challenge is to build a machine learning system to predict flight delays and deploy it as a production-ready API.

The solution includes:
- Data preprocessing
- Model training
- API development
- Containerization
- Deployment
- Load testing

---

## 2. Model Approach

Based on the exploratory notebook, the best-performing model was:

**XGBoost Classifier**

Key decisions:
- Used `scale_pos_weight` to handle class imbalance
- One-hot encoding for categorical variables:
  - OPERA
  - TIPOVUELO
  - MES
- Fixed feature space to avoid training/serving mismatch

---

## 3. Engineering Decisions

### Model
- Implemented in `DelayModel`
- Lazy training inside `predict()` for simplicity

### API
- Built using FastAPI
- Endpoints:
  - `/health`
  - `/predict`
- Custom validation:
  - OPERA must be valid airline
  - TIPOVUELO ∈ {I, N}
  - MES ∈ [1,12]

### Testing
- All provided tests passed:
  - `make model-test`
  - `make api-test`

---

## 4. Containerization

- Base image: `python:3.10-slim`
- Added `libgomp1` for XGBoost
- Configured container to use dynamic port:

```bash
uvicorn challenge.api:app --host 0.0.0.0 --port $PORT