# 🌊 Kenya Flood Early Warning System

A Streamlit application that predicts next-month flood risk for Kenyan counties using XGBoost and provides AI-powered safety guidance via RAG.

## Features

1. **Model Overview** — Performance metrics, historical flood rates by county.
2. **Risk Predictor** — Real-time flood probability with adjustable weather inputs.
3. **AI Assistant** — RAG-powered chatbot grounded in Kenya flood safety protocols.

## Setup

### 1. Generate model artifacts
Run the export cell in `ElNinoModelling.ipynb`. This produces:
- `artifacts/xgb_model.pkl`
- `artifacts/feature_columns.json`
- `artifacts/county_rates.json`
- `artifacts/latest_features.csv`

### 2. Install dependencies
```bash
pip install -r requirements.txt