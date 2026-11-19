# Kenya Flood Early Warning System (Myles Branch)

Welcome to the `myles` branch of the El Niño project! This folder (`myles_app`) contains a Streamlit application that predicts next-month flood risks for Kenyan counties and provides AI-powered safety advice.

## What the App Does
The application is divided into three main sections:

1. **Historical Insights:** Visualizes 20+ years of flood data. Users can see the counties most prone to flooding, the seasonality of floods (e.g., Long Rains vs. Short Rains), and the severity of past events.
2. **Risk Predictor:** An interactive tool that allows users to select a county, input current weather conditions, and instantly get a **Low, Moderate, or High** flood risk assessment with actionable safety advice.
3. **AI Assistant:** A custom chatbot powered by Llama 3 (via Groq). It acts as an agent that can:
   * Query the machine learning model in real-time (e.g., *"Will Kisumu flood next month?"*)
   * Search a RAG (Retrieval-Augmented Generation) database containing official Kenya flood safety protocols.
   * Respond in both **English and Kiswahili**.

## Under the Hood: The Machine Learning
* **The Model:** We use an **XGBoost Classifier** trained on historical weather data (Rainfall, Oceanic Niño Index, Dipole Mode Index) and flood records from 2002 to 2019.
* **Performance:** The model achieved an impressive **ROC-AUC of 0.896** and a **PR-AUC of 0.392** (which is 8x better than a random baseline for rare events).
* **The Key Feature:** The most important predictor is `flood_this_month` (antecedent conditions). If a county is already flooding, the model heavily prioritizes that data.
* **Human-Friendly Design:** We purposely hid complex oceanic indices (ONI/DMI) from the UI. The app silently pulls the latest climate data in the background, making the app accessible to everyday users.
