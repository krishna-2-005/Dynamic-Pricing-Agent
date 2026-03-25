# Dynamic Pricing Agent

A machine learning-powered pricing system that predicts product demand and recommends a revenue-maximizing price.

This project combines:
- A full preprocessing and feature engineering pipeline
- Multi-model regression training and model selection
- Optimization-based price recommendation using predicted demand
- An interactive Streamlit application for what-if pricing analysis

## Why This Project

Retail pricing is a balance between margin and demand. Pricing too high can hurt conversion; pricing too low can leave revenue on the table.

This project models demand as a function of:
- Product quality and physical attributes
- Competitor prices and scores
- Seasonality and calendar effects
- Engineered interactions between price and product characteristics

Then it solves:

max Revenue(price) = price x PredictedDemand(features, price)

using bounded numerical optimization.

## Highlights

- Trains and compares 4 regressors:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Automatically selects the best model by test R2
- Engineers 22 predictive features from raw retail data
- Generates production artifacts:
  - Trained model
  - Feature column order
  - Product/category encodings
  - Model performance summary
- Ships with a modern Streamlit UI for:
  - Optimal price recommendation
  - Revenue and demand curves
  - Dataset insights and visual analytics
  - Model performance dashboard

## Repository Structure

```text
.
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- retail_price_dataset.csv
|   `-- processed_data.csv
|-- models/
|   |-- demand_model.pkl
|   |-- feature_cols.pkl
|   |-- product_map.pkl
|   |-- category_map.pkl
|   `-- model_results.json
|-- notebooks/
|   `-- eda.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- train.py
|   `-- predict.py
`-- requirements.txt
```

## Data and Features

Raw data source: data/retail_price_dataset.csv

Target:
- qty (units sold)

Feature groups:
- Price and product attributes:
  - unit_price, product_score, product_weight_g, product_photos_qty
  - product_name_lenght, product_description_lenght
- Customer and calendar context:
  - customers, weekday, weekend, holiday, month, year
- Competitive intelligence:
  - avg_comp_price, min_comp_price, avg_comp_score
  - price_vs_avg_comp, price_vs_min_comp
- Price interaction terms:
  - price_score = unit_price * product_score
  - price_volume = unit_price * volume
- Encoded categorical IDs:
  - product_id_enc, product_category_enc

Notes:
- Missing values are dropped for modeling.
- Extreme qty outliers above the 99th percentile are removed.

## Model Performance (Current Artifacts)

From models/model_results.json:

| Model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.1276 | 9.0948 | 12.2713 |
| Random Forest | 0.2777 | 7.3618 | 11.1660 |
| Gradient Boosting | 0.1964 | 7.5134 | 11.7770 |
| XGBoost | 0.2586 | 7.4542 | 11.3123 |

Best model: Random Forest

## Quick Start

### 1) Clone and enter the project

```bash
git clone <your-repo-url>
cd dynamic-pricing-agent
```

### 2) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install plotly
```

Why extra plotly install: the Streamlit app uses Plotly charts and imports plotly.graph_objects / plotly.express.

### 4) Train or retrain the model (optional if artifacts already exist)

```bash
python src/train.py
```

This will regenerate:
- data/processed_data.csv
- models/demand_model.pkl
- models/feature_cols.pkl
- models/product_map.pkl
- models/category_map.pkl
- models/model_results.json

### 5) Launch the app

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed by Streamlit (typically http://localhost:8501).

## How Pricing Optimization Works

1. Build a feature vector from user inputs in the same column order used in training.
2. Predict demand with the trained regression model.
3. Define revenue as:
   - Revenue(price) = price x demand(price)
4. Search the best price in a user-selected range using scipy.optimize.minimize_scalar (bounded).
5. Return:
   - optimal price
   - expected demand and revenue
   - delta vs current price
   - revenue and demand curves

## Typical Workflow

1. Train once with src/train.py (or use existing model artifacts).
2. Run streamlit app.
3. Enter product, market, and time context.
4. Set a search range for candidate prices.
5. Click Find Optimal Price.
6. Compare current vs optimized revenue/profit estimates.

## Inference Module

The src/predict.py module provides reusable functions for programmatic use:
- predict_demand(feature_vector)
- find_optimal_price(base_features, price_bounds=(1.0, 500.0))

Use this module if you want to integrate the pricing engine into an API, batch job, or service.

## Troubleshooting

- ModuleNotFoundError for plotly:
  - Run: pip install plotly
- Streamlit command not found:
  - Run: pip install streamlit
  - Or use: python -m streamlit run app/streamlit_app.py
- Model files missing when opening app:
  - Run: python src/train.py
- App seems slow on first run:
  - Expected; model and dataset are loaded and cached.

## Future Improvements

- Hyperparameter tuning (Optuna/RandomizedSearchCV)
- Time-aware train/validation split for stronger temporal generalization
- Model explainability (SHAP feature contributions)
- Cost-aware optimization for true profit maximization
- Confidence intervals around predicted demand/revenue
- API deployment (FastAPI) for production integration

## License

Add your preferred license (MIT, Apache-2.0, etc.) in a LICENSE file.

## Acknowledgments

Built with scikit-learn, XGBoost, SciPy, Streamlit, and Plotly.