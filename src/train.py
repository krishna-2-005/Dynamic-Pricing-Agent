import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from preprocessing import load_and_preprocess

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'R2':   round(r2_score(y_test, y_pred), 4),
        'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    }

def train():
    df_model, feature_cols, product_map, category_map, df_raw = load_and_preprocess()
    df_model.to_csv('data/processed_data.csv', index=False)

    X = df_model[feature_cols]
    y = df_model['qty']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression':   LinearRegression(),
        'Random Forest':       RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingRegressor(n_estimators=200, random_state=42),
        'XGBoost':             XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbosity=0)
    }

    results = {}
    best_model = None
    best_r2 = -np.inf

    for name, m in models.items():
        m.fit(X_train, y_train)
        metrics = evaluate(m, X_test, y_test)
        results[name] = metrics
        print(f"{name:25s} -> R2: {metrics['R2']:.4f}  MAE: {metrics['MAE']:.4f}  RMSE: {metrics['RMSE']:.4f}")
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_model = m
            best_name = name

    print(f"\nBest model: {best_name} (R2={best_r2:.4f})")

    import os; os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/demand_model.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    joblib.dump(product_map, 'models/product_map.pkl')
    joblib.dump(category_map, 'models/category_map.pkl')

    with open('models/model_results.json', 'w') as f:
        json.dump({'results': results, 'best_model': best_name}, f, indent=2)

    print("All artifacts saved to models/")
    return best_model, results

if __name__ == '__main__':
    train()
