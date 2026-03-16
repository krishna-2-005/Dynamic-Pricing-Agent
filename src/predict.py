import numpy as np
import joblib
from scipy.optimize import minimize_scalar

model        = joblib.load('models/demand_model.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

def predict_demand(feature_vector: np.ndarray) -> float:
    pred = model.predict(feature_vector.reshape(1, -1))[0]
    return max(0.0, float(pred))

def _neg_revenue(price, base_features: np.ndarray, price_idx: int):
    fv = base_features.copy()
    fv[price_idx] = price
    # update derived features that depend on price
    # price_score = price * product_score (index of product_score is 5)
    fv[feature_cols.index('price_score')]  = price * base_features[5]
    # price_volume = price * volume (index of volume is 12)
    fv[feature_cols.index('price_volume')] = price * base_features[12]
    demand = predict_demand(fv)
    return -(price * demand)

def find_optimal_price(base_features: np.ndarray, price_bounds=(1.0, 500.0)):
    price_idx = feature_cols.index('unit_price')
    result = minimize_scalar(
        _neg_revenue,
        bounds=price_bounds,
        args=(base_features, price_idx),
        method='bounded'
    )
    opt_price   = result.x
    max_revenue = -result.fun
    return float(opt_price), float(max_revenue)
