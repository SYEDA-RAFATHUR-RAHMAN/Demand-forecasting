"""
models.py
All forecasting models used in the project:
  1. Moving Average
  2. Linear Regression  (scikit-learn)
  3. Random Forest      (scikit-learn)
  4. Prophet            (Facebook/Meta time-series model)

Also includes inventory optimization functions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════
# 1. MOVING AVERAGE
# ════════════════════════════════════════════════════
def moving_average_forecast(series, window=3, periods=3):
    """Average of last `window` months, rolled forward."""
    vals = list(series)
    result = []
    for _ in range(periods):
        avg = np.mean(vals[-window:])
        result.append(round(avg))
        vals.append(avg)
    return result


# ════════════════════════════════════════════════════
# 2. LINEAR REGRESSION
# ════════════════════════════════════════════════════
def linear_regression_forecast(series, periods=3):
    """Fits trend line through historical data."""
    n = len(series)
    X = np.arange(n).reshape(-1, 1)
    y = np.array(series)
    model = LinearRegression().fit(X, y)
    future = np.arange(n, n + periods).reshape(-1, 1)
    return [max(0, round(p)) for p in model.predict(future)]


# ════════════════════════════════════════════════════
# 3. RANDOM FOREST
# ════════════════════════════════════════════════════
def _make_features(series):
    vals = list(series)
    X, y = [], []
    for i in range(3, len(vals)):
        X.append([vals[i-1], vals[i-2], vals[i-3],
                  np.mean(vals[i-3:i]), i, (i % 12) + 1])
        y.append(vals[i])
    return np.array(X), np.array(y)

def random_forest_forecast(series, periods=3):
    """Ensemble of 100 decision trees using lag + seasonal features."""
    vals = list(series)
    X, y = _make_features(vals)
    if len(X) < 5:
        return moving_average_forecast(series, periods=periods)
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    ext = list(vals)
    preds = []
    for step in range(periods):
        n = len(ext)
        feat = [ext[-1], ext[-2], ext[-3], np.mean(ext[-3:]),
                n + step, ((n + step) % 12) + 1]
        p = max(0, round(model.predict([feat])[0]))
        preds.append(p)
        ext.append(p)
    return preds


# ════════════════════════════════════════════════════
# 4. PROPHET  ← Main model per synopsis
# ════════════════════════════════════════════════════
def prophet_forecast(dates, series, periods=3):
    """
    Facebook Prophet — handles seasonality + trend automatically.
    This is the PRIMARY model as specified in the project synopsis.
    """
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": series})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.80,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)

    # Extract only future rows
    future_fc = forecast.tail(periods)
    predicted  = [max(0, round(v)) for v in future_fc["yhat"].tolist()]
    lower      = [max(0, round(v)) for v in future_fc["yhat_lower"].tolist()]
    upper      = [max(0, round(v)) for v in future_fc["yhat_upper"].tolist()]
    fut_dates  = future_fc["ds"].tolist()

    return predicted, lower, upper, fut_dates, forecast


# ════════════════════════════════════════════════════
# MODEL EVALUATION  (train/test split)
# ════════════════════════════════════════════════════
def evaluate_all_models(dates, series, test_months=3):
    """
    Splits last `test_months` as test set.
    Returns MAE and RMSE for all 4 models.
    """
    train_s = series[:-test_months]
    train_d = dates[:-test_months]
    actual  = series[-test_months:]
    results = {}

    # Moving Average
    p = moving_average_forecast(train_s, periods=test_months)
    results["Moving Average"] = {
        "MAE":  round(mean_absolute_error(actual, p), 1),
        "RMSE": round(np.sqrt(mean_squared_error(actual, p)), 1),
        "preds": p
    }

    # Linear Regression
    p = linear_regression_forecast(train_s, periods=test_months)
    results["Linear Regression"] = {
        "MAE":  round(mean_absolute_error(actual, p), 1),
        "RMSE": round(np.sqrt(mean_squared_error(actual, p)), 1),
        "preds": p
    }

    # Random Forest
    p = random_forest_forecast(train_s, periods=test_months)
    results["Random Forest"] = {
        "MAE":  round(mean_absolute_error(actual, p), 1),
        "RMSE": round(np.sqrt(mean_squared_error(actual, p)), 1),
        "preds": p
    }

    # Prophet
    try:
        p, _, _, _, _ = prophet_forecast(train_d, train_s, periods=test_months)
        results["Prophet"] = {
            "MAE":  round(mean_absolute_error(actual, p), 1),
            "RMSE": round(np.sqrt(mean_squared_error(actual, p)), 1),
            "preds": p
        }
    except Exception as e:
        results["Prophet"] = {"MAE": "N/A", "RMSE": "N/A", "preds": [], "error": str(e)}

    return results, list(actual)


# ════════════════════════════════════════════════════
# INVENTORY OPTIMIZATION  (per synopsis)
# ════════════════════════════════════════════════════
def reorder_point(avg_daily_demand, lead_time_days=7, service_level_z=1.65):
    """
    Reorder Point = (Avg Daily Demand × Lead Time) + Safety Stock
    Z = 1.65 for 95% service level
    """
    return round(avg_daily_demand * lead_time_days)

def safety_stock(std_daily_demand, lead_time_days=7, service_level_z=1.65):
    """Safety Stock = Z × StdDev × sqrt(Lead Time)"""
    return round(service_level_z * std_daily_demand * np.sqrt(lead_time_days))

def suggested_order_qty(current_stock, forecast_demand, safety_stk, buffer=1.1):
    """How many units to order to cover forecast + safety stock."""
    target = forecast_demand * buffer + safety_stk
    return max(0, round(target - current_stock))

def inventory_status(current_stock, forecast_demand):
    ratio = current_stock / max(forecast_demand, 1)
    if ratio < 0.5:   return "🔴 Critical"
    elif ratio < 0.9: return "🟡 Low Stock"
    elif ratio > 1.8: return "🔵 Overstock"
    else:             return "🟢 Healthy"
