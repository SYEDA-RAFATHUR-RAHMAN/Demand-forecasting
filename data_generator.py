"""
data_generator.py
Generates 2 years of realistic monthly retail sales data
for 5 products. Saved as CSV — used by the Streamlit app.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

PRODUCTS = {
    "P001": {"name": "Wireless Earbuds",  "category": "Electronics", "base": 320, "price": 1299},
    "P002": {"name": "Running Shoes",     "category": "Footwear",    "base": 180, "price": 2499},
    "P003": {"name": "Rice 5kg",          "category": "Groceries",   "base": 540, "price": 350},
    "P004": {"name": "Laptop Backpack",   "category": "Accessories", "base": 210, "price": 899},
    "P005": {"name": "Face Moisturizer",  "category": "Beauty",      "base": 290, "price": 599},
}

SEASONAL = {1:0.90, 2:0.80, 3:0.85, 4:0.95, 5:1.00, 6:1.05,
            7:1.10, 8:1.00, 9:1.05, 10:1.20, 11:1.40, 12:1.30}

def generate():
    records = []
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    for pid, info in PRODUCTS.items():
        for i, dt in enumerate(months):
            trend  = 1 + i * 0.015
            season = SEASONAL[dt.month]
            if info["category"] == "Footwear" and dt.month in [6,7]:
                season *= 1.25
            noise  = np.random.uniform(0.85, 1.15)
            units  = max(10, int(info["base"] * trend * season * noise))
            records.append({
                "ds":           dt,                    # Prophet uses 'ds' for date
                "product_id":   pid,
                "product_name": info["name"],
                "category":     info["category"],
                "units_sold":   units,
                "unit_price":   info["price"],
                "revenue":      units * info["price"],
            })
    df = pd.DataFrame(records)
    out = os.path.join(os.path.dirname(__file__), "sales_data.csv")
    df.to_csv(out, index=False)
    print(f"✅ sales_data.csv saved — {len(df)} rows")
    return df

def get_inventory():
    rows = [
        {"product_id":"P001","product_name":"Wireless Earbuds",  "category":"Electronics","current_stock":480,"unit_price":1299,"reorder_level":160},
        {"product_id":"P002","product_name":"Running Shoes",      "category":"Footwear",   "current_stock":120,"unit_price":2499,"reorder_level":90},
        {"product_id":"P003","product_name":"Rice 5kg",           "category":"Groceries",  "current_stock":900,"unit_price":350, "reorder_level":270},
        {"product_id":"P004","product_name":"Laptop Backpack",    "category":"Accessories","current_stock":310,"unit_price":899, "reorder_level":105},
        {"product_id":"P005","product_name":"Face Moisturizer",   "category":"Beauty",     "current_stock":205,"unit_price":599, "reorder_level":145},
    ]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    generate()
