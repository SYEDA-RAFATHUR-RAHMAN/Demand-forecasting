"""
app.py  ←  THIS IS THE FILE YOU RUN
═══════════════════════════════════════════════════════
AI-Driven Demand Forecasting & Inventory Optimization
VTU 6th Semester — CSE Major Project
Ghousia College of Engineering
═══════════════════════════════════════════════════════
HOW TO RUN:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from data_generator import generate, get_inventory
from models import (
    moving_average_forecast, linear_regression_forecast,
    random_forest_forecast, prophet_forecast,
    evaluate_all_models, reorder_point, safety_stock,
    suggested_order_qty, inventory_status
)

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="AI Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
    .main-title {
        font-size: 2rem; font-weight: 800; color: #58a6ff;
        text-align: center; padding: 10px 0 4px;
    }
    .sub-title {
        font-size: 0.95rem; color: #8b949e;
        text-align: center; margin-bottom: 24px;
    }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 18px 20px; text-align: center;
    }
    .metric-val  { font-size: 2rem; font-weight: 800; color: #58a6ff; }
    .metric-lab  { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }
    .status-card {
        background: #161b22; border-radius: 10px;
        padding: 14px 16px; margin-bottom: 10px;
        border-left: 4px solid #58a6ff;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #e6edf3;
        border-bottom: 1px solid #30363d; padding-bottom: 8px;
        margin: 20px 0 14px;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"]    label { color: #c9d1d9 !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# LOAD DATA  (cached so it doesn't reload every click)
# ════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("sales_data.csv", parse_dates=["ds"])
    except FileNotFoundError:
        df = generate()
        df = pd.read_csv("sales_data.csv", parse_dates=["ds"])
    inv = get_inventory()
    return df, inv

sales_df, inv_df = load_data()
PRODUCTS     = sales_df["product_name"].unique().tolist()
PRODUCT_IDS  = sales_df["product_id"].unique().tolist()


# ════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    selected_product = st.selectbox(
        "Select Product",
        options=PRODUCTS,
        index=0
    )

    forecast_months = st.slider(
        "Forecast Horizon (months)", 1, 6, 3
    )

    model_choice = st.selectbox(
        "Forecasting Model",
        ["Prophet (Recommended)", "Random Forest", "Linear Regression", "Moving Average"]
    )

    st.markdown("---")
    st.markdown("### 📋 Project Info")
    st.markdown("""
    **Title:** AI-Driven Demand Forecasting & Inventory Optimization

    **College:** Ghousia College of Engineering

    **Guide:** Dr T Venkatesh

    **Team:**
    - Syeda Kounain (1GC23CS158)
    - Syeda Qurratul Ain Aliya (1GC23CS159)
    - Syeda Rafath Ur Rahman (1GC23CS160)
    - Tejaswini D (1GC23CS162)
    """)


# ════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════
st.markdown('<div class="main-title">📦 AI-Driven Demand Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Retail Supply Chain · VTU 6th Semester CSE Project · Ghousia College of Engineering</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🔮 Demand Forecast",
    "📦 Inventory Optimizer",
    "🏆 Model Comparison"
])


# ─── Get product data ────────────────────────────────
pid_row    = sales_df[sales_df["product_name"] == selected_product]
pid        = pid_row["product_id"].iloc[0]
prod_df    = pid_row.sort_values("ds")
dates      = prod_df["ds"].tolist()
series     = prod_df["units_sold"].tolist()
inv_row    = inv_df[inv_df["product_id"] == pid].iloc[0]
cur_stock  = int(inv_row["current_stock"])


# ════════════════════════════════════════════════════
# TAB 1 — DASHBOARD (overview of all products)
# ════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📈 Overall Performance — All Products</div>', unsafe_allow_html=True)

    # KPI Row
    total_rev   = sales_df["revenue"].sum()
    total_units = sales_df["units_sold"].sum()
    avg_monthly = sales_df.groupby("ds")["revenue"].sum().mean()
    num_products = sales_df["product_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">₹{total_rev/1e6:.1f}M</div><div class="metric-lab">Total Revenue (24 months)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{total_units:,}</div><div class="metric-lab">Total Units Sold</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">₹{avg_monthly/1e5:.1f}L</div><div class="metric-lab">Avg Monthly Revenue</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{num_products}</div><div class="metric-lab">Products Tracked</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue trend chart
    monthly_rev = sales_df.groupby("ds")["revenue"].sum().reset_index()
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=monthly_rev["ds"], y=monthly_rev["revenue"],
        fill="tozeroy", fillcolor="rgba(88,166,255,0.12)",
        line=dict(color="#58a6ff", width=2.5),
        mode="lines+markers", marker=dict(size=5),
        name="Monthly Revenue"
    ))
    fig_rev.update_layout(
        title="Monthly Revenue Trend — All Products",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"), height=300,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d",
                   tickformat="₹,.0f",
                   tickprefix="₹")
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # Sales by category
    cat_df = sales_df.groupby("category")["units_sold"].sum().reset_index()
    fig_cat = px.pie(
        cat_df, names="category", values="units_sold",
        title="Units Sold by Category",
        color_discrete_sequence=px.colors.sequential.Blues_r,
        hole=0.4
    )
    fig_cat.update_layout(
        paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"), height=320
    )

    # Sales by product bar
    prod_rev = sales_df.groupby("product_name")["revenue"].sum().reset_index()
    fig_bar  = px.bar(
        prod_rev, x="product_name", y="revenue",
        title="Total Revenue by Product",
        color="revenue",
        color_continuous_scale="Blues",
        labels={"product_name": "Product", "revenue": "Revenue (₹)"}
    )
    fig_bar.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"), height=320,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        coloraxis_showscale=False,
        showlegend=False
    )

    col_a, col_b = st.columns(2)
    with col_a: st.plotly_chart(fig_cat, use_container_width=True)
    with col_b: st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 2 — DEMAND FORECAST
# ════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-header">🔮 Forecast: {selected_product} — {model_choice}</div>', unsafe_allow_html=True)

    # Run selected model
    with st.spinner("Running forecast model..."):
        model_name = model_choice.split(" ")[0]

        if model_name == "Prophet":
            try:
                preds, lower, upper, fut_dates, full_fc = prophet_forecast(
                    dates, series, periods=forecast_months
                )
                use_prophet = True
            except Exception as e:
                st.error(f"Prophet error: {e}. Falling back to Random Forest.")
                preds = random_forest_forecast(series, periods=forecast_months)
                use_prophet = False
        elif model_name == "Random":
            preds = random_forest_forecast(series, periods=forecast_months)
            use_prophet = False
        elif model_name == "Linear":
            preds = linear_regression_forecast(series, periods=forecast_months)
            use_prophet = False
        else:
            preds = moving_average_forecast(series, periods=forecast_months)
            use_prophet = False

    # KPIs
    avg_hist = round(np.mean(series))
    change   = round(((preds[0] - series[-1]) / series[-1]) * 100, 1)
    k1, k2, k3 = st.columns(3)
    k1.metric("Last Month Actual",  f"{series[-1]:,} units")
    k2.metric("Next Month Forecast", f"{preds[0]:,} units", f"{change:+}%")
    k3.metric("Avg Historical Demand", f"{avg_hist:,} units/month")

    # Forecast chart
    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=dates, y=series,
        name="Historical Sales",
        line=dict(color="#58a6ff", width=2.2),
        mode="lines+markers", marker=dict(size=5)
    ))

    # Forecast line
    import datetime
    last_date  = dates[-1]
    fut_dates_plot = [last_date + pd.DateOffset(months=i+1)
                      for i in range(forecast_months)]

    # Connect last actual to first forecast
    fig.add_trace(go.Scatter(
        x=[dates[-1]] + fut_dates_plot,
        y=[series[-1]] + preds,
        name=f"Forecast ({model_choice.split('(')[0].strip()})",
        line=dict(color="#f78166", width=2.5, dash="dash"),
        mode="lines+markers", marker=dict(size=7, symbol="diamond")
    ))

    # Confidence band (Prophet only)
    if use_prophet and "lower" in dir():
        fig.add_trace(go.Scatter(
            x=[dates[-1]] + fut_dates_plot + fut_dates_plot[::-1],
            y=[series[-1]] + upper + lower[::-1],
            fill="toself", fillcolor="rgba(247,129,102,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence Interval",
            showlegend=True
        ))

    # Divider line
    fig.add_vline(x=pd.Timestamp(dates[-1]).timestamp() * 1000, line_dash="dot",
                  line_color="#444c56", annotation_text="Forecast starts")

    fig.update_layout(
        title=f"Demand Forecast — {selected_product}",
        xaxis_title="Month", yaxis_title="Units Sold",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"), height=420,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown('<div class="section-header">Forecast Values</div>', unsafe_allow_html=True)
    fc_table = pd.DataFrame({
        "Month":           [d.strftime("%B %Y") for d in fut_dates_plot],
        "Forecasted Units": preds,
        "Expected Revenue (₹)": [p * int(inv_row["unit_price"]) for p in preds]
    })
    fc_table["Expected Revenue (₹)"] = fc_table["Expected Revenue (₹)"].apply(lambda x: f"₹{x:,}")
    st.dataframe(fc_table, use_container_width=True, hide_index=True)

    # Prophet trend decomposition
    if use_prophet:
        st.markdown('<div class="section-header">Prophet Trend & Seasonality</div>', unsafe_allow_html=True)
        st.info("📊 Prophet automatically decomposes demand into **trend** + **yearly seasonality** — this is what makes it more accurate than simple statistical models.")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=full_fc["ds"], y=full_fc["trend"],
            name="Trend", line=dict(color="#d2a8ff", width=2)
        ))
        fig_trend.update_layout(
            title="Demand Trend Component",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9"), height=280,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 3 — INVENTORY OPTIMIZER
# ════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📦 Inventory Optimization — All Products</div>', unsafe_allow_html=True)
    st.info("Inventory recommendations are calculated using **Reorder Point** and **Safety Stock** formulas based on forecasted demand.")

    rows = []
    for _, inv_row_i in inv_df.iterrows():
        p     = inv_row_i["product_id"]
        pname = inv_row_i["product_name"]
        s_df  = sales_df[sales_df["product_id"] == p].sort_values("ds")
        s     = s_df["units_sold"].tolist()
        d     = s_df["ds"].tolist()

        # Forecast next month
        try:
            fc, _, _, _, _ = prophet_forecast(d, s, periods=1)
            fc1 = fc[0]
        except:
            fc1 = random_forest_forecast(s, periods=1)[0]

        avg_daily   = fc1 / 30
        std_daily   = np.std(s) / 30
        rop         = reorder_point(avg_daily)
        ss          = safety_stock(std_daily)
        order_qty   = suggested_order_qty(inv_row_i["current_stock"], fc1, ss)
        status      = inventory_status(inv_row_i["current_stock"], fc1)

        rows.append({
            "Product":          pname,
            "Current Stock":    int(inv_row_i["current_stock"]),
            "Next Month Forecast": fc1,
            "Reorder Point":    rop,
            "Safety Stock":     ss,
            "Order Qty":        order_qty,
            "Status":           status,
        })

    inv_table = pd.DataFrame(rows)
    st.dataframe(inv_table, use_container_width=True, hide_index=True)

    # Inventory vs Forecast chart
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Bar(
        name="Current Stock",
        x=inv_table["Product"],
        y=inv_table["Current Stock"],
        marker_color="#1f6feb"
    ))
    fig_inv.add_trace(go.Bar(
        name="Next Month Forecast",
        x=inv_table["Product"],
        y=inv_table["Next Month Forecast"],
        marker_color="#f78166"
    ))
    fig_inv.add_trace(go.Scatter(
        name="Reorder Point",
        x=inv_table["Product"],
        y=inv_table["Reorder Point"],
        mode="markers+lines",
        marker=dict(size=10, color="#ffa657", symbol="diamond"),
        line=dict(color="#ffa657", dash="dot")
    ))
    fig_inv.update_layout(
        title="Stock vs Forecast vs Reorder Point",
        barmode="group",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"), height=400,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Units"),
    )
    st.plotly_chart(fig_inv, use_container_width=True)

    # Formulas explanation
    st.markdown('<div class="section-header">📐 Formulas Used</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Reorder Point (ROP)**
        ```
        ROP = Avg Daily Demand × Lead Time (days)
        Lead Time assumed = 7 days
        ```
        """)
    with col2:
        st.markdown("""
        **Safety Stock**
        ```
        SS = Z × StdDev(daily demand) × √Lead Time
        Z = 1.65  (95% service level)
        ```
        """)


# ════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-header">🏆 Model Accuracy Comparison — {selected_product}</div>', unsafe_allow_html=True)
    st.info("The last 3 months of data are used as a **test set**. All 4 models predict those months — lower MAE/RMSE = better accuracy.")

    with st.spinner("Evaluating all models..."):
        eval_results, actual_test = evaluate_all_models(dates, series, test_months=3)

    # Table
    rows_eval = []
    for mname, res in eval_results.items():
        best = min([r["MAE"] for r in eval_results.values()
                    if isinstance(r["MAE"], float)], default=0)
        is_best = isinstance(res["MAE"], float) and res["MAE"] == best
        rows_eval.append({
            "Model":  mname + (" ⭐ BEST" if is_best else ""),
            "MAE (units)":  res["MAE"],
            "RMSE (units)": res["RMSE"],
        })
    st.dataframe(pd.DataFrame(rows_eval), use_container_width=True, hide_index=True)

    # Bar chart
    valid = {k: v for k, v in eval_results.items()
             if isinstance(v["MAE"], float)}
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Bar(
        name="MAE",
        x=list(valid.keys()),
        y=[v["MAE"] for v in valid.values()],
        marker_color="#58a6ff"
    ))
    fig_eval.add_trace(go.Bar(
        name="RMSE",
        x=list(valid.keys()),
        y=[v["RMSE"] for v in valid.values()],
        marker_color="#56d364"
    ))
    fig_eval.update_layout(
        title="MAE & RMSE Comparison (lower = better)",
        barmode="group",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"), height=380,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Error (units)"),
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    # Predictions vs Actual chart
    test_months_list = dates[-3:]
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=test_months_list, y=actual_test,
        name="Actual", line=dict(color="#58a6ff", width=3),
        mode="lines+markers", marker=dict(size=8)
    ))
    colors = {"Moving Average":"#d2a8ff","Linear Regression":"#ffa657",
              "Random Forest":"#56d364","Prophet":"#f78166"}
    for mname, res in eval_results.items():
        if res["preds"]:
            fig_pred.add_trace(go.Scatter(
                x=test_months_list, y=res["preds"],
                name=mname, line=dict(color=colors.get(mname,"#aaa"), dash="dot"),
                mode="lines+markers", marker=dict(size=6)
            ))
    fig_pred.update_layout(
        title="Predicted vs Actual (Test Set — Last 3 Months)",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"), height=360,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Units"),
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown('<div class="section-header">Why Prophet?</div>', unsafe_allow_html=True)
    st.markdown("""
    | Feature | Moving Avg | Linear Regression | Random Forest | **Prophet** |
    |---|---|---|---|---|
    | Handles Trend | ❌ | ✅ | ✅ | ✅ |
    | Handles Seasonality | ❌ | ❌ | Partial | ✅ |
    | Confidence Intervals | ❌ | ❌ | ❌ | ✅ |
    | Interpretable | ✅ | ✅ | ❌ | ✅ |
    | Industry Standard | ❌ | ❌ | ❌ | ✅ (Meta/Amazon) |
    """)
