import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(page_title="Dynamic Pricing Agent", page_icon="🤖", layout="wide")

# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load('models/demand_model.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    product_map  = joblib.load('models/product_map.pkl')
    category_map = joblib.load('models/category_map.pkl')
    with open('models/model_results.json') as f:
        results = json.load(f)
    df = pd.read_csv('data/processed_data.csv')
    return model, feature_cols, product_map, category_map, results, df

model, feature_cols, product_map, category_map, model_results, df = load_artifacts()

# ── Helpers ──────────────────────────────────────────────────────────────────
def predict_demand(fv: np.ndarray) -> float:
    return max(0.0, float(model.predict(fv.reshape(1, -1))[0]))

def build_fv(inputs: dict) -> np.ndarray:
    return np.array([inputs[c] for c in feature_cols], dtype=float)

def revenue_curve(base_fv, price_range):
    revenues, demands = [], []
    pi = feature_cols.index('unit_price')
    si = feature_cols.index('price_score')
    vi = feature_cols.index('price_volume')
    ps_val  = base_fv[feature_cols.index('product_score')]
    vol_val = base_fv[feature_cols.index('volume')]
    for p in price_range:
        fv = base_fv.copy()
        fv[pi] = p
        fv[si] = p * ps_val
        fv[vi] = p * vol_val
        d = predict_demand(fv)
        revenues.append(p * d)
        demands.append(d)
    return revenues, demands

from scipy.optimize import minimize_scalar

def find_optimal_price(base_fv, bounds=(1.0, 500.0)):
    pi = feature_cols.index('unit_price')
    si = feature_cols.index('price_score')
    vi = feature_cols.index('price_volume')
    ps_val  = base_fv[feature_cols.index('product_score')]
    vol_val = base_fv[feature_cols.index('volume')]

    def neg_rev(price):
        fv = base_fv.copy()
        fv[pi] = price
        fv[si] = price * ps_val
        fv[vi] = price * vol_val
        return -(price * predict_demand(fv))

    res = minimize_scalar(neg_rev, bounds=bounds, method='bounded')
    return float(res.x), float(-res.fun)

# ── Page config ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px; padding: 20px; text-align: center; color: white;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Based Dynamic Pricing Agent")
st.caption("Predict optimal product prices using machine learning — maximizing revenue based on demand, competition & seasonality.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💰 Price Optimizer", "📊 Data Insights", "🏆 Model Performance"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price Optimizer
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Product Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Product Info**")
        product_id_enc       = st.selectbox("Product ID", sorted(set(product_map.values())),
                                             format_func=lambda x: f"Product #{x}")
        product_category_enc = st.selectbox("Category", sorted(set(category_map.values())),
                                             format_func=lambda x: f"Category #{x}")
        unit_price           = st.number_input("Current Unit Price ($)", min_value=1.0, value=50.0, step=1.0)
        product_score        = st.slider("Product Rating", 1.0, 5.0, 4.0, 0.1)
        product_photos_qty   = st.number_input("Number of Photos", min_value=1, value=2, step=1)

    with c2:
        st.markdown("**Physical Attributes**")
        product_name_lenght        = st.number_input("Product Name Length", min_value=1, value=40, step=1)
        product_description_lenght = st.number_input("Description Length",  min_value=1, value=200, step=10)
        product_weight_g           = st.number_input("Weight (g)",          min_value=1, value=500, step=50)
        volume                     = st.number_input("Volume (cm³)",        min_value=1.0, value=5000.0, step=100.0)
        customers                  = st.number_input("Monthly Customers",   min_value=0, value=50, step=5)

    with c3:
        st.markdown("**Market & Time**")
        avg_comp_price  = st.number_input("Avg Competitor Price ($)", min_value=0.1, value=45.0, step=1.0)
        min_comp_price  = st.number_input("Min Competitor Price ($)", min_value=0.1, value=40.0, step=1.0)
        avg_comp_score  = st.slider("Avg Competitor Score", 1.0, 5.0, 3.8, 0.1)
        month           = st.selectbox("Month", list(range(1, 13)),
                                       format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                                               'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        year            = st.selectbox("Year", [2017, 2018, 2024, 2025])
        weekday         = st.selectbox("Weekday", list(range(7)),
                                       format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
        weekend         = 1 if weekday >= 5 else 0
        holiday         = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    # Derived
    price_vs_avg_comp = unit_price - avg_comp_price
    price_vs_min_comp = unit_price - min_comp_price
    price_score_feat  = unit_price * product_score
    price_volume_feat = unit_price * volume

    inputs = {
        'unit_price': unit_price,
        'product_name_lenght': product_name_lenght,
        'product_description_lenght': product_description_lenght,
        'product_photos_qty': product_photos_qty,
        'product_weight_g': product_weight_g,
        'product_score': product_score,
        'customers': customers,
        'weekday': weekday,
        'weekend': weekend,
        'holiday': holiday,
        'month': month,
        'year': year,
        'volume': volume,
        'avg_comp_price': avg_comp_price,
        'min_comp_price': min_comp_price,
        'avg_comp_score': avg_comp_score,
        'price_vs_avg_comp': price_vs_avg_comp,
        'price_vs_min_comp': price_vs_min_comp,
        'price_score': price_score_feat,
        'price_volume': price_volume_feat,
        'product_id_enc': product_id_enc,
        'product_category_enc': product_category_enc,
    }

    st.divider()
    col_btn, col_bounds = st.columns([1, 2])
    with col_bounds:
        price_min, price_max = st.slider("Price Search Range ($)", 1, 1000, (5, 300))

    if st.button("🔍 Find Optimal Price", use_container_width=True, type="primary"):
        base_fv = build_fv(inputs)
        opt_price, max_revenue = find_optimal_price(base_fv, bounds=(float(price_min), float(price_max)))

        # Current metrics
        curr_demand  = predict_demand(base_fv)
        curr_revenue = unit_price * curr_demand

        # Optimal demand
        opt_fv = base_fv.copy()
        opt_fv[feature_cols.index('unit_price')]    = opt_price
        opt_fv[feature_cols.index('price_score')]   = opt_price * product_score
        opt_fv[feature_cols.index('price_volume')]  = opt_price * volume
        opt_demand = predict_demand(opt_fv)

        rev_change_pct = ((max_revenue - curr_revenue) / max(curr_revenue, 0.01)) * 100

        st.markdown("### Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Optimal Price",    f"${opt_price:.2f}",    f"vs current ${unit_price:.2f}")
        m2.metric("Expected Demand",  f"{opt_demand:.1f} units", f"{opt_demand - curr_demand:+.1f}")
        m3.metric("Expected Revenue", f"${max_revenue:.2f}",  f"{rev_change_pct:+.1f}%")
        m4.metric("Current Revenue",  f"${curr_revenue:.2f}", "baseline")

        if opt_price > unit_price:
            st.success(f"📈 Raise price to **${opt_price:.2f}** — projected revenue increase of **{rev_change_pct:.1f}%**")
        elif opt_price < unit_price:
            st.warning(f"📉 Lower price to **${opt_price:.2f}** — projected revenue change of **{rev_change_pct:.1f}%**")
        else:
            st.info("✅ Current price is already optimal.")

        # Revenue curve chart
        price_range = np.linspace(price_min, price_max, 120)
        revenues, demands = revenue_curve(base_fv, price_range)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_range, y=revenues, mode='lines',
                                 name='Revenue', line=dict(color='#2d6a9f', width=2)))
        fig.add_vline(x=opt_price,   line_dash='dash', line_color='green',
                      annotation_text=f"Optimal ${opt_price:.2f}", annotation_position="top right")
        fig.add_vline(x=unit_price,  line_dash='dot',  line_color='orange',
                      annotation_text=f"Current ${unit_price:.2f}", annotation_position="top left")
        fig.update_layout(title="Revenue vs Price Curve", xaxis_title="Price ($)",
                          yaxis_title="Expected Revenue ($)", template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Demand curve
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=price_range, y=demands, mode='lines',
                                  name='Demand', line=dict(color='#e07b39', width=2), fill='tozeroy'))
        fig2.add_vline(x=opt_price, line_dash='dash', line_color='green')
        fig2.update_layout(title="Demand vs Price Curve", xaxis_title="Price ($)",
                           yaxis_title="Predicted Demand (units)", template="plotly_dark", height=320)
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Data Insights
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Dataset Overview")
    raw = pd.read_csv('data/retail_price_dataset.csv')

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Records",    len(raw))
    r2.metric("Unique Products",  raw['product_id'].nunique())
    r3.metric("Categories",       raw['product_category_name'].nunique())
    r4.metric("Avg Unit Price",   f"${raw['unit_price'].mean():.2f}")

    c_left, c_right = st.columns(2)

    with c_left:
        fig = px.histogram(raw, x='unit_price', nbins=40, title='Unit Price Distribution',
                           color_discrete_sequence=['#2d6a9f'], template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(raw, x='unit_price', y='qty', color='product_category_name',
                         title='Price vs Demand', template='plotly_dark', opacity=0.6,
                         labels={'unit_price': 'Unit Price ($)', 'qty': 'Quantity Sold'})
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        fig = px.histogram(raw, x='qty', nbins=40, title='Demand (Qty) Distribution',
                           color_discrete_sequence=['#e07b39'], template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

        monthly = raw.groupby('month')['qty'].mean().reset_index()
        fig = px.bar(monthly, x='month', y='qty', title='Avg Demand by Month',
                     template='plotly_dark', color='qty', color_continuous_scale='Blues',
                     labels={'month': 'Month', 'qty': 'Avg Qty Sold'})
        st.plotly_chart(fig, use_container_width=True)

    # Category breakdown
    cat_avg = raw.groupby('product_category_name').agg(
        avg_price=('unit_price', 'mean'), avg_qty=('qty', 'mean'), count=('qty', 'count')
    ).reset_index().sort_values('avg_price', ascending=False)

    fig = px.bar(cat_avg, x='product_category_name', y='avg_price',
                 title='Avg Price by Category', template='plotly_dark',
                 color='avg_qty', color_continuous_scale='Viridis',
                 labels={'product_category_name': 'Category', 'avg_price': 'Avg Price ($)'})
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Competitor comparison
    raw['avg_comp'] = raw[['comp_1', 'comp_2', 'comp_3']].mean(axis=1)
    raw['price_diff'] = raw['unit_price'] - raw['avg_comp']
    fig = px.scatter(raw, x='price_diff', y='qty', title='Price vs Competitor Gap → Demand Impact',
                     template='plotly_dark', opacity=0.6,
                     labels={'price_diff': 'Our Price − Avg Competitor Price ($)', 'qty': 'Qty Sold'},
                     color_discrete_sequence=['#9b59b6'])
    fig.add_vline(x=0, line_dash='dash', line_color='white', annotation_text="Same as competitor")
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Comparison")
    best_name = model_results['best_model']
    results   = model_results['results']

    perf_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    perf_df['Best'] = perf_df['Model'].apply(lambda x: "⭐ Best" if x == best_name else "")

    st.dataframe(perf_df.style.highlight_max(subset=['R2'], color='#1a5276')
                              .highlight_min(subset=['MAE', 'RMSE'], color='#1a5276'),
                 use_container_width=True)

    fig = go.Figure()
    colors = ['#2ecc71' if m == best_name else '#2d6a9f' for m in perf_df['Model']]
    fig.add_trace(go.Bar(x=perf_df['Model'], y=perf_df['R2'], marker_color=colors, name='R² Score'))
    fig.update_layout(title='R² Score by Model (higher = better)', template='plotly_dark',
                      yaxis_title='R² Score', height=350)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=perf_df['Model'], y=perf_df['MAE'],
                             marker_color='#e07b39', name='MAE'))
        fig.update_layout(title='MAE (lower = better)', template='plotly_dark', height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=perf_df['Model'], y=perf_df['RMSE'],
                             marker_color='#9b59b6', name='RMSE'))
        fig.update_layout(title='RMSE (lower = better)', template='plotly_dark', height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"✅ Best model: **{best_name}** — saved and used for all predictions above.")

    st.markdown("### How It Works")
    st.markdown("""
    1. **Data** — 676 retail records with price, demand, competitor prices, seasonality & product attributes.
    2. **Features** — 22 engineered features including competitor gap, price-score interaction, and seasonality.
    3. **Model** — Trained 4 ML models; best selected by R² on held-out test set.
    4. **Optimization** — `scipy.optimize.minimize_scalar` finds the price that maximizes `price × predicted_demand`.
    5. **Output** — Optimal price, expected demand, revenue curve, and comparison with current price.
    """)
