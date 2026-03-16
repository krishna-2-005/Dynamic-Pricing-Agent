import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize_scalar

st.set_page_config(page_title="Dynamic Pricing Agent", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 50%, #0a1628 100%); }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 2rem 2.5rem; max-width: 1400px; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f2847 0%, #1a3a6b 40%, #0d2d5e 70%, #162040 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute; bottom: -30%; left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(167,139,250,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #a78bfa, #f6ad55);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0; line-height: 1.2;
}
.hero-sub {
    color: rgba(255,255,255,0.6); font-size: 1rem; font-weight: 400; margin: 0;
}
.hero-badges { margin-top: 1.2rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
.badge {
    background: rgba(99,179,237,0.12); border: 1px solid rgba(99,179,237,0.25);
    color: #63b3ed; padding: 0.3rem 0.9rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px; padding: 4px; gap: 4px;
    border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; padding: 0.6rem 1.4rem;
    font-weight: 600; font-size: 0.9rem; color: rgba(255,255,255,0.5);
    background: transparent; border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a3a6b, #2d5a9e) !important;
    color: white !important; box-shadow: 0 4px 15px rgba(45,90,158,0.4);
}

/* Section cards */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem;
}
.section-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #63b3ed; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}

/* Inputs */
.stNumberInput input, .stSelectbox select, div[data-baseweb="select"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: white !important;
    font-size: 0.9rem !important;
}
.stNumberInput input:focus, div[data-baseweb="select"]:focus-within {
    border-color: rgba(99,179,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
}
label { color: rgba(255,255,255,0.7) !important; font-size: 0.82rem !important; font-weight: 500 !important; }

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a6b 0%, #2d5a9e 50%, #3b6fd4 100%) !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.75rem 2rem !important;
    color: white !important; letter-spacing: 0.5px;
    box-shadow: 0 4px 20px rgba(45,90,158,0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(45,90,158,0.6) !important;
}

/* Result metric cards */
.result-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }
.result-card {
    background: linear-gradient(135deg, rgba(26,58,107,0.6), rgba(45,90,158,0.3));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center;
    backdrop-filter: blur(10px);
}
.result-card.green { background: linear-gradient(135deg, rgba(16,85,60,0.6), rgba(22,101,52,0.3)); border-color: rgba(52,211,153,0.25); }
.result-card.amber { background: linear-gradient(135deg, rgba(92,60,10,0.6), rgba(120,80,10,0.3)); border-color: rgba(251,191,36,0.25); }
.result-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: rgba(255,255,255,0.5); margin-bottom: 0.5rem; }
.result-value { font-size: 1.8rem; font-weight: 800; color: white; line-height: 1; }
.result-delta { font-size: 0.8rem; margin-top: 0.4rem; color: rgba(255,255,255,0.5); }
.result-delta.pos { color: #34d399; }
.result-delta.neg { color: #f87171; }

/* Insight banner */
.insight-banner {
    border-radius: 12px; padding: 1rem 1.5rem;
    font-weight: 600; font-size: 0.95rem; margin: 1rem 0;
    display: flex; align-items: center; gap: 0.8rem;
}
.insight-banner.up { background: rgba(16,85,60,0.3); border: 1px solid rgba(52,211,153,0.3); color: #34d399; }
.insight-banner.down { background: rgba(92,60,10,0.3); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24; }
.insight-banner.neutral { background: rgba(30,58,138,0.3); border: 1px solid rgba(99,179,237,0.3); color: #63b3ed; }

/* Stat cards for data tab */
.stat-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.2rem 1.5rem; text-align: center;
}
.stat-num { font-size: 2rem; font-weight: 800; color: #63b3ed; }
.stat-lbl { font-size: 0.75rem; color: rgba(255,255,255,0.5); font-weight: 500; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.8px; }

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
    margin: 1.5rem 0;
}

/* Slider */
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_demand(fv):
    return max(0.0, float(model.predict(fv.reshape(1, -1))[0]))

def build_fv(inputs):
    return np.array([inputs[c] for c in feature_cols], dtype=float)

def revenue_curve(base_fv, price_range):
    revenues, demands = [], []
    pi = feature_cols.index('unit_price')
    si = feature_cols.index('price_score')
    vi = feature_cols.index('price_volume')
    ps_val  = base_fv[feature_cols.index('product_score')]
    vol_val = base_fv[feature_cols.index('volume')]
    for p in price_range:
        fv = base_fv.copy(); fv[pi]=p; fv[si]=p*ps_val; fv[vi]=p*vol_val
        d = predict_demand(fv)
        revenues.append(p * d); demands.append(d)
    return revenues, demands

def find_optimal_price(base_fv, bounds=(1.0, 500.0)):
    pi = feature_cols.index('unit_price')
    si = feature_cols.index('price_score')
    vi = feature_cols.index('price_volume')
    ps_val  = base_fv[feature_cols.index('product_score')]
    vol_val = base_fv[feature_cols.index('volume')]
    def neg_rev(price):
        fv = base_fv.copy(); fv[pi]=price; fv[si]=price*ps_val; fv[vi]=price*vol_val
        return -(price * predict_demand(fv))
    res = minimize_scalar(neg_rev, bounds=bounds, method='bounded')
    return float(res.x), float(-res.fun)

def _f(v, d=0.0): return float(v) if v is not None and v != "" else d
def _i(v, d=0):   return int(v)   if v is not None and v != "" else d

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
)

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🤖 Dynamic Pricing Agent</p>
  <p class="hero-sub">AI-powered price optimization using machine learning — maximize revenue through demand forecasting, competitor analysis & seasonality intelligence.</p>
  <div class="hero-badges">
    <span class="badge">⚡ Random Forest</span>
    <span class="badge">📊 XGBoost</span>
    <span class="badge">🎯 Revenue Optimization</span>
    <span class="badge">🏪 676 Retail Records</span>
    <span class="badge">22 Features</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💰  Price Optimizer", "📊  Data Insights", "🏆  Model Performance"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price Optimizer
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown('<div class="section-title">🏷️ Product Info</div>', unsafe_allow_html=True)
        product_names    = [""] + sorted(product_map.keys())
        category_names   = [""] + sorted(category_map.keys())
        selected_product  = st.selectbox("Product", product_names)
        selected_category = st.selectbox("Category", category_names)
        product_id_enc       = product_map.get(selected_product, 0)
        product_category_enc = category_map.get(selected_category, 0)
        unit_price        = st.number_input("Current Unit Price ($)", min_value=0.0, value=None, step=1.0, placeholder="e.g. 50.00")
        product_score     = st.number_input("Product Rating (1–5)", min_value=0.0, max_value=5.0, value=None, step=0.1, placeholder="e.g. 4.2")
        product_photos_qty = st.number_input("Number of Photos", min_value=0, value=None, step=1, placeholder="e.g. 3")

    with c2:
        st.markdown('<div class="section-title">📦 Physical Attributes</div>', unsafe_allow_html=True)
        product_name_lenght        = st.number_input("Product Name Length",  min_value=0, value=None, step=1,    placeholder="e.g. 40")
        product_description_lenght = st.number_input("Description Length",   min_value=0, value=None, step=10,   placeholder="e.g. 200")
        product_weight_g           = st.number_input("Weight (g)",           min_value=0, value=None, step=50,   placeholder="e.g. 500")
        volume                     = st.number_input("Volume (cm³)",         min_value=0.0, value=None, step=100.0, placeholder="e.g. 5000")
        customers                  = st.number_input("Monthly Customers",    min_value=0, value=None, step=5,    placeholder="e.g. 50")

    with c3:
        st.markdown('<div class="section-title">🌐 Market & Time</div>', unsafe_allow_html=True)
        avg_comp_price = st.number_input("Avg Competitor Price ($)", min_value=0.0, value=None, step=1.0, placeholder="e.g. 45.00")
        min_comp_price = st.number_input("Min Competitor Price ($)", min_value=0.0, value=None, step=1.0, placeholder="e.g. 40.00")
        avg_comp_score = st.number_input("Avg Competitor Score (1–5)", min_value=0.0, max_value=5.0, value=None, step=0.1, placeholder="e.g. 3.8")
        month   = st.selectbox("Month", [""] + list(range(1,13)),
                               format_func=lambda x: x if x=="" else ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        year    = st.selectbox("Year", ["", 2017, 2018, 2024, 2025])
        weekday = st.selectbox("Weekday", [""] + list(range(7)),
                               format_func=lambda x: x if x=="" else ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
        weekend = 1 if (weekday != "" and weekday >= 5) else 0
        holiday = st.selectbox("Holiday?", ["", 0, 1], format_func=lambda x: "" if x=="" else ("Yes" if x else "No"))

    # Track raw values for validation (before coercion)
    _raw = {
        'unit_price': unit_price, 'product_score': product_score,
        'avg_comp_price': avg_comp_price, 'min_comp_price': min_comp_price,
        'volume': volume, 'product_weight_g': product_weight_g,
        'customers': customers, 'month': month, 'year': year,
        'weekday': weekday, 'product': selected_product, 'category': selected_category,
    }

    # Coerce
    unit_price     = _f(unit_price);    product_score  = _f(product_score)
    avg_comp_price = _f(avg_comp_price); min_comp_price = _f(min_comp_price)
    avg_comp_score = _f(avg_comp_score); volume         = _f(volume)
    product_name_lenght        = _i(product_name_lenght)
    product_description_lenght = _i(product_description_lenght)
    product_photos_qty         = _i(product_photos_qty)
    product_weight_g           = _i(product_weight_g)
    customers = _i(customers)
    month   = _i(month, 1);  year    = _i(year, 2024)
    weekday = _i(weekday, 0); holiday = _i(holiday, 0)

    price_vs_avg_comp = unit_price - avg_comp_price
    price_vs_min_comp = unit_price - min_comp_price
    price_score_feat  = unit_price * product_score
    price_volume_feat = unit_price * volume

    inputs = {
        'unit_price': unit_price, 'product_name_lenght': product_name_lenght,
        'product_description_lenght': product_description_lenght,
        'product_photos_qty': product_photos_qty, 'product_weight_g': product_weight_g,
        'product_score': product_score, 'customers': customers,
        'weekday': weekday, 'weekend': weekend, 'holiday': holiday,
        'month': month, 'year': year, 'volume': volume,
        'avg_comp_price': avg_comp_price, 'min_comp_price': min_comp_price,
        'avg_comp_score': avg_comp_score, 'price_vs_avg_comp': price_vs_avg_comp,
        'price_vs_min_comp': price_vs_min_comp, 'price_score': price_score_feat,
        'price_volume': price_volume_feat, 'product_id_enc': product_id_enc,
        'product_category_enc': product_category_enc,
    }

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    col_range, col_btn = st.columns([2, 1], gap="medium")
    with col_range:
        st.markdown('<div class="section-title">🎯 Price Search Range ($)</div>', unsafe_allow_html=True)
        price_min, price_max = st.slider("", 1, 1000, (5, 300), label_visibility="collapsed")
        st.caption(f"Optimizer will search for the best price between **${price_min}** and **${price_max}**")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🔍  Find Optimal Price", use_container_width=True, type="primary")

    if run:
        # Validate required fields
        missing = []
        if not _raw['product']:              missing.append("Product")
        if not _raw['category']:             missing.append("Category")
        if _raw['unit_price']   in (None, ""): missing.append("Current Unit Price")
        if _raw['product_score'] in (None, ""): missing.append("Product Rating")
        if _raw['avg_comp_price'] in (None, ""): missing.append("Avg Competitor Price")
        if _raw['min_comp_price'] in (None, ""): missing.append("Min Competitor Price")
        if _raw['volume']        in (None, ""): missing.append("Volume")
        if _raw['product_weight_g'] in (None, ""): missing.append("Weight")
        if _raw['customers']     in (None, ""): missing.append("Monthly Customers")
        if _raw['month']         == "":        missing.append("Month")
        if _raw['year']          == "":        missing.append("Year")
        if _raw['weekday']       == "":        missing.append("Weekday")

        if missing:
            st.markdown(f"""
            <div style="background:rgba(127,29,29,0.35);border:1px solid rgba(248,113,113,0.35);
                        border-radius:12px;padding:1rem 1.5rem;margin-top:1rem;">
              <div style="color:#f87171;font-weight:700;font-size:0.95rem;margin-bottom:0.5rem">
                ⚠️ Please fill in all required fields before running the optimizer
              </div>
              <div style="color:rgba(255,255,255,0.6);font-size:0.85rem">
                Missing: <strong style="color:#fca5a5">{", ".join(missing)}</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Running price optimization..."):
                base_fv = build_fv(inputs)
                opt_price, max_revenue = find_optimal_price(base_fv, bounds=(float(price_min), float(price_max)))
                curr_demand  = predict_demand(base_fv)
                curr_revenue = unit_price * curr_demand
                opt_fv = base_fv.copy()
                opt_fv[feature_cols.index('unit_price')]   = opt_price
                opt_fv[feature_cols.index('price_score')]  = opt_price * product_score
                opt_fv[feature_cols.index('price_volume')] = opt_price * volume
                opt_demand = predict_demand(opt_fv)
                rev_change_pct = ((max_revenue - curr_revenue) / max(curr_revenue, 0.01)) * 100
                price_change   = opt_price - unit_price

            # Result cards
            delta_cls = "pos" if rev_change_pct >= 0 else "neg"
            price_cls = "pos" if price_change >= 0 else "neg"
            st.markdown(f"""
            <div class="result-grid">
              <div class="result-card green">
                <div class="result-label">Optimal Price</div>
                <div class="result-value">${opt_price:.2f}</div>
                <div class="result-delta {price_cls}">{price_change:+.2f} vs current</div>
              </div>
              <div class="result-card">
                <div class="result-label">Expected Demand</div>
                <div class="result-value">{opt_demand:.1f}</div>
                <div class="result-delta {delta_cls}">{opt_demand - curr_demand:+.1f} units</div>
              </div>
              <div class="result-card green">
                <div class="result-label">Expected Revenue</div>
                <div class="result-value">${max_revenue:.2f}</div>
                <div class="result-delta {delta_cls}">{rev_change_pct:+.1f}%</div>
              </div>
              <div class="result-card amber">
                <div class="result-label">Current Revenue</div>
                <div class="result-value">${curr_revenue:.2f}</div>
                <div class="result-delta">@ ${unit_price:.2f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if opt_price > unit_price:
                st.markdown(f'<div class="insight-banner up">📈 Raise price to <strong>${opt_price:.2f}</strong> — projected revenue increase of <strong>{rev_change_pct:.1f}%</strong></div>', unsafe_allow_html=True)
            elif opt_price < unit_price:
                st.markdown(f'<div class="insight-banner down">📉 Lower price to <strong>${opt_price:.2f}</strong> — projected revenue change of <strong>{rev_change_pct:.1f}%</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-banner neutral">✅ Current price is already at the optimal point.</div>', unsafe_allow_html=True)

            # Charts
            price_range = np.linspace(price_min, price_max, 150)
            revenues, demands = revenue_curve(base_fv, price_range)

            ch1, ch2 = st.columns(2, gap="medium")
            with ch1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_range, y=revenues, mode='lines', name='Revenue',
                    line=dict(color='#63b3ed', width=2.5),
                    fill='tozeroy', fillcolor='rgba(99,179,237,0.07)'
                ))
                fig.add_vline(x=opt_price, line_dash='dash', line_color='#34d399', line_width=2,
                              annotation_text=f"Optimal ${opt_price:.2f}", annotation_font_color='#34d399')
                fig.add_vline(x=unit_price, line_dash='dot', line_color='#f6ad55', line_width=1.5,
                              annotation_text=f"Current ${unit_price:.2f}", annotation_font_color='#f6ad55')
                fig.update_layout(**CHART_LAYOUT, title=dict(text="Revenue vs Price", font=dict(size=15, color='white')), height=360)
                st.plotly_chart(fig, use_container_width=True)

            with ch2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=price_range, y=demands, mode='lines', name='Demand',
                    line=dict(color='#a78bfa', width=2.5),
                    fill='tozeroy', fillcolor='rgba(167,139,250,0.07)'
                ))
                fig2.add_vline(x=opt_price, line_dash='dash', line_color='#34d399', line_width=2,
                               annotation_text=f"Optimal ${opt_price:.2f}", annotation_font_color='#34d399')
                fig2.update_layout(**CHART_LAYOUT, title=dict(text="Demand vs Price", font=dict(size=15, color='white')), height=360)
                st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Data Insights
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    raw = pd.read_csv('data/retail_price_dataset.csv')

    # Stat cards
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    for col, num, lbl in [
        (s1, len(raw), "Total Records"),
        (s2, raw['product_id'].nunique(), "Unique Products"),
        (s3, raw['product_category_name'].nunique(), "Categories"),
        (s4, f"${raw['unit_price'].mean():.2f}", "Avg Unit Price"),
    ]:
        col.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    cl, cr = st.columns(2, gap="medium")
    with cl:
        fig = px.histogram(raw, x='unit_price', nbins=40, title='Unit Price Distribution',
                           color_discrete_sequence=['#63b3ed'])
        fig.update_layout(**CHART_LAYOUT, height=320)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(raw, x='unit_price', y='qty', color='product_category_name',
                         title='Price vs Demand by Category', opacity=0.65,
                         labels={'unit_price':'Unit Price ($)','qty':'Qty Sold'})
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(**CHART_LAYOUT, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        fig = px.histogram(raw, x='qty', nbins=40, title='Demand Distribution',
                           color_discrete_sequence=['#a78bfa'])
        fig.update_layout(**CHART_LAYOUT, height=320)
        st.plotly_chart(fig, use_container_width=True)

        monthly = raw.groupby('month')['qty'].mean().reset_index()
        monthly['month_name'] = monthly['month'].apply(lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        fig = px.bar(monthly, x='month_name', y='qty', title='Avg Demand by Month',
                     color='qty', color_continuous_scale='Blues',
                     labels={'month_name':'Month','qty':'Avg Qty Sold'})
        fig.update_layout(**CHART_LAYOUT, height=340, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Category breakdown
    cat_avg = raw.groupby('product_category_name').agg(
        avg_price=('unit_price','mean'), avg_qty=('qty','mean')
    ).reset_index().sort_values('avg_price', ascending=True)
    fig = px.bar(cat_avg, x='avg_price', y='product_category_name', orientation='h',
                 title='Avg Price by Category', color='avg_qty',
                 color_continuous_scale='Viridis',
                 labels={'product_category_name':'','avg_price':'Avg Price ($)','avg_qty':'Avg Qty'})
    fig.update_layout(**CHART_LAYOUT, height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Competitor gap
    raw['avg_comp'] = raw[['comp_1','comp_2','comp_3']].mean(axis=1)
    raw['price_diff'] = raw['unit_price'] - raw['avg_comp']
    fig = px.scatter(raw, x='price_diff', y='qty',
                     title='Competitor Price Gap vs Demand',
                     color='product_category_name', opacity=0.65,
                     labels={'price_diff':'Our Price − Avg Competitor ($)','qty':'Qty Sold'})
    fig.add_vline(x=0, line_dash='dash', line_color='rgba(255,255,255,0.3)',
                  annotation_text="Same as competitor", annotation_font_color='rgba(255,255,255,0.5)')
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(**CHART_LAYOUT, height=380)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    best_name = model_results['best_model']
    results   = model_results['results']
    perf_df   = pd.DataFrame(results).T.reset_index().rename(columns={'index':'Model'})

    # Model cards
    cols = st.columns(len(perf_df), gap="medium")
    for i, (_, row) in enumerate(perf_df.iterrows()):
        is_best = row['Model'] == best_name
        border  = "rgba(52,211,153,0.4)" if is_best else "rgba(255,255,255,0.08)"
        bg      = "rgba(16,85,60,0.25)" if is_best else "rgba(255,255,255,0.03)"
        badge   = " ⭐" if is_best else ""
        cols[i].markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:14px;padding:1.2rem;text-align:center;">
          <div style="font-size:0.8rem;font-weight:700;color:rgba(255,255,255,0.5);letter-spacing:1px;text-transform:uppercase;margin-bottom:0.8rem">{row['Model']}{badge}</div>
          <div style="font-size:1.6rem;font-weight:800;color:#63b3ed">{row['R2']:.3f}</div>
          <div style="font-size:0.7rem;color:rgba(255,255,255,0.4);margin-bottom:0.6rem">R² Score</div>
          <div style="font-size:0.85rem;color:rgba(255,255,255,0.6)">MAE: <strong>{row['MAE']:.3f}</strong></div>
          <div style="font-size:0.85rem;color:rgba(255,255,255,0.6)">RMSE: <strong>{row['RMSE']:.3f}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Charts
    colors = ['#34d399' if m == best_name else '#63b3ed' for m in perf_df['Model']]
    mc1, mc2, mc3 = st.columns(3, gap="medium")

    with mc1:
        fig = go.Figure(go.Bar(x=perf_df['Model'], y=perf_df['R2'], marker_color=colors,
                               text=perf_df['R2'].round(3), textposition='outside'))
        fig.update_layout(**CHART_LAYOUT, title=dict(text="R² Score ↑", font=dict(size=14,color='white')), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with mc2:
        fig = go.Figure(go.Bar(x=perf_df['Model'], y=perf_df['MAE'], marker_color='#a78bfa',
                               text=perf_df['MAE'].round(3), textposition='outside'))
        fig.update_layout(**CHART_LAYOUT, title=dict(text="MAE ↓ (lower is better)", font=dict(size=14,color='white')), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with mc3:
        fig = go.Figure(go.Bar(x=perf_df['Model'], y=perf_df['RMSE'], marker_color='#f6ad55',
                               text=perf_df['RMSE'].round(3), textposition='outside'))
        fig.update_layout(**CHART_LAYOUT, title=dict(text="RMSE ↓ (lower is better)", font=dict(size=14,color='white')), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="section-title">⚙️ How It Works</div>', unsafe_allow_html=True)
    hw1, hw2, hw3, hw4, hw5 = st.columns(5, gap="medium")
    for col, icon, title, desc in [
        (hw1, "📂", "Data", "676 retail records with price, demand, competitor prices & seasonality"),
        (hw2, "🔧", "Features", "22 engineered features — competitor gap, price-score interaction, time signals"),
        (hw3, "🤖", "Training", "4 ML models trained; best selected by R² on 20% held-out test set"),
        (hw4, "🎯", "Optimization", "scipy minimize_scalar finds price maximizing price × predicted demand"),
        (hw5, "📈", "Output", "Optimal price, revenue curve, demand curve & comparison with current price"),
    ]:
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1.1rem;text-align:center;height:100%">
          <div style="font-size:1.8rem;margin-bottom:0.5rem">{icon}</div>
          <div style="font-size:0.8rem;font-weight:700;color:#63b3ed;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.8px">{title}</div>
          <div style="font-size:0.78rem;color:rgba(255,255,255,0.5);line-height:1.5">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
