# ============================================================
# PAGE: REGRESSION  📈
# ============================================================
# User inputs → Sales amount predict பண்றோம்
# 3 models compare பண்றோம்
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.insert(0, 'src')
from model_preparation import load_data, load_all_models, set_page_style, FEATURE_COLS

st.set_page_config(page_title="Regression", page_icon="📈", layout="wide")
set_page_style()

df     = load_data()
models = load_all_models()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 📈 Regression")
    st.markdown("Sales amount predict பண்றோம்")
    st.markdown("---")
    model_choice = st.selectbox(
        "Choose Model",
        ["XGBoost ⭐", "Random Forest", "Linear Regression"]
    )

# ── Header ──
st.markdown("""
<h1 style='color:#a78bfa'>📈 Sales Regression Predictor</h1>
<p style='color:#888'>Input values கொடுங்க → ML model Sales amount predict பண்ணும்</p>
<hr style='border-color:#333'>
""", unsafe_allow_html=True)

# ============================================================
# INPUT FORM
# ============================================================
st.markdown("### 🎛️ Enter Sale Details")

c1, c2, c3, c4 = st.columns(4)

with c1:
    month        = st.selectbox("📅 Month", list(range(1,13)),
                                format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    day_of_week  = st.selectbox("📆 Day",
                                [0,1,2,3,4,5,6],
                                format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    quarter      = st.selectbox("🗓️ Quarter", [1,2,3,4])

with c2:
    is_weekend   = st.selectbox("🏖️ Weekend?",  [0,1], format_func=lambda x: "Yes" if x else "No")
    is_holiday   = st.selectbox("🎉 Holiday?",  [0,1], format_func=lambda x: "Yes" if x else "No")
    season_enc   = st.selectbox("🌿 Season",
                                [0,1,2,3],
                                format_func=lambda x: ['Autumn','Monsoon','Summer','Winter'][x])

with c3:
    customer_count = st.slider("👥 Customer Count", 1, 100, 30)
    has_promotion  = st.selectbox("🎯 Promotion?", [0,1], format_func=lambda x: "Yes" if x else "No")
    max_discount   = st.slider("💸 Max Discount %", 0, 50, 10)

with c4:
    store_age      = st.slider("🏪 Store Age (years)", 1, 20, 5)
    total_stock    = st.slider("📦 Total Stock", 0, 1000, 200)
    profit_margin  = st.slider("💹 Profit Margin %", 0.0, 100.0, 45.0)

c5, c6, c7, c8 = st.columns(4)
with c5:
    category_enc = st.selectbox("🏷️ Category",  [0,1,2,3,4],
                                format_func=lambda x: ['Electronics','Food','Sports','Clothing','Home'][x])
with c6:
    brand_enc    = st.selectbox("🔖 Brand", [0,1,2,3,4,5,6],
                                format_func=lambda x: f"Brand {x+1}")
with c7:
    store_type_enc = st.selectbox("🏬 Store Type", [0,1,2,3],
                                  format_func=lambda x: ['Mall','Supermarket','Outlet','Mini'][x])
with c8:
    region_enc   = st.selectbox("🗺️ Region", [0,1],
                                format_func=lambda x: ['North','South'][x])

# ============================================================
# PREDICT BUTTON
# ============================================================
st.markdown("---")
predict_btn = st.button("🔮 Predict Sales Amount", use_container_width=True)

if predict_btn:
    # Build feature vector
    avg_discount         = max_discount * 0.8
    stock_available      = 1 if total_stock > 0 else 0
    revenue_per_customer = 3500.0   # avg estimate
    discount_sales_ratio = max_discount / 5000.0
    price_range_enc      = 2        # Mid-range default
    payment_enc          = 1        # Card default

    input_data = pd.DataFrame([{
        'Month'                 : month,
        'DayOfWeek'             : day_of_week,
        'Quarter'               : quarter,
        'Is_Weekend'            : is_weekend,
        'Is_Holiday'            : is_holiday,
        'Season_Encoded'        : season_enc,
        'Category_Encoded'      : category_enc,
        'Brand_Encoded'         : brand_enc,
        'Store_Type_Encoded'    : store_type_enc,
        'Region_Encoded'        : region_enc,
        'Price_Range_Encoded'   : price_range_enc,
        'Payment_Method_Encoded': payment_enc,
        'Has_Promotion'         : has_promotion,
        'Max_Discount'          : max_discount,
        'Store_Age_Years'       : store_age,
        'Total_Stock'           : total_stock,
        'Customer_Count'        : customer_count,
        'Profit_Margin'         : profit_margin,
        'Revenue_Per_Customer'  : revenue_per_customer,
        'Avg_Discount'          : avg_discount,
        'Stock_Available'       : stock_available,
        'Discount_Sales_Ratio'  : discount_sales_ratio,
    }])

    scaler    = models['scaler']
    input_sc  = scaler.transform(input_data)

    # Predict with all 3 models
    lr_pred  = models['linear_reg'].predict(input_sc)[0]
    rf_pred  = models['rf_reg'].predict(input_data)[0]
    xgb_pred = models['xgb_reg'].predict(input_data)[0]

    # Show predictions
    st.markdown("### 🎯 Predicted Sales")
    r1, r2, r3 = st.columns(3)

    with r1:
        color = "#60a5fa"
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{color}22,{color}11);
        border-left:4px solid {color};border-radius:12px;padding:20px;text-align:center'>
        <p style='color:#888;margin:0'>Linear Regression</p>
        <p style='color:{color};font-size:2.2rem;font-weight:700;margin:8px 0'>
            ₹{lr_pred:,.0f}
        </p>
        <p style='color:#555;font-size:12px'>R² = 0.46</p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        color = "#34d399"
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{color}22,{color}11);
        border-left:4px solid {color};border-radius:12px;padding:20px;text-align:center'>
        <p style='color:#888;margin:0'>Random Forest</p>
        <p style='color:{color};font-size:2.2rem;font-weight:700;margin:8px 0'>
            ₹{rf_pred:,.0f}
        </p>
        <p style='color:#555;font-size:12px'>R² = 0.97</p>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        color = "#a78bfa"
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{color}22,{color}11);
        border-left:4px solid {color};border-radius:12px;padding:20px;text-align:center'>
        <p style='color:#888;margin:0'>XGBoost ⭐ Best</p>
        <p style='color:{color};font-size:2.2rem;font-weight:700;margin:8px 0'>
            ₹{xgb_pred:,.0f}
        </p>
        <p style='color:#555;font-size:12px'>R² = 0.9988</p>
        </div>
        """, unsafe_allow_html=True)

    # Comparison chart
    st.markdown("---")
    st.markdown("#### 📊 Model Comparison")
    comp_df = pd.DataFrame({
        'Model'     : ['Linear Reg', 'Random Forest', 'XGBoost'],
        'Prediction': [lr_pred, rf_pred, xgb_pred]
    })
    fig = px.bar(
        comp_df, x='Model', y='Prediction',
        color='Model',
        color_discrete_sequence=['#60a5fa','#34d399','#a78bfa'],
        template='plotly_dark', text='Prediction'
    )
    fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
    fig.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', showlegend=False,
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("#### 🔍 What affects Sales most? (XGBoost)")
    importance = dict(zip(FEATURE_COLS, models['xgb_reg'].feature_importances_))
    imp_df = pd.DataFrame(list(importance.items()),
                          columns=['Feature','Importance']
                          ).sort_values('Importance', ascending=True).tail(10)
    fig2 = px.bar(
        imp_df, x='Importance', y='Feature',
        orientation='h', color='Importance',
        color_continuous_scale='Purples',
        template='plotly_dark'
    )
    fig2.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👆 Values enter பண்ணி **Predict Sales Amount** button click பண்ணுங்க!")

    # Show historical distribution
    st.markdown("#### 📊 Historical Sales Distribution")
    fig = px.histogram(
        df.sample(5000), x='Sales', nbins=40,
        color_discrete_sequence=['#a78bfa'],
        template='plotly_dark'
    )
    fig.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
