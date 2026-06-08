# ============================================================
# APP.PY  — MAIN HOME PAGE  🏠
# ============================================================
# streamlit run app.py னு run பண்ணா இந்த page திறக்கும்
# Left sidebar-ல மத்த pages links இருக்கும்
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, 'src')

from model_preparation import load_data, load_all_models, set_page_style

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title  = "Retail Sales ML",
    page_icon   = "🛒",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

set_page_style()

# ── Load Data ────────────────────────────────────────────────
df = load_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🛒 Retail ML Project")
    st.markdown("---")
    st.markdown("### 📋 Pages")
    st.markdown("""
    - 🏠 **Home** ← You are here
    - 📊 **EDA** — Data Analysis
    - ✅ **Validation** — Model Results
    - 📈 **Regression** — Sales Predict
    - 🏷️ **Classification** — High/Low
    - 🔵 **Clustering** — Store Groups
    """)
    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    st.info(f"""
    **Rows:** {len(df):,}
    **Columns:** {df.shape[1]}
    **Period:** {df['Date'].min().date()} → {df['Date'].max().date()}
    """)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<h1 style='text-align:center; color:#a78bfa; font-size:2.8rem'>
    🛒 Retail Sales ML Dashboard
</h1>
<p style='text-align:center; color:#888; font-size:1.1rem'>
    Machine Learning — Regression · Classification · Clustering
</p>
<hr style='border-color:#333'>
""", unsafe_allow_html=True)

# ============================================================
# KPI METRICS ROW
# ============================================================
st.markdown("### 📊 Key Metrics")

total_sales     = df['Sales'].sum()
avg_sales       = df['Sales'].mean()
total_customers = df['Customer_Count'].sum()
total_products  = df['Product_ID'].nunique()
total_stores    = df['Store_ID'].nunique()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("💰 Total Sales",    f"₹{total_sales/1e6:.1f}M")
with c2:
    st.metric("📈 Avg Sale",       f"₹{avg_sales:,.0f}")
with c3:
    st.metric("👥 Total Customers",f"{total_customers/1e6:.1f}M")
with c4:
    st.metric("📦 Products",       f"{total_products}")
with c5:
    st.metric("🏪 Stores",         f"{total_stores}")

st.markdown("---")

# ============================================================
# CHARTS ROW
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📅 Monthly Sales Trend")
    monthly = df.groupby('Month')['Sales'].sum().reset_index()
    monthly['Month_Name'] = ['Jan','Feb','Mar','Apr','May','Jun',
                              'Jul','Aug','Sep','Oct','Nov','Dec'][:len(monthly)]
    fig = px.line(
        monthly, x='Month_Name', y='Sales',
        markers=True,
        color_discrete_sequence=['#a78bfa'],
        template='plotly_dark'
    )
    fig.update_traces(line_width=3, marker_size=8)
    fig.update_layout(
        plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
        font_color='white', margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### 🏷️ Sales by Category")
    cat_sales = df.groupby('Category')['Sales'].sum().reset_index()
    fig2 = px.pie(
        cat_sales, names='Category', values='Sales',
        color_discrete_sequence=px.colors.qualitative.Vivid,
        hole=0.45, template='plotly_dark'
    )
    fig2.update_layout(
        plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
        font_color='white', margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# SECOND ROW
# ============================================================
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 🏪 Top 10 Stores by Sales")
    store_sales = df.groupby('Store_ID')['Sales'].sum().nlargest(10).reset_index()
    fig3 = px.bar(
        store_sales, x='Sales', y='Store_ID',
        orientation='h',
        color='Sales',
        color_continuous_scale='Purples',
        template='plotly_dark'
    )
    fig3.update_layout(
        plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
        font_color='white', margin=dict(l=0,r=0,t=20,b=0),
        yaxis={'categoryorder':'total ascending'}
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown("#### 💳 Payment Method Split")
    pay = df['Payment_Method'].value_counts().reset_index()
    pay.columns = ['Method','Count']
    fig4 = px.bar(
        pay, x='Method', y='Count',
        color='Method',
        color_discrete_sequence=['#a78bfa','#34d399','#60a5fa'],
        template='plotly_dark'
    )
    fig4.update_layout(
        plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
        font_color='white', margin=dict(l=0,r=0,t=20,b=0),
        showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# ML MODEL SUMMARY CARDS
# ============================================================
st.markdown("---")
st.markdown("### 🤖 ML Models Overview")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#7c3aed22,#7c3aed11);
    border-left:4px solid #7c3aed;border-radius:12px;padding:20px'>
    <h3 style='color:#a78bfa;margin:0'>📈 Regression</h3>
    <p style='color:#888;margin:8px 0 4px'>Predict Sales Amount</p>
    <p style='color:white;font-size:1.5rem;font-weight:700;margin:0'>R² = 0.9988</p>
    <p style='color:#34d399;margin:4px 0 0'>XGBoost Best Model ⭐</p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#059669 22,#05966911);
    border-left:4px solid #059669;border-radius:12px;padding:20px'>
    <h3 style='color:#34d399;margin:0'>🏷️ Classification</h3>
    <p style='color:#888;margin:8px 0 4px'>Predict High/Medium/Low</p>
    <p style='color:white;font-size:1.5rem;font-weight:700;margin:0'>99.4% Accuracy</p>
    <p style='color:#34d399;margin:4px 0 0'>XGBoost Best Model ⭐</p>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#2563eb22,#2563eb11);
    border-left:4px solid #2563eb;border-radius:12px;padding:20px'>
    <h3 style='color:#60a5fa;margin:0'>🔵 Clustering</h3>
    <p style='color:#888;margin:8px 0 4px'>Group Similar Stores</p>
    <p style='color:white;font-size:1.5rem;font-weight:700;margin:0'>3 Groups Found</p>
    <p style='color:#34d399;margin:4px 0 0'>KMeans Algorithm ⭐</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#555;font-size:13px'>
    🛒 Retail Sales ML Project · Built with Streamlit + XGBoost + Scikit-learn
</p>
""", unsafe_allow_html=True)
