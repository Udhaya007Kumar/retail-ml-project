# ============================================================
# PAGE: CLUSTERING  🔵
# ============================================================
# Store groups காட்றோம்
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.insert(0, 'src')
from model_preparation import load_data, load_cluster_data, load_all_models, set_page_style

st.set_page_config(page_title="Clustering", page_icon="🔵", layout="wide")
set_page_style()

df         = load_data()
cluster_df = load_cluster_data()
models     = load_all_models()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🔵 Clustering")
    st.markdown("KMeans — Stores grouping")
    st.markdown("---")
    st.markdown("**3 Groups:**")
    st.markdown("- 🔴 Group 0")
    st.markdown("- 🟡 Group 1")
    st.markdown("- 🟢 Group 2")

# ── Header ──
st.markdown("""
<h1 style='color:#60a5fa'>🔵 Store Clustering Analysis</h1>
<p style='color:#888'>KMeans algorithm stores-ஐ similar groups-ஆ பிரிக்குது</p>
<hr style='border-color:#333'>
""", unsafe_allow_html=True)

# ── Cluster labels ──
CLUSTER_COLOR = {0: '#ef4444', 1: '#f59e0b', 2: '#22c55e'}
CLUSTER_NAME  = {0: '🔴 Group 0 — Low', 1: '🟡 Group 1 — Mid', 2: '🟢 Group 2 — High'}

cluster_df['Cluster_Name']  = cluster_df['Cluster'].map(CLUSTER_NAME)
cluster_df['Cluster_Color'] = cluster_df['Cluster'].map(CLUSTER_COLOR)

# ── KPIs ──
st.markdown("### 📊 Cluster Overview")
c0 = cluster_df[cluster_df['Cluster']==0]
c1 = cluster_df[cluster_df['Cluster']==1]
c2 = cluster_df[cluster_df['Cluster']==2]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Stores",      f"{len(cluster_df)}")
k2.metric("🔴 Group 0 Stores", f"{len(c0)}")
k3.metric("🟡 Group 1 Stores", f"{len(c1)}")
k4.metric("🟢 Group 2 Stores", f"{len(c2)}")

st.markdown("---")

# ============================================================
# CHARTS
# ============================================================
col1, col2 = st.columns(2)

# ── Scatter: Total Sales vs Transactions ──
with col1:
    st.markdown("#### 🔵 Cluster Scatter Plot")
    st.caption("Total Sales vs Transactions — color = cluster")
    fig1 = px.scatter(
        cluster_df,
        x='Transactions', y='Total_Sales',
        color='Cluster_Name',
        size='Avg_Sales',
        color_discrete_map={v:CLUSTER_COLOR[k] for k,v in CLUSTER_NAME.items()},
        template='plotly_dark',
        hover_data=['Store_ID','Avg_Sales'],
        labels={'Total_Sales':'Total Sales (₹)', 'Transactions':'Transaction Count'}
    )
    fig1.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── Bar: Avg Sales per Cluster ──
with col2:
    st.markdown("#### 📊 Avg Sales by Cluster")
    agg = cluster_df.groupby('Cluster_Name').agg(
        Avg_Sales=('Avg_Sales','mean'),
        Stores=('Store_ID','count')
    ).reset_index()
    fig2 = px.bar(
        agg, x='Cluster_Name', y='Avg_Sales',
        color='Cluster_Name',
        color_discrete_map={v:CLUSTER_COLOR[k] for k,v in CLUSTER_NAME.items()},
        template='plotly_dark', text='Avg_Sales'
    )
    fig2.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
    fig2.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', showlegend=False,
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

# ── Pie: Cluster distribution ──
with col3:
    st.markdown("#### 🥧 Store Count by Cluster")
    fig3 = px.pie(
        cluster_df, names='Cluster_Name',
        color='Cluster_Name',
        color_discrete_map={v:CLUSTER_COLOR[k] for k,v in CLUSTER_NAME.items()},
        hole=0.4, template='plotly_dark'
    )
    fig3.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Box: Sales distribution per cluster ──
with col4:
    st.markdown("#### 📦 Total Sales Distribution")
    fig4 = px.box(
        cluster_df, x='Cluster_Name', y='Total_Sales',
        color='Cluster_Name',
        color_discrete_map={v:CLUSTER_COLOR[k] for k,v in CLUSTER_NAME.items()},
        template='plotly_dark', points='all'
    )
    fig4.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', showlegend=False,
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Store-level table ──
st.markdown("---")
st.markdown("### 📋 Store-level Cluster Details")

show_cluster = st.selectbox(
    "Filter by Cluster",
    ["All"] + list(CLUSTER_NAME.values())
)

if show_cluster == "All":
    display_df = cluster_df
else:
    display_df = cluster_df[cluster_df['Cluster_Name'] == show_cluster]

display_df_show = display_df[[
    'Store_ID','Cluster_Name','Total_Sales',
    'Avg_Sales','Total_Customers','Transactions'
]].copy()
display_df_show['Total_Sales']     = display_df_show['Total_Sales'].apply(lambda x: f"₹{x:,.0f}")
display_df_show['Avg_Sales']       = display_df_show['Avg_Sales'].apply(lambda x: f"₹{x:,.0f}")
display_df_show['Total_Customers'] = display_df_show['Total_Customers'].apply(lambda x: f"{x:,}")

st.dataframe(display_df_show, use_container_width=True, height=300)

# ── Insights ──
st.markdown("---")
st.markdown("### 💡 Cluster Insights")
i1, i2, i3 = st.columns(3)
with i1:
    st.error(f"""
    **🔴 Group 0 — {len(c0)} Stores**
    Low performing stores.
    Need promotions & support.
    Avg Sales: ₹{c0['Avg_Sales'].mean():,.0f}
    """)
with i2:
    st.warning(f"""
    **🟡 Group 1 — {len(c1)} Stores**
    Average performing stores.
    Can be improved further.
    Avg Sales: ₹{c1['Avg_Sales'].mean():,.0f}
    """)
with i3:
    st.success(f"""
    **🟢 Group 2 — {len(c2)} Stores**
    Top performing stores!
    Best practices here.
    Avg Sales: ₹{c2['Avg_Sales'].mean():,.0f}
    """)
