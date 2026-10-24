from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Big Data Analytics Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# PATHS & DATA LOADERS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# Cluster business names
CLUSTER_NAMES = {
    0: "Lower-Spending Customers",
    1: "Recently Active Customers",
    2: "Repeat, Higher-Spending Customers",
    3: "Least Recently Active Customers"
}

# Cluster color palette for consistent visual identity
CLUSTER_COLORS = {
    0: "#38BDF8",  # Sky Blue
    1: "#10B981",  # Emerald
    2: "#8B5CF6",  # Violet
    3: "#F59E0B"   # Amber
}

# Brazilian state names for clean tooltips & labels
BRAZIL_STATES = {
    "SP": "São Paulo", "RJ": "Rio de Janeiro", "MG": "Minas Gerais", "RS": "Rio Grande do Sul",
    "PR": "Paraná", "SC": "Santa Catarina", "BA": "Bahia", "DF": "Distrito Federal",
    "GO": "Goiás", "ES": "Espírito Santo", "PE": "Pernambuco", "CE": "Ceará",
    "PA": "Pará", "MT": "Mato Grosso", "MA": "Maranhão", "MS": "Mato Grosso do Sul",
    "PB": "Paraíba", "PI": "Piauí", "RN": "Rio Grande do Norte", "AL": "Alagoas",
    "SE": "Sergipe", "TO": "Tocantins", "RO": "Rondônia", "AM": "Amazonas",
    "AC": "Acre", "AP": "Amapá", "RR": "Roraima"
}

@st.cache_data
def load_rfm_parameters():
    model_path = OUTPUT_DIR / "rfm_kmeans_model.json"
    with open(model_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_data(filename: str) -> pd.DataFrame:
    file_path = OUTPUT_DIR / filename
    return pd.read_csv(file_path)

# Load core datasets
monthly_sales = load_data("monthly_sales.csv")
category_sales = load_data("category_sales.csv")
regional_sales = load_data("regional_sales.csv")
payment_analysis = load_data("payment_analysis.csv")
review_analysis = load_data("review_analysis.csv")
profiles = load_data("cluster_profiles.csv")

# Precompute profile metadata
profiles["segment_name"] = profiles["cluster"].map(CLUSTER_NAMES)
total_customers = profiles["customer_count"].sum()
profiles["customer_percentage"] = (profiles["customer_count"] / total_customers * 100).round(2)

# ---------------------------------------------------------
# CUSTOM CSS: MODERN PROFESSIONAL DARK THEME
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Background and Layout */
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
    }

    /* High contrast markdown paragraphs */
    .stMarkdown p {
        color: #E2E8F0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0E1424 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Sidebar Navigation Radio Buttons - High Contrast Fix */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] span {
        color: #E2E8F0 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stRadio"] label:hover p,
    div[data-testid="stRadio"] label:hover span {
        color: #FFFFFF !important;
    }

    div[data-testid="stRadio"] div[aria-checked="true"] p,
    div[data-testid="stRadio"] div[aria-checked="true"] span,
    div[data-testid="stRadio"] input:checked + div p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* All Widget Labels (Inputs, Selectboxes, Sliders) - High Contrast Fix */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span,
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] p,
    .stWidgetLabel,
    .stWidgetLabel p,
    .stWidgetLabel label {
        color: #F8FAFC !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }

    /* Form Inputs - Sleek Dark Number Input & Text Input */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background-color: #161F36 !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 1px #6366F1 !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #334155 !important;
        color: #FFFFFF !important;
    }

    /* Selectbox Styling */
    div[data-baseweb="select"] > div {
        background-color: #161F36 !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }

    div[data-baseweb="popover"] {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    li[role="option"] {
        background-color: #111827 !important;
        color: #E2E8F0 !important;
    }

    li[role="option"]:hover {
        background-color: #1F293D !important;
        color: #FFFFFF !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] * {
        color: #E2E8F0 !important;
    }

    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Primary & Form Submit Buttons - Fixed Visible Gradient & High Contrast */
    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    button[kind="secondaryFormSubmit"],
    button[kind="primaryFormSubmit"],
    button[kind="secondary"],
    button[kind="primary"] {
        background: linear-gradient(90deg, #4F46E5, #6366F1) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: 1px solid rgba(99, 102, 241, 0.5) !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    button[kind="secondaryFormSubmit"]:hover,
    button[kind="primaryFormSubmit"]:hover {
        background: linear-gradient(90deg, #4338CA, #4F46E5) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button p,
    div[data-testid="stFormSubmitButton"] > button p,
    button[kind="secondaryFormSubmit"] p,
    button[kind="primaryFormSubmit"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Captions, Help Tooltip Icons, and Subtitles */
    div[data-testid="stCaptionContainer"] p,
    .stCaption,
    .stCaption p {
        color: #CBD5E1 !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stTooltipIcon"] svg {
        fill: #93C5FD !important;
    }

    /* Custom KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }

    .kpi-card {
        background: linear-gradient(180deg, #161F36 0%, #111827 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #38BDF8);
    }

    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }

    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #CBD5E1;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .kpi-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.18);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
        white-space: nowrap;
    }

    .kpi-subtext {
        font-size: 0.82rem;
        color: #94A3B8;
        line-height: 1.4;
    }

    /* Content Cards */
    .content-card {
        background-color: #111827;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    .insight-card {
        background-color: #131D33;
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .insight-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 4px;
    }

    .insight-desc {
        font-size: 0.84rem;
        color: #CBD5E1;
        line-height: 1.5;
        margin: 0;
    }

    /* Notice callout */
    .notice-box {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 16px 0 24px 0;
        font-size: 0.88rem;
        color: #E2E8F0;
        line-height: 1.55;
    }

    .notice-box strong {
        color: #FFFFFF;
    }

    /* Sidebar Header */
    .sidebar-brand {
        padding: 12px 4px 20px 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    .brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.01em;
        margin-bottom: 4px;
    }

    .brand-tag {
        font-size: 0.72rem;
        color: #A5B4FC;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .brand-subtitle {
        font-size: 0.80rem;
        color: #CBD5E1;
        margin-top: 4px;
    }

    /* Prediction Hero Card */
    .prediction-hero {
        background: linear-gradient(135deg, #1E1B4B 0%, #111827 100%);
        border: 1px solid rgba(129, 140, 248, 0.4);
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 20px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }

    .prediction-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 12px;
        border-radius: 9999px;
        background: #4338CA;
        color: #FFFFFF;
        margin-bottom: 12px;
    }

    .prediction-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 6px;
    }

    .prediction-cluster {
        font-size: 0.92rem;
        color: #C7D2FE;
        margin-bottom: 18px;
    }

    /* Clean Streamlit metric suppression when custom used */
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PLOTLY MODERN DARK THEME HELPER
# ---------------------------------------------------------
def apply_plotly_theme(fig, title: str = None, height: int = 400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=30, r=30, t=55 if title else 25, b=35),
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Inter, sans-serif", size=15, color="#F3F4F6"),
            x=0.01,
            y=0.96
        ) if title else None,
        font=dict(family="Inter, -apple-system, sans-serif", color="#9CA3AF", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#D1D5DB"),
            bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(
            bgcolor="#1F2937",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#F9FAFB",
            bordercolor="rgba(255, 255, 255, 0.15)"
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.06)",
        linecolor="rgba(255, 255, 255, 0.1)",
        tickfont=dict(color="#9CA3AF", size=11)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.06)",
        linecolor="rgba(255, 255, 255, 0.1)",
        tickfont=dict(color="#9CA3AF", size=11)
    )
    return fig

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & PLATFORM METADATA
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-tag">Big Data Platform</div>
        <div class="brand-title">E-Commerce Analytics</div>
        <div class="brand-subtitle">Brazilian Olist Engine • PySpark ML</div>
    </div>
    """, unsafe_allow_html=True)

    navigation_options = [
        "Overview",
        "Sales Analytics",
        "Product Categories",
        "Regional Analysis",
        "Customer Segmentation",
        "Customer Segment Predictor"
    ]

    selected_page = st.radio(
        "Navigation",
        options=navigation_options,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Engine Status & Pipeline Metadata
    st.markdown("""
    <div style="padding: 10px 4px; font-size: 0.76rem; color: #CBD5E1;">
        <div style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #FFFFFF; margin-bottom: 8px;">
            Pipeline Metadata
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span>Engine:</span>
            <span style="color: #E2E8F0; font-weight: 500;">Apache Spark</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span>Dataset:</span>
            <span style="color: #E2E8F0; font-weight: 500;">Olist Brazilian E-Com</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span>Fulfillment:</span>
            <span style="color: #E2E8F0; font-weight: 500;">Delivered Orders</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span>ML Clustering:</span>
            <span style="color: #E2E8F0; font-weight: 500;">K-Means (k=4)</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Silhouette:</span>
            <span style="color: #38BDF8; font-weight: 600;">0.2973</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if selected_page == "Overview":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 2.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
            Executive Overview
        </h1>
        <p style="font-size: 0.92rem; color: #9CA3AF; margin: 0;">
            Macro-level performance indicators, order volumes, revenue distribution, and customer sentiment across Brazilian e-commerce.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Core Calculations
    total_revenue = monthly_sales["item_revenue"].sum()
    total_orders = monthly_sales["order_count"].sum()
    top_category = category_sales.sort_values("item_revenue", ascending=False).iloc[0]
    top_cat_name = top_category["category"].replace("_", " ").title()
    top_cat_rev = top_category["item_revenue"]

    total_review_count = review_analysis["review_count"].sum()
    avg_review = (
        (review_analysis["review_score"] * review_analysis["review_count"]).sum()
        / total_review_count
    )

    # 4 KPI Cards (Responsive grid, strictly no truncation)
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Total Item Revenue</span>
                <span class="kpi-badge">Delivered</span>
            </div>
            <div class="kpi-value">R$ {total_revenue/1_000_000:.2f}M</div>
            <div class="kpi-subtext">Sum of item prices (excl. freight)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Delivered Orders</span>
                <span class="kpi-badge">Completed</span>
            </div>
            <div class="kpi-value">{int(total_orders):,}</div>
            <div class="kpi-subtext">Total verified fulfillments</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Top Category</span>
                <span class="kpi-badge">Rank #1</span>
            </div>
            <div class="kpi-value">{top_cat_name}</div>
            <div class="kpi-subtext">R$ {top_cat_rev/1_000_000:.2f}M item revenue</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Average Review</span>
                <span class="kpi-badge">{int(total_review_count):,} Ratings</span>
            </div>
            <div class="kpi-value">{avg_review:.2f} <span style="font-size: 1.1rem; color: #9CA3AF; font-weight: 500;">/ 5.0</span></div>
            <div class="kpi-subtext">Weighted score across orders</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Project Description & Architectural Context
    st.markdown("""
    <div class="content-card">
        <h3 style="font-size: 1.15rem; font-weight: 600; color: #F3F4F6; margin-bottom: 8px;">
            Platform Architecture & Scope
        </h3>
        <p style="font-size: 0.88rem; color: #9CA3AF; line-height: 1.6; margin-bottom: 12px;">
            This big data analytics engine processes transaction histories from the <strong>Brazilian Olist E-Commerce Dataset</strong> 
            utilizing <strong>Apache Spark</strong> for distributed data aggregation and <strong>Spark MLlib</strong> for unsupervised 
            customer behavioral modeling. The metrics presented reflect validated, delivered customer orders between 2016 and 2018.
        </p>
        <div style="display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.80rem; color: #CBD5E1;">
            <div style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 6px;">
                <span style="color: #6366F1; font-weight: 600;">●</span> Focus: Commercial item revenues (excluding freight and marketplace fees)
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 6px;">
                <span style="color: #10B981; font-weight: 600;">●</span> Data Integrity: Null and undefined payment types isolated
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 6px;">
                <span style="color: #38BDF8; font-weight: 600;">●</span> Clustering: Standardized RFM vectorization with K-Means (k=4)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Business Insights Section
    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 600; color: #FFFFFF; margin-bottom: 14px;'>Key Business Insights</h3>", unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">November 2017 Revenue Surge (Black Friday)</div>
            <p class="insight-desc">
                Monthly revenue peaked at <strong>R$ 987,765</strong> across <strong>7,289 delivered orders</strong> in November 2017, representing the annual retail zenith driven by Black Friday campaigns in Brazil.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card" style="border-left-color: #10B981;">
            <div class="insight-title">Marketplace Category Concentration</div>
            <p class="insight-desc">
                The top 3 categories (<strong>Health & Beauty</strong>, <strong>Watches & Gifts</strong>, and <strong>Bed, Bath & Table</strong>) combined generated <strong>R$ 3.42M</strong>—accounting for more than 25% of all commercial sales.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_i2:
        st.markdown("""
        <div class="insight-card" style="border-left-color: #38BDF8;">
            <div class="insight-title">Geographic Gravity: Southeast Dominance</div>
            <p class="insight-desc">
                The state of <strong>São Paulo (SP)</strong> accounts for <strong>R$ 5.07M (37.5%)</strong> of total national item sales. Together with Rio de Janeiro and Minas Gerais, Southeast Brazil captures over 62% of transaction volume.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card" style="border-left-color: #F59E0B;">
            <div class="insight-title">High Customer Satisfaction & Credit Dominance</div>
            <p class="insight-desc">
                <strong>57.8%</strong> of all reviews are perfect 5-star ratings (average 4.14/5.0). Transactions heavily favor <strong>Credit Cards</strong>, representing <strong>78.3% (R$ 12.54M)</strong> of total payment value.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Payment & Review Analysis (Requirement 6)
    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 600; color: #FFFFFF; margin-bottom: 14px;'>Payment Methods & Customer Sentiment</h3>", unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        valid_payments = payment_analysis[
            (payment_analysis["payment_type"] != "not_defined") &
            (payment_analysis["total_payment_value"] > 0)
        ].copy()
        valid_payments["payment_label"] = valid_payments["payment_type"].str.replace("_", " ").str.title()
        total_pay = valid_payments["total_payment_value"].sum()
        valid_payments["percentage"] = (valid_payments["total_payment_value"] / total_pay * 100).round(1)

        fig_payment = px.pie(
            valid_payments,
            names="payment_label",
            values="total_payment_value",
            hole=0.55,
            color_discrete_sequence=["#6366F1", "#38BDF8", "#10B981", "#F59E0B"]
        )
        fig_payment.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Value: R$ %{value:,.2f}<br>Share: %{percent}<extra></extra>"
        )
        apply_plotly_theme(fig_payment, title="Payment Value by Method", height=360)
        st.plotly_chart(fig_payment, use_container_width=True)

    with col_p2:
        review_data = review_analysis.sort_values("review_score").copy()
        score_colors = ["#EF4444", "#F97316", "#F59E0B", "#84CC16", "#10B981"]

        fig_review = go.Figure()
        fig_review.add_trace(go.Bar(
            x=[f"{int(s)} Star{'s' if s>1 else ''}" for s in review_data["review_score"]],
            y=review_data["review_count"],
            text=[f"{p:.1f}% ({c:,})" for p, c in zip(review_data["percentage"], review_data["review_count"])],
            textposition="auto",
            marker=dict(
                color=score_colors,
                line=dict(color="rgba(255,255,255,0.15)", width=1)
            ),
            hovertemplate="<b>%{x}</b><br>Reviews: %{y:,}<br>Share: %{text}<extra></extra>"
        ))
        apply_plotly_theme(fig_review, title="Distribution of Customer Review Scores", height=360)
        fig_review.update_yaxes(title="Total Reviews")
        st.plotly_chart(fig_review, use_container_width=True)


# =========================================================
# PAGE 2: SALES ANALYTICS
# =========================================================
elif selected_page == "Sales Analytics":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 2.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
            Sales & Revenue Analytics
        </h1>
        <p style="font-size: 0.92rem; color: #9CA3AF; margin: 0;">
            Track historical item revenue trajectories, delivery order volumes, and average ticket sizes across operational months.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="notice-box">
        <strong>Commercial Definition:</strong> Revenue displayed represents the aggregate item prices for completed, delivered orders. 
        It isolates commercial product turnover, does not represent gross margin or net profit, and excludes freight and seller commission fees.
    </div>
    """, unsafe_allow_html=True)

    # Date Range Filter
    monthly_sales_sorted = monthly_sales.sort_values("order_month").copy()
    available_months = sorted(monthly_sales_sorted["order_month"].dropna().unique())

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        selected_range = st.select_slider(
            "Select Time Horizon",
            options=available_months,
            value=(available_months[0], available_months[-1])
        )
    with col_ctrl2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        view_mode = st.radio(
            "Chart Metric",
            options=["Revenue & Orders", "Revenue Only", "Orders Only"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # Filtered dataframe
    filtered_sales = monthly_sales_sorted[
        monthly_sales_sorted["order_month"].between(selected_range[0], selected_range[1])
    ].copy()
    filtered_sales["avg_order_value"] = (
        filtered_sales["item_revenue"] / filtered_sales["order_count"]
    ).round(2)

    # Filtered Summary Metrics
    f_revenue = filtered_sales["item_revenue"].sum()
    f_orders = filtered_sales["order_count"].sum()
    f_aov = f_revenue / f_orders if f_orders > 0 else 0
    peak_row = filtered_sales.sort_values("item_revenue", ascending=False).iloc[0]

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">Filtered Item Revenue</span></div>
            <div class="kpi-value">R$ {f_revenue/1_000_000:.2f}M</div>
            <div class="kpi-subtext">R$ {f_revenue:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">Filtered Orders</span></div>
            <div class="kpi-value">{int(f_orders):,}</div>
            <div class="kpi-subtext">Delivered orders in period</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">Average Order Value</span></div>
            <div class="kpi-value">R$ {f_aov:.2f}</div>
            <div class="kpi-subtext">Item spend per delivered order</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">Peak Month</span></div>
            <div class="kpi-value">{peak_row['order_month']}</div>
            <div class="kpi-subtext">R$ {peak_row['item_revenue']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Interactive Trend Chart
    fig_trend = go.Figure()

    if view_mode in ["Revenue & Orders", "Revenue Only"]:
        fig_trend.add_trace(go.Scatter(
            x=filtered_sales["order_month"],
            y=filtered_sales["item_revenue"],
            name="Item Revenue (BRL)",
            mode="lines+markers",
            line=dict(color="#38BDF8", width=3),
            marker=dict(size=6, color="#38BDF8"),
            fill="tozeroy" if view_mode == "Revenue Only" else None,
            fillcolor="rgba(56, 189, 248, 0.08)" if view_mode == "Revenue Only" else None,
            hovertemplate="<b>%{x}</b><br>Item Revenue: R$ %{y:,.2f}<extra></extra>"
        ))

    if view_mode in ["Revenue & Orders", "Orders Only"]:
        yaxis_target = "y2" if view_mode == "Revenue & Orders" else "y"
        fig_trend.add_trace(go.Bar(
            x=filtered_sales["order_month"],
            y=filtered_sales["order_count"],
            name="Delivered Orders",
            yaxis=yaxis_target,
            opacity=0.35 if view_mode == "Revenue & Orders" else 0.85,
            marker=dict(
                color="#6366F1",
                line=dict(color="rgba(99, 102, 241, 0.8)", width=1)
            ),
            hovertemplate="<b>%{x}</b><br>Delivered Orders: %{y:,}<extra></extra>"
        ))

    layout_updates = dict(
        title="Monthly Item Revenue & Order Volume Dynamics",
        height=450
    )
    apply_plotly_theme(fig_trend, **layout_updates)
    fig_trend.update_xaxes(title="Order Month", tickangle=-45)
    fig_trend.update_yaxes(title="Item Revenue (BRL)", tickprefix="R$ ")

    if view_mode == "Revenue & Orders":
        fig_trend.update_layout(
            yaxis2=dict(
                title=dict(text="Delivered Orders", font=dict(color="#818CF8", size=12)),
                overlaying="y",
                side="right",
                showgrid=False,
                tickfont=dict(color="#818CF8", size=11)
            )
        )

    st.plotly_chart(fig_trend, use_container_width=True)

    # Formatted Data Table Expander
    with st.expander("Explore Monthly Performance Data Table"):
        display_monthly = filtered_sales.copy()
        display_monthly["item_revenue_formatted"] = display_monthly["item_revenue"].apply(lambda x: f"R$ {x:,.2f}")
        display_monthly["avg_order_value_formatted"] = display_monthly["avg_order_value"].apply(lambda x: f"R$ {x:,.2f}")
        display_monthly["order_count_formatted"] = display_monthly["order_count"].apply(lambda x: f"{int(x):,}")

        st.dataframe(
            display_monthly[
                ["order_month", "item_revenue_formatted", "order_count_formatted", "avg_order_value_formatted"]
            ].rename(columns={
                "order_month": "Order Month",
                "item_revenue_formatted": "Item Revenue (BRL)",
                "order_count_formatted": "Delivered Orders",
                "avg_order_value_formatted": "Avg Order Value (AOV)"
            }),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PAGE 3: PRODUCT CATEGORIES
# =========================================================
elif selected_page == "Product Categories":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 2.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
            Product Category Intelligence
        </h1>
        <p style="font-size: 0.92rem; color: #9CA3AF; margin: 0;">
            Comparative analysis of merchandise revenue generation, units sold, and average ticket sizes by product sector.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Clean category dataset
    cat_df = category_sales.copy()
    cat_df["category_clean"] = cat_df["category"].str.replace("_", " ").str.title()
    cat_df["avg_unit_price"] = (cat_df["item_revenue"] / cat_df["items_sold"]).round(2)

    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    with col_c1:
        top_n = st.selectbox("Display Top Categories", options=[10, 15, 20, 30], index=0)
    with col_c2:
        sort_by = st.selectbox("Sort Ranking By", options=["Item Revenue", "Items Sold", "Orders"], index=0)
    with col_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        total_cats = len(cat_df)
        st.caption(f"Catalog contains {total_cats} distinct categories")

    sort_col_map = {
        "Item Revenue": "item_revenue",
        "Items Sold": "items_sold",
        "Orders": "orders"
    }
    metric_col = sort_col_map[sort_by]
    top_categories = cat_df.sort_values(metric_col, ascending=False).head(top_n)

    # Category Summary KPIs
    lead_cat = top_categories.iloc[0]
    total_cat_rev = top_categories["item_revenue"].sum()

    col_ck1, col_ck2, col_ck3 = st.columns(3)
    with col_ck1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">Top Category by {sort_by}</span></div>
            <div class="kpi-value">{lead_cat['category_clean']}</div>
            <div class="kpi-subtext">R$ {lead_cat['item_revenue']:,.2f} ({int(lead_cat['orders']):,} orders)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ck2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">Top {top_n} Revenue Share</span></div>
            <div class="kpi-value">R$ {total_cat_rev/1_000_000:.2f}M</div>
            <div class="kpi-subtext">{total_cat_rev / cat_df['item_revenue'].sum() * 100:.1f}% of catalog revenue</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ck3:
        avg_price_top = (top_categories["item_revenue"].sum() / top_categories["items_sold"].sum())
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">Top {top_n} Avg Item Price</span></div>
            <div class="kpi-value">R$ {avg_price_top:.2f}</div>
            <div class="kpi-subtext">Weighted price per physical item</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Horizontal Bar Chart
    chart_data = top_categories.sort_values(metric_col, ascending=True)

    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        y=chart_data["category_clean"],
        x=chart_data[metric_col],
        orientation="h",
        marker=dict(
            color=chart_data[metric_col],
            colorscale="Viridis",
            line=dict(color="rgba(255,255,255,0.15)", width=1)
        ),
        text=chart_data[metric_col].apply(
            lambda v: f"R$ {v/1_000_000:.2f}M" if metric_col == "item_revenue" else f"{int(v):,}"
        ),
        textposition="auto",
        customdata=np.stack((
            chart_data["item_revenue"],
            chart_data["items_sold"],
            chart_data["orders"],
            chart_data["avg_unit_price"]
        ), axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Item Revenue: R$ %{customdata[0]:,.2f}<br>"
            "Items Sold: %{customdata[1]:,}<br>"
            "Orders: %{customdata[2]:,}<br>"
            "Avg Unit Price: R$ %{customdata[3]:.2f}<extra></extra>"
        )
    ))

    apply_plotly_theme(fig_cat, title=f"Top {top_n} Categories ranked by {sort_by}", height=max(420, top_n * 26))
    fig_cat.update_xaxes(
        title=f"{sort_by} {'(BRL)' if metric_col == 'item_revenue' else 'Count'}",
        tickprefix="R$ " if metric_col == "item_revenue" else None
    )
    fig_cat.update_yaxes(title="Category")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Full Category Table
    with st.expander("View Full Product Category Registry"):
        table_df = cat_df.sort_values("item_revenue", ascending=False).copy()
        table_df["item_revenue_str"] = table_df["item_revenue"].apply(lambda x: f"R$ {x:,.2f}")
        table_df["avg_unit_price_str"] = table_df["avg_unit_price"].apply(lambda x: f"R$ {x:,.2f}")
        table_df["items_sold_str"] = table_df["items_sold"].apply(lambda x: f"{int(x):,}")
        table_df["orders_str"] = table_df["orders"].apply(lambda x: f"{int(x):,}")

        st.dataframe(
            table_df[
                ["category_clean", "item_revenue_str", "items_sold_str", "orders_str", "avg_unit_price_str"]
            ].rename(columns={
                "category_clean": "Category Name",
                "item_revenue_str": "Total Item Revenue",
                "items_sold_str": "Units Sold",
                "orders_str": "Order Count",
                "avg_unit_price_str": "Avg Unit Price"
            }),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PAGE 4: REGIONAL ANALYSIS
# =========================================================
elif selected_page == "Regional Analysis":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 2.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
            Geographic & Regional Analysis
        </h1>
        <p style="font-size: 0.92rem; color: #9CA3AF; margin: 0;">
            Spatial density of commercial revenue, order distribution, and ticket size across Brazil's 27 federative units.
        </p>
    </div>
    """, unsafe_allow_html=True)

    reg_df = regional_sales.copy()
    reg_df["state_name"] = reg_df["customer_state"].map(BRAZIL_STATES).fillna(reg_df["customer_state"])
    reg_df["state_display"] = reg_df["customer_state"] + " - " + reg_df["state_name"]
    reg_df["avg_order_value"] = (reg_df["item_revenue"] / reg_df["orders"]).round(2)
    reg_df = reg_df.sort_values("item_revenue", ascending=False)

    total_nat_rev = reg_df["item_revenue"].sum()
    sp_row = reg_df[reg_df["customer_state"] == "SP"].iloc[0]
    southeast_states = ["SP", "RJ", "MG", "ES"]
    southeast_rev = reg_df[reg_df["customer_state"].isin(southeast_states)]["item_revenue"].sum()

    col_rk1, col_rk2, col_rk3 = st.columns(3)
    with col_rk1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">Leading State (São Paulo)</span></div>
            <div class="kpi-value">R$ {sp_row['item_revenue']/1_000_000:.2f}M</div>
            <div class="kpi-subtext">{sp_row['item_revenue']/total_nat_rev*100:.1f}% of national item revenue</div>
        </div>
        """, unsafe_allow_html=True)
    with col_rk2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">Southeast Region Share</span></div>
            <div class="kpi-value">R$ {southeast_rev/1_000_000:.2f}M</div>
            <div class="kpi-subtext">{southeast_rev/total_nat_rev*100:.1f}% share (SP, RJ, MG, ES)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_rk3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-label">National Coverage</span></div>
            <div class="kpi-value">{len(reg_df)} States</div>
            <div class="kpi-subtext">All 26 states + Federal District (DF)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Top 10 States Bar Chart
    top_10_states = reg_df.head(10).copy()

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(
        x=top_10_states["customer_state"],
        y=top_10_states["item_revenue"],
        marker=dict(
            color=top_10_states["item_revenue"],
            colorscale="Blues",
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        ),
        text=top_10_states["item_revenue"].apply(lambda v: f"R$ {v/1_000_000:.2f}M"),
        textposition="outside",
        customdata=np.stack((
            top_10_states["state_name"],
            top_10_states["orders"],
            top_10_states["items_sold"],
            top_10_states["avg_order_value"]
        ), axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]} (%{x})</b><br>"
            "Item Revenue: R$ %{y:,.2f}<br>"
            "Delivered Orders: %{customdata[1]:,}<br>"
            "Items Sold: %{customdata[2]:,}<br>"
            "Avg Spend / Order: R$ %{customdata[3]:.2f}<extra></extra>"
        )
    ))

    apply_plotly_theme(fig_reg, title="Top 10 Brazilian States by Commercial Item Revenue", height=430)
    fig_reg.update_xaxes(title="State Code")
    fig_reg.update_yaxes(title="Item Revenue (BRL)", tickprefix="R$ ")
    st.plotly_chart(fig_reg, use_container_width=True)

    # Full State Performance Registry
    with st.expander("View Complete State-wise Revenue Registry"):
        reg_table = reg_df.copy()
        reg_table["item_revenue_str"] = reg_table["item_revenue"].apply(lambda x: f"R$ {x:,.2f}")
        reg_table["avg_order_value_str"] = reg_table["avg_order_value"].apply(lambda x: f"R$ {x:,.2f}")
        reg_table["orders_str"] = reg_table["orders"].apply(lambda x: f"{int(x):,}")
        reg_table["items_sold_str"] = reg_table["items_sold"].apply(lambda x: f"{int(x):,}")
        reg_table["share_pct"] = (reg_table["item_revenue"] / total_nat_rev * 100).apply(lambda x: f"{x:.2f}%")

        st.dataframe(
            reg_table[
                ["customer_state", "state_name", "item_revenue_str", "share_pct", "orders_str", "items_sold_str", "avg_order_value_str"]
            ].rename(columns={
                "customer_state": "State",
                "state_name": "State Name",
                "item_revenue_str": "Item Revenue",
                "share_pct": "National Share",
                "orders_str": "Delivered Orders",
                "items_sold_str": "Items Sold",
                "avg_order_value_str": "Avg Order Spend"
            }),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PAGE 5: CUSTOMER SEGMENTATION
# =========================================================
elif selected_page == "Customer Segmentation":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 2.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
            Customer Segmentation (RFM + K-Means)
        </h1>
        <p style="font-size: 0.92rem; color: #9CA3AF; margin: 0;">
            Unsupervised behavioral clustering using Apache Spark MLlib on Recency, Frequency, and Monetary metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Silhouette Score Technical Validation Callout
    st.markdown("""
    <div class="content-card" style="border-left: 4px solid #38BDF8;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF;">
                Model Validation & Silhouette Diagnostics
            </span>
            <span style="font-size: 0.82rem; font-weight: 700; color: #38BDF8; background: rgba(56, 189, 248, 0.15); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.3);">
                Silhouette Score: 0.2973
            </span>
        </div>
        <p style="font-size: 0.85rem; color: #9CA3AF; line-height: 1.55; margin: 0;">
            The model achieves a <strong>silhouette score of 0.2973</strong>, which indicates <strong>modest cluster separation</strong>. 
            In real-world retail e-commerce, customer behavior exhibits massive variance with extreme skew toward one-time buyers 
            (over 96% of customers have Frequency = 1). While cluster boundaries are inherently diffuse rather than sharply isolated, 
            the 4 resulting segments offer distinct, actionable business cohorts for targeted marketing, retention, and reactivation strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Segment Cards
    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 600; color: #FFFFFF; margin-bottom: 14px;'>Customer Segment Cohorts</h3>", unsafe_allow_html=True)

    col_s0, col_s1, col_s2, col_s3 = st.columns(4)
    cluster_tags = {
        0: "Low Basket Size",
        1: "Active & Engaged",
        2: "VIP / Repeat Buyers",
        3: "At-Risk / Inactive"
    }

    cols = [col_s0, col_s1, col_s2, col_s3]
    for idx, row in profiles.sort_values("cluster").iterrows():
        c_id = int(row["cluster"])
        c_name = row["segment_name"]
        c_count = int(row["customer_count"])
        c_pct = row["customer_percentage"]
        c_color = CLUSTER_COLORS.get(c_id, "#6366F1")
        c_tag = cluster_tags.get(c_id, "Segment")

        with cols[c_id]:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {c_color};">
                <div class="kpi-header">
                    <span class="kpi-label">Cluster {c_id}</span>
                    <span class="kpi-badge" style="background: rgba(255,255,255,0.08); color: {c_color}; border-color: {c_color}44;">
                        {c_tag}
                    </span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px; line-height: 1.3;">
                    {c_name}
                </div>
                <div class="kpi-value" style="font-size: 1.55rem;">{c_count:,}</div>
                <div class="kpi-subtext"><strong>{c_pct:.1f}%</strong> of total customer base</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Visualizations: Customer Count Distribution & RFM Profiles
    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 600; color: #FFFFFF; margin-bottom: 14px;'>Segment Distributions & Behavioral Metrics</h3>", unsafe_allow_html=True)

    # 1. Customer Count Distribution Chart
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=[f"Cluster {int(c)}: {CLUSTER_NAMES[int(c)]}" for c in profiles["cluster"]],
        y=profiles["customer_count"],
        text=[f"{int(count):,} ({pct:.1f}%)" for count, pct in zip(profiles["customer_count"], profiles["customer_percentage"])],
        textposition="auto",
        marker=dict(
            color=[CLUSTER_COLORS.get(int(c), "#6366F1") for c in profiles["cluster"]],
            line=dict(color="rgba(255,255,255,0.15)", width=1)
        ),
        hovertemplate="<b>%{x}</b><br>Customer Count: %{y:,}<br>Share: %{text}<extra></extra>"
    ))
    apply_plotly_theme(fig_dist, title="Customer Volume by Cluster Cohort", height=350)
    fig_dist.update_yaxes(title="Customer Count")
    st.plotly_chart(fig_dist, use_container_width=True)

    # 2. RFM Profile Comparison (Separated into 3 individual legible metrics so Frequency is never squashed)
    st.markdown("<h4 style='font-size: 1.05rem; font-weight: 600; color: #E5E7EB; margin-bottom: 8px;'>RFM Behavioral Metric Comparisons</h4>", unsafe_allow_html=True)
    
    col_rfm1, col_rfm2, col_rfm3 = st.columns(3)

    with col_rfm1:
        fig_rec = go.Figure()
        fig_rec.add_trace(go.Bar(
            x=[f"C{int(c)}" for c in profiles["cluster"]],
            y=profiles["avg_recency_days"],
            text=profiles["avg_recency_days"].apply(lambda v: f"{v:.0f} d"),
            textposition="auto",
            marker=dict(color=[CLUSTER_COLORS.get(int(c), "#6366F1") for c in profiles["cluster"]]),
            hovertemplate="<b>Cluster %{x}</b><br>Avg Recency: %{y:.1f} days<extra></extra>"
        ))
        apply_plotly_theme(fig_rec, title="Avg Recency (Days since purchase)", height=280)
        fig_rec.update_yaxes(title="Days")
        st.plotly_chart(fig_rec, use_container_width=True)

    with col_rfm2:
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Bar(
            x=[f"C{int(c)}" for c in profiles["cluster"]],
            y=profiles["avg_frequency"],
            text=profiles["avg_frequency"].apply(lambda v: f"{v:.2f}"),
            textposition="auto",
            marker=dict(color=[CLUSTER_COLORS.get(int(c), "#6366F1") for c in profiles["cluster"]]),
            hovertemplate="<b>Cluster %{x}</b><br>Avg Orders: %{y:.2f}<extra></extra>"
        ))
        apply_plotly_theme(fig_freq, title="Avg Frequency (Delivered orders)", height=280)
        fig_freq.update_yaxes(title="Orders")
        st.plotly_chart(fig_freq, use_container_width=True)

    with col_rfm3:
        fig_mon = go.Figure()
        fig_mon.add_trace(go.Bar(
            x=[f"C{int(c)}" for c in profiles["cluster"]],
            y=profiles["avg_monetary"],
            text=profiles["avg_monetary"].apply(lambda v: f"R$ {v:.0f}"),
            textposition="auto",
            marker=dict(color=[CLUSTER_COLORS.get(int(c), "#6366F1") for c in profiles["cluster"]]),
            hovertemplate="<b>Cluster %{x}</b><br>Avg Item Spend: R$ %{y:.2f}<extra></extra>"
        ))
        apply_plotly_theme(fig_mon, title="Avg Monetary (Item spend in BRL)", height=280)
        fig_mon.update_yaxes(title="R$ (BRL)")
        st.plotly_chart(fig_mon, use_container_width=True)

    # Cluster Summary Table
    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 600; color: #FFFFFF; margin-bottom: 12px;'>Cluster Summary Registry</h3>", unsafe_allow_html=True)
    
    summary_display = profiles[[
        "cluster", "segment_name", "customer_count", "customer_percentage",
        "avg_recency_days", "avg_frequency", "avg_monetary"
    ]].copy()

    summary_display["cluster_label"] = summary_display["cluster"].apply(lambda x: f"Cluster {int(x)}")
    summary_display["customer_count_str"] = summary_display["customer_count"].apply(lambda x: f"{int(x):,}")
    summary_display["customer_percentage_str"] = summary_display["customer_percentage"].apply(lambda x: f"{x:.2f}%")
    summary_display["avg_recency_str"] = summary_display["avg_recency_days"].apply(lambda x: f"{x:.1f} days")
    summary_display["avg_frequency_str"] = summary_display["avg_frequency"].apply(lambda x: f"{x:.2f} orders")
    summary_display["avg_monetary_str"] = summary_display["avg_monetary"].apply(lambda x: f"R$ {x:,.2f}")

    st.dataframe(
        summary_display[[
            "cluster_label", "segment_name", "customer_count_str", "customer_percentage_str",
            "avg_recency_str", "avg_frequency_str", "avg_monetary_str"
        ]].rename(columns={
            "cluster_label": "Cluster ID",
            "segment_name": "Customer Segment",
            "customer_count_str": "Customer Count",
            "customer_percentage_str": "Share (%)",
            "avg_recency_str": "Avg. Recency",
            "avg_frequency_str": "Avg. Frequency",
            "avg_monetary_str": "Avg. Monetary Spend"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Deep-Dive Individual Customer Explorer
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 600; color: #FFFFFF; margin-bottom: 12px;'>Explore Segment Customer Records</h3>", unsafe_allow_html=True)

    selected_cluster = st.selectbox(
        "Select a customer segment to inspect",
        options=sorted(CLUSTER_NAMES.keys()),
        format_func=lambda c: f"Cluster {c}: {CLUSTER_NAMES[c]} ({profiles.loc[profiles['cluster']==c, 'customer_count'].values[0]:,} customers)"
    )

    segments_df = load_data("customer_segments.csv")
    cluster_customers = segments_df[segments_df["cluster"] == selected_cluster].sort_values("Monetary", ascending=False)

    col_exp1, col_exp2 = st.columns([1, 2])
    with col_exp1:
        target_profile = profiles[profiles["cluster"] == selected_cluster].iloc[0]
        st.markdown(f"""
        <div class="content-card">
            <div style="font-size: 0.80rem; font-weight: 600; color: {CLUSTER_COLORS.get(selected_cluster, '#6366F1')}; text-transform: uppercase; margin-bottom: 4px;">
                Cluster {selected_cluster} Profile
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 10px;">
                {target_profile['segment_name']}
            </div>
            <div style="font-size: 0.85rem; color: #9CA3AF; margin-bottom: 12px;">
                Total Customers: <strong style="color: #FFFFFF;">{int(target_profile['customer_count']):,}</strong> ({target_profile['customer_percentage']}%)
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
                <div>• <strong>Recency:</strong> ~{target_profile['avg_recency_days']:.1f} days ago</div>
                <div>• <strong>Frequency:</strong> ~{target_profile['avg_frequency']:.2f} delivered orders</div>
                <div>• <strong>Item Spend:</strong> R$ {target_profile['avg_monetary']:.2f} avg</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_exp2:
        st.markdown(f"<div style='font-size: 0.85rem; color: #9CA3AF; margin-bottom: 8px;'>Top 20 highest-spending customer records in <strong>{CLUSTER_NAMES[selected_cluster]}</strong>:</div>", unsafe_allow_html=True)
        top_cluster_samples = cluster_customers.head(20).copy()
        top_cluster_samples["Recency_str"] = top_cluster_samples["Recency"].apply(lambda x: f"{int(x)} days")
        top_cluster_samples["Frequency_str"] = top_cluster_samples["Frequency"].apply(lambda x: f"{int(x)} orders")
        top_cluster_samples["Monetary_str"] = top_cluster_samples["Monetary"].apply(lambda x: f"R$ {x:,.2f}")

        st.dataframe(
            top_cluster_samples[
                ["customer_unique_id", "Recency_str", "Frequency_str", "Monetary_str"]
            ].rename(columns={
                "customer_unique_id": "Customer Unique ID",
                "Recency_str": "Recency (Days)",
                "Frequency_str": "Order Frequency",
                "Monetary_str": "Total Item Spend"
            }),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PAGE 6: CUSTOMER SEGMENT PREDICTOR
# =========================================================
elif selected_page == "Customer Segment Predictor":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 2.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
            Customer Segment Predictor
        </h1>
        <p style="font-size: 0.92rem; color: #9CA3AF; margin: 0;">
            Project customer RFM profiles into the trained Apache Spark K-Means feature space to assign the closest behavioral cohort.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Technical Scope & Predictive Boundary Disclaimer
    st.markdown("""
    <div class="notice-box">
        <strong>Methodology Notice:</strong> This predictor computes standardized Euclidean distance between the submitted 
        Recency-Frequency-Monetary (RFM) vector and the trained K-Means cluster centroids. 
        It <strong>assigns a customer to an established behavioral cohort</strong>; it does <strong>not</strong> forecast 
        future purchase dates, predict churn probability, or calculate prospective customer lifetime value (LTV).
    </div>
    """, unsafe_allow_html=True)

    col_form, col_context = st.columns([1, 1])

    with col_form:
        st.markdown("<h3 style='font-size: 1.15rem; font-weight: 600; color: #FFFFFF; margin-bottom: 12px;'>Enter Customer RFM Attributes</h3>", unsafe_allow_html=True)
        
        with st.form("customer_prediction_form"):
            recency_input = st.number_input(
                "Days Since Last Purchase (Recency)",
                min_value=1,
                max_value=714,
                value=30,
                help="Observed training range: 1 to 714 days."
            )

            frequency_input = st.number_input(
                "Number of Delivered Orders (Frequency)",
                min_value=1,
                max_value=15,
                value=1,
                help="Observed training range: 1 to 15 orders."
            )

            monetary_input = st.number_input(
                "Total Item Spending in BRL (Monetary)",
                min_value=0.085,
                max_value=13440.0,
                value=150.0,
                step=10.0,
                help="Observed training range: R$ 0.085 to R$ 13,440.00."
            )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            predict_button = st.form_submit_button("Predict Customer Segment", use_container_width=True)

    with col_context:
        st.markdown("<h3 style='font-size: 1.15rem; font-weight: 600; color: #FFFFFF; margin-bottom: 12px;'>Training Space Boundaries</h3>", unsafe_allow_html=True)
        model_data = load_rfm_parameters()
        means = np.array(model_data["mean"])
        stds = np.array(model_data["std"])

        st.markdown(f"""
        <div class="content-card">
            <div style="font-size: 0.88rem; color: #E2E8F0; font-weight: 600; margin-bottom: 12px;">
                Trained Feature Normalization Parameters (StandardScaler):
            </div>
            <div style="font-size: 0.84rem; color: #E2E8F0; line-height: 1.8;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px;">
                    <span style="color: #CBD5E1;">Recency Mean ± Std:</span>
                    <strong style="color: #FFFFFF;">{means[0]:.2f} ± {stds[0]:.2f} days</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.08); padding: 6px 0;">
                    <span style="color: #CBD5E1;">Frequency Mean ± Std:</span>
                    <strong style="color: #FFFFFF;">{means[1]:.2f} ± {stds[1]:.2f} orders</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding-top: 6px;">
                    <span style="color: #CBD5E1;">Monetary Mean ± Std:</span>
                    <strong style="color: #FFFFFF;">R$ {means[2]:.2f} ± R$ {stds[2]:.2f}</strong>
                </div>
            </div>
            <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 14px; line-height: 1.5; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px;">
                <strong style="color: #CBD5E1;">Distance Metric:</strong> Euclidean distance across standardized features (Z-Score scaling) matching Apache Spark pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Prediction Execution
    if predict_button:
        try:
            # Prepare user's RFM vector
            user_rfm = np.array([[
                float(recency_input),
                float(frequency_input),
                float(monetary_input)
            ]])

            # Apply same standard scaling as PySpark ML training
            scaled_input = (user_rfm - means) / stds

            # Calculate distance to trained centroids
            centers = np.array(model_data["centers"])
            distances = np.sum((centers - scaled_input) ** 2, axis=1)
            predicted_cluster = int(np.argmin(distances))

            matching_profile = profiles[profiles["cluster"] == predicted_cluster]

            if not matching_profile.empty:
                seg_name = matching_profile.iloc[0]["segment_name"]
                c_color = CLUSTER_COLORS.get(predicted_cluster, "#6366F1")
                avg_rec = matching_profile.iloc[0]["avg_recency_days"]
                avg_freq = matching_profile.iloc[0]["avg_frequency"]
                avg_mon = matching_profile.iloc[0]["avg_monetary"]

                # Recommendations tailored to business cohort
                cohort_recommendations = {
                    0: "Offer entry-level product bundles and low-threshold free shipping promotions to expand cart sizes and lift average order value.",
                    1: "Deploy post-purchase follow-up series, loyalty point incentives, and product category recommendations while brand affinity is fresh.",
                    2: "Enroll in VIP tier with priority customer service, exclusive early-access sales, and personalized premium reward privileges.",
                    3: "Initiate win-back reactivation campaigns with personalized discounts, feedback inquiries, and highlight new product arrivals."
                }

                st.markdown(f"""
                <div class="prediction-hero" style="border-color: {c_color}66;">
                    <div class="prediction-badge" style="background: {c_color}33; color: #FFFFFF; border: 1px solid {c_color};">
                        Predicted Segment Assignment
                    </div>
                    <div class="prediction-title" style="color: #FFFFFF;">
                        {seg_name}
                    </div>
                    <div class="prediction-cluster">
                        Assigned to <strong>Cluster ID: {predicted_cluster}</strong> based on nearest centroid Euclidean distance
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; padding: 14px 0; border-top: 1px solid rgba(255,255,255,0.1); border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 16px;">
                        <div>
                            <div style="font-size: 0.76rem; color: #CBD5E1; font-weight: 600; text-transform: uppercase;">Submitted Recency</div>
                            <div style="font-size: 1.20rem; font-weight: 700; color: #FFFFFF;">{recency_input} days</div>
                            <div style="font-size: 0.78rem; color: #94A3B8;">Cluster avg: {avg_rec:.1f} d</div>
                        </div>
                        <div>
                            <div style="font-size: 0.76rem; color: #CBD5E1; font-weight: 600; text-transform: uppercase;">Submitted Frequency</div>
                            <div style="font-size: 1.20rem; font-weight: 700; color: #FFFFFF;">{frequency_input} order{'s' if frequency_input > 1 else ''}</div>
                            <div style="font-size: 0.78rem; color: #94A3B8;">Cluster avg: {avg_freq:.2f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.76rem; color: #CBD5E1; font-weight: 600; text-transform: uppercase;">Submitted Monetary</div>
                            <div style="font-size: 1.20rem; font-weight: 700; color: #FFFFFF;">R$ {monetary_input:,.2f}</div>
                            <div style="font-size: 0.78rem; color: #94A3B8;">Cluster avg: R$ {avg_mon:.2f}</div>
                        </div>
                    </div>
                    <div style="font-size: 0.88rem; color: #F8FAFC; line-height: 1.55;">
                        <strong style="color: #38BDF8;">Recommended Engagement Strategy:</strong> {cohort_recommendations.get(predicted_cluster, "")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction execution failed: {e}")

    st.caption(
        "For meaningful results, input values should remain within the observed training bounds "
        "(Recency: 1-714 days, Frequency: 1-15 orders, Monetary: R$ 0.09 - 13,440.00). "
        "Predictions outside these ranges may yield skewed centroid proximity."
    )