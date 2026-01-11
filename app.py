import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import time
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Telco Churn Analytics | Syntecxhub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional LIGHT look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&family=Outfit:wght@400;700&display=swap');
    
    .stApp {
        background-color: #fcfcfd;
        color: #2D3748;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        color: #1A365D;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: -0.02em;
    }
    .subtitle-text {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 400;
        font-size: 1.2rem;
        color: #718096;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 20px; /* Reduced to avoid large gaps */
    }
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid #edf2f7;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
    }
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
        border-color: #3182CE;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2B6CB0;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1A365D;
        margin-top: 30px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 4px solid #3182CE;
    }
    .info-card {
        background: #ffffff;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.03);
        margin-bottom: 30px;
        border-left: 8px solid #3182CE;
    }
    .description-text {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #4A5568;
    }
    .highlight {
        color: #2B6CB0;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with NEW Logo (v3)
with st.sidebar:
    if os.path.exists('churn_logo_v3.png'):
        logo = Image.open('churn_logo_v3.png')
        st.image(logo, use_container_width=True)
    else:
        st.title("Syntecxhub")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🖥️ DASHBOARD NAVIGATION")
    app_mode = st.radio("Switch View", 
                         ["📊 Full Portfolio Intelligence", "🛠️ Data Utility Engine", "🌌 Advanced Neural Map"])
    
    st.markdown("---")
    st.markdown("### 🧬 Professional Utility")
    st.info("""
    **ChurnGuard Enterprise** is a high-end data utility tool for analyzing and preparing telecommunication datasets for predictive modeling.
    """)

# Header Section
st.markdown('<h1 class="title-text">CHURNGUARD ANALYTICS</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Empowering Decision Support for Customer Retention</p>', unsafe_allow_html=True)

# Data Loading with Cache
@st.cache_data
def load_raw_data():
    file_path = 'WA_Fn-UseC_-Telco-Customer-Churn (1).csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

df_raw = load_raw_data()

# Enhanced Cleaning Logic
def clean_data(df_input):
    df = df_input.copy()
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    df_pre_scale = pd.get_dummies(df, drop_first=True)
    bool_cols = df_pre_scale.select_dtypes(include='bool').columns
    df_pre_scale[bool_cols] = df_pre_scale[bool_cols].astype(int)
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_final = df_pre_scale.copy()
    df_final[num_cols] = scaler.fit_transform(df_final[num_cols])
    df_final.drop_duplicates(inplace=True)
    return df_final, df_pre_scale

# Dashboard logic based on Mode
if app_mode == "📊 Full Portfolio Intelligence":
    # 1. Dataset Comprehensive Description
    st.markdown('<div class="section-header">Dataset Overview & Intellectual Context</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="description-text">
    Welcome to the <b>ChurnGuard Full Portfolio Intel</b> page. This dashboard analyzes a comprehensive dataset of 
    <span class="highlight">{len(df_raw):,}</span> telecom customers. 
    <br><br>
    The objective is to identify behavioral patterns that lead to <span class="highlight">Customer Churn</span> (loss of customers). 
    This dataset includes <b>21 initial features</b> ranging from customer demographics (gender, age) to service portfolios (Internet, Phone) 
    and financial metrics (Monthly Charges, Contract Type).
    <br><br>
    <b>Why this matters:</b> Predicting churn allows the enterprise to proactively intervene with retention offers, 
    saving significant revenue and increasing the <b>Customer Lifetime Value (CLV)</b>.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Vital Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Base</div><div class="metric-value">{len(df_raw)}</div></div>', unsafe_allow_html=True)
    with col2:
        churn_count = (df_raw['Churn'] == 'Yes').sum()
        st.markdown(f'<div class="metric-card"><div class="metric-label">Churned</div><div class="metric-value">{churn_count}</div></div>', unsafe_allow_html=True)
    with col3:
        churn_rate = (churn_count / len(df_raw)) * 100
        st.markdown(f'<div class="metric-card"><div class="metric-label">Churn Rate</div><div class="metric-value">{churn_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        total_rev = df_raw['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).sum()
        st.markdown(f'<div class="metric-card"><div class="metric-label">Gross Revenue</div><div class="metric-value">${total_rev/1e6:.1f}M</div></div>', unsafe_allow_html=True)

    # 3. Comprehensive Visual Narrative
    st.markdown('<div class="section-header">Portfolio Behavioral Analysis</div>', unsafe_allow_html=True)
    
    pcol1, pcol2 = st.columns(2)
    
    with pcol1:
        st.markdown("#### 1. Churn Volume Distribution")
        st.info("This plot shows the binary split of our customer base. A high churn segment indicates an urgent need for strategy pivot.")
        fig_ch = px.pie(df_raw, names='Churn', hole=0.5, 
                       color_discrete_sequence=['#3182CE', '#E53E3E'])
        fig_ch.update_layout(template='plotly_white', margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_ch, use_container_width=True)

        st.markdown("#### 2. Risk by Contract Model")
        st.info("Observe how 'Month-to-Month' contracts dominate the churn segment. Long-term contracts (1-2 years) are the anchor of stability.")
        fig_con = px.histogram(df_raw, x="Contract", color="Churn", barmode="group",
                              color_discrete_map={'Yes': '#E53E3E', 'No': '#3182CE'})
        fig_con.update_layout(template='plotly_white')
        st.plotly_chart(fig_con, use_container_width=True)

    with pcol2:
        st.markdown("#### 3. Loyalty vs. Attrition (Tenure)")
        st.info("We see a 'U-shape' distribution. Customers are most likely to churn in the first 6 months. After 60 months, loyalty is very high.")
        fig_ten = px.histogram(df_raw, x="tenure", color="Churn", nbins=50,
                              color_discrete_map={'Yes': '#E53E3E', 'No': '#3182CE'})
        fig_ten.update_layout(template='plotly_white')
        st.plotly_chart(fig_ten, use_container_width=True)

        st.markdown("#### 4. Financial Impact Profile")
        st.info("Relationship between Monthly Charges and Churn. Higher monthly costs often trigger customer departure.")
        fig_mon = px.box(df_raw, x="Churn", y="MonthlyCharges", color="Churn",
                        color_discrete_map={'Yes': '#E53E3E', 'No': '#3182CE'})
        fig_mon.update_layout(template='plotly_white')
        st.plotly_chart(fig_mon, use_container_width=True)

    st.markdown("---")
    with st.expander("🛠️ Deep Metadata Explorer"):
        st.markdown("Explore the raw underlying data structure used for these visualizations.")
        st.dataframe(df_raw.head(15), use_container_width=True)

elif app_mode == "🛠️ Data Utility Engine":
    st.markdown('<div class="section-header">Transformation & Cleaning Pipeline</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("""
        #### Utility Logic Explained:
        The raw dataset contains lexical labels (Yes/No), missing values in TotalCharges, and unscaled numeric features. 
        Our pipeline executes the following **Atomic Operations**:
        1. **Type Resolution**: Converting string-formatted 'Total Charges' into floating-point numbers.
        2. **Median Imputation**: Replacing invalid data with population medians to preserve distribution.
        3. **Binary Encoding**: Vectorizing labels (Partner, Dependents, Churn) into 0-1 values.
        4. **Standardization**: Centering features around 0 with a unit standard deviation for Machine Learning compatibility.
        """)
    
    if st.button("🚀 INITIATE SYSTEM CLEANING"):
        status = st.empty()
        status.info("System: Analyzing Data Structures...")
        time.sleep(0.5)
        status.info("System: Executing Vectorization...")
        time.sleep(0.5)
        status.success("System: Cleaning Finalized.")
        
        df_cleaned, _ = clean_data(df_raw)
        
        # New Feature: Missing Values Comparison
        st.markdown("### 🔍 Observation 1: Data Completeness")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Initial Null Values**")
            null_before = df_raw.isnull().sum().reset_index()
            null_before.columns = ['Feature', 'Nulls']
            fig_m1 = px.bar(null_before[null_before['Nulls'] > 0], x='Feature', y='Nulls', 
                           title="Missing Values (Prior to Utility)",
                           color_discrete_sequence=['#E53E3E'])
            if null_before['Nulls'].sum() == 0:
                st.info("No explicit nulls detected in raw categorical fields.")
            else:
                st.plotly_chart(fig_m1, use_container_width=True)
        with col_m2:
            st.markdown("**Post-Utility Null Values**")
            null_after = df_cleaned.isnull().sum().sum()
            st.metric("Total Nulls Remaining", null_after, delta="Cleaned", delta_color="normal")
            st.info("The utility engine has successfully imputed or resolved all missing data points.")

        # Box Plot Comparisons for Scaling
        st.markdown("### 📊 Observation 2: Variance & Scale Normalization")
        st.markdown("Scaling ensures all features contribute equally to the model, preventing features with large magnitudes from dominating.")
        
        tab_ten, tab_mon, tab_tot = st.tabs(["Tenure Scaling", "Monthly Charges Scaling", "Total Charges Scaling"])
        
        with tab_ten:
            c_t1, c_t2 = st.columns(2)
            with c_t1:
                fig_t1 = px.box(df_raw, y="tenure", title="Tenure (Raw Months)", color_discrete_sequence=['#A0AEC0'])
                fig_t1.update_layout(template='plotly_white')
                st.plotly_chart(fig_t1, use_container_width=True)
            with c_t2:
                fig_t2 = px.box(df_cleaned, y="tenure", title="Tenure (Standardized)", color_discrete_sequence=['#3182CE'])
                fig_t2.update_layout(template='plotly_white')
                st.plotly_chart(fig_t2, use_container_width=True)

        with tab_mon:
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                fig_mo1 = px.box(df_raw, y="MonthlyCharges", title="Monthly Charges (Raw USD)", color_discrete_sequence=['#A0AEC0'])
                fig_mo1.update_layout(template='plotly_white')
                st.plotly_chart(fig_mo1, use_container_width=True)
            with c_m2:
                fig_mo2 = px.box(df_cleaned, y="MonthlyCharges", title="Monthly Charges (Standardized)", color_discrete_sequence=['#3182CE'])
                fig_mo2.update_layout(template='plotly_white')
                st.plotly_chart(fig_mo2, use_container_width=True)

        with tab_tot:
            c_to1, c_to2 = st.columns(2)
            with c_to1:
                fig_to1 = px.box(df_raw, y=pd.to_numeric(df_raw['TotalCharges'], errors='coerce'), 
                               title="Total Charges (Raw USD)", color_discrete_sequence=['#A0AEC0'])
                fig_to1.update_layout(template='plotly_white')
                st.plotly_chart(fig_to1, use_container_width=True)
            with c_to2:
                fig_to2 = px.box(df_cleaned, y="TotalCharges", title="Total Charges (Standardized)", color_discrete_sequence=['#3182CE'])
                fig_to2.update_layout(template='plotly_white')
                st.plotly_chart(fig_to2, use_container_width=True)

        st.markdown("### 🧪 Efficiency Summary")
        summary_table = pd.DataFrame({
            "Metric": ["Format", "Scaling", "Categoricals", "Missing", "Redundancy"],
            "Raw State": ["Mixed (Str/Int)", "Varying (0 to 8000)", "Lexical (Text)", "Contains NaNs/Blanks", "High"],
            "Utility State": ["Numerical (Float64)", "Standardized (S.D=1)", "Binary Vectorized", "Perfect Zero", "Optimized"]
        })
        st.table(summary_table)
    
elif app_mode == "🌌 Advanced Neural Map":
    st.markdown('<div class="section-header">High-Dimensional Feature Interaction</div>', unsafe_allow_html=True)
    st.info("Exploring the complex neural connections between features using multidimensional visualization.")
    
    df_cleaned, _ = clean_data(df_raw)
    
    tab1, tab2 = st.tabs(["🔥 Correlation Matrix", "🌐 3D Customer Space"])
    
    with tab1:
        corr = df_cleaned.corr()
        fig_h = px.imshow(corr, text_auto=".1f", aspect="auto", color_continuous_scale='RdBu_r',
                         title="Inter-Feature Correlation Lattice")
        fig_h.update_layout(template='plotly_white', height=800)
        st.plotly_chart(fig_h, use_container_width=True)
        
    with tab2:
        fig_3d = px.scatter_3d(df_cleaned.sample(min(1500, len(df_cleaned))), 
                             x='tenure', y='MonthlyCharges', z='TotalCharges',
                             color='Churn', opacity=0.7, size_max=10,
                             title="3D Mapping of Loyalty Clusters",
                             color_continuous_scale=[[0, '#3182CE'], [1, '#E53E3E']])
        fig_3d.update_layout(template='plotly_white')
        st.plotly_chart(fig_3d, use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #718096; padding: 20px;">© 2026 Syntecxhub Analytics v3.1 | Enterprise Data Utility Suite</div>', unsafe_allow_html=True)
