"""
SynchroChain AI Dashboard - Main Streamlit Application
Complete supply chain optimization demo with ML models
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append('.')

# Import our custom modules
from login import display_login_page
from admin_dashboard import admin_dashboard
from user_simulation import user_simulation
from model_integration import ModelManager

# Page configuration
st.set_page_config(
    page_title="SynchroChain AI Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic styling for better readability
st.markdown("""
<style>
.main .block-container {
    font-size: 16px;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-size: 24px;
}
.stMarkdown h4 {
    font-size: 20px;
}
.stButton > button {
    font-size: 16px;
    padding: 0.5rem 1rem;
}
.stMetric {
    font-size: 18px;
}
.stDataFrame {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all ML models with caching."""
    try:
        model_manager = ModelManager()
        return model_manager
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'user_session' not in st.session_state:
        st.session_state.user_session = []
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Load models
    if st.session_state.model_manager is None:
        with st.spinner("Loading AI models..."):
            st.session_state.model_manager = load_models()
    
    # Main header
    st.markdown('<h1 class="main-header">🚀 SynchroChain AI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Check authentication and route to appropriate page
    if not st.session_state.logged_in:
        display_login_page()
    else:
        # Logout button in sidebar
        with st.sidebar:
            st.markdown("### User Info")
            st.write(f"**Role:** {st.session_state.role.title()}")
            st.write(f"**Status:** Logged In")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.role = None
                st.session_state.user_session = []
                st.session_state.predictions = {}
                st.rerun()
        
        # Debug: Print current role
        print(f"DEBUG: Current role in app.py: {st.session_state.role}")
        
        # Route to appropriate dashboard
        if st.session_state.role == 'admin':
            admin_dashboard()
        elif st.session_state.role == 'user':
            user_simulation()
        elif st.session_state.role == 'analyst':
            analyst_dashboard()
        else:
            st.error(f"Invalid role: {st.session_state.role}. Please log in again.")
            st.session_state.logged_in = False
            st.rerun()

def analyst_dashboard():
    """Analyst dashboard with focus on data analysis and insights."""
    st.markdown("### 📊 Data Analyst Dashboard")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 📈 Analysis Tools")
        page = st.radio(
            "Select Analysis",
            ["Model Performance", "Data Insights", "Trend Analysis", "Predictive Analytics"],
            index=0
        )
    
    if page == "Model Performance":
        st.markdown("#### 🧠 Model Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Intent Transformer Metrics")
            intent_metrics = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                'Value': [0.847, 0.823, 0.856, 0.839, 0.912]
            }
            st.dataframe(pd.DataFrame(intent_metrics), use_container_width=True)
        
        with col2:
            st.markdown("##### Delay Risk GNN Metrics")
            gnn_metrics = {
                'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE'],
                'Value': [0.0234, 0.153, 0.089, 0.912, 12.5]
            }
            st.dataframe(pd.DataFrame(gnn_metrics), use_container_width=True)
        
        # Performance trends
        st.markdown("##### Performance Trends Over Time")
        trend_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Intent_Accuracy': np.random.uniform(0.8, 0.9, 30),
            'GNN_R2': np.random.uniform(0.85, 0.95, 30),
            'PPO_Reward': np.random.uniform(10, 15, 30)
        })
        
        fig = px.line(trend_data, x='Date', y=['Intent_Accuracy', 'GNN_R2', 'PPO_Reward'],
                     title="Model Performance Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Data Insights":
        st.markdown("#### 📊 Data Insights & Patterns")
        
        # Data quality metrics
        st.markdown("##### Data Quality Assessment")
        quality_data = {
            'Dataset': ['User Sessions', 'Order Data', 'Supply Chain', 'Predictions'],
            'Completeness': [0.95, 0.87, 0.92, 0.98],
            'Accuracy': [0.89, 0.91, 0.88, 0.94],
            'Consistency': [0.92, 0.85, 0.90, 0.96]
        }
        st.dataframe(pd.DataFrame(quality_data), use_container_width=True)
        
        # Feature importance
        st.markdown("##### Feature Importance Analysis")
        feature_importance = pd.DataFrame({
            'Feature': ['Session Length', 'Add to Cart Count', 'Product Category', 'Order Value', 'Time of Day'],
            'Importance': [0.25, 0.30, 0.20, 0.15, 0.10]
        })
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', 
                    orientation='h', title="Feature Importance for Intent Prediction")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Trend Analysis":
        st.markdown("#### 📈 Trend Analysis")
        
        # User behavior trends
        st.markdown("##### User Behavior Patterns")
        behavior_data = pd.DataFrame({
            'Hour': range(24),
            'Sessions': np.random.poisson(50, 24) + np.sin(np.arange(24) * np.pi / 12) * 20,
            'Conversions': np.random.poisson(5, 24) + np.sin(np.arange(24) * np.pi / 12) * 3
        })
        
        fig = px.line(behavior_data, x='Hour', y=['Sessions', 'Conversions'],
                     title="Hourly User Activity Patterns")
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal trends
        st.markdown("##### Seasonal Trends")
        seasonal_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'Intent_Score': [0.65, 0.68, 0.72, 0.75, 0.78, 0.82, 0.80, 0.77, 0.74, 0.79, 0.85, 0.88],
            'Delay_Risk': [0.35, 0.32, 0.28, 0.25, 0.22, 0.18, 0.20, 0.23, 0.26, 0.21, 0.15, 0.12]
        })
        
        fig = px.line(seasonal_data, x='Month', y=['Intent_Score', 'Delay_Risk'],
                     title="Monthly Trends in Intent and Delay Risk")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Predictive Analytics":
        st.markdown("#### 🔮 Predictive Analytics")
        
        # Forecast next 7 days
        st.markdown("##### 7-Day Forecast")
        forecast_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=14, freq='D'),
            'Predicted_Intent': np.random.uniform(0.7, 0.9, 14),
            'Predicted_Delay_Risk': np.random.uniform(0.1, 0.3, 14),
            'Confidence': np.random.uniform(0.8, 0.95, 14)
        })
        
        fig = px.line(forecast_data, x='Date', y=['Predicted_Intent', 'Predicted_Delay_Risk'],
                     title="7-Day Predictive Forecast")
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment
        st.markdown("##### Risk Assessment Matrix")
        risk_matrix = pd.DataFrame({
            'Scenario': ['High Intent + Low Risk', 'High Intent + High Risk', 'Low Intent + Low Risk', 'Low Intent + High Risk'],
            'Probability': [0.25, 0.15, 0.45, 0.15],
            'Impact': ['High Revenue', 'High Cost', 'Low Revenue', 'Medium Cost'],
            'Recommendation': ['Scale Up', 'Mitigate Risk', 'Maintain', 'Monitor Closely']
        })
        st.dataframe(risk_matrix, use_container_width=True)
    
    # Quick stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", "1,247", "↗️ 12%")
    
    with col2:
        st.metric("Model Accuracy", "87.3%", "↗️ 2.1%")
    
    with col3:
        st.metric("Data Points", "45.2K", "↗️ 8.5%")
    
    with col4:
        st.metric("Insights Generated", "156", "↗️ 23%")

if __name__ == "__main__":
    main()
