# src/dashboard/streamlit_app.py
"""
SynchroChain Streamlit Dashboard
Interactive web interface for supply chain AI system
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.system.orchestrator import ModelOrchestrator

# Page configuration
st.set_page_config(
    page_title="SynchroChain - AI Supply Chain Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_orchestrator():
    """Load the model orchestrator (cached for performance)."""
    try:
        return ModelOrchestrator()
    except Exception as e:
        st.error(f"Failed to load AI models: {e}")
        return None

def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Sample products
    products = [
        "Wireless Headphones", "Smartphone Case", "Laptop Stand", 
        "USB-C Cable", "Wireless Mouse", "Bluetooth Speaker",
        "Phone Charger", "Tablet Stand", "Keyboard", "Monitor"
    ]
    
    # Sample user sessions
    sessions = []
    for i in range(50):
        session_length = np.random.randint(2, 8)
        session = []
        for j in range(session_length):
            product = np.random.choice(products)
            action = np.random.choice(['view', 'add_to_cart', 'search'], p=[0.7, 0.2, 0.1])
            timestamp = datetime.now() - timedelta(minutes=np.random.randint(0, 60))
            session.append({
                'product_name': product,
                'action': action,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
        sessions.append(session)
    
    return sessions, products

def simulate_real_time_data():
    """Simulate real-time supply chain metrics."""
    return {
        'active_orders': np.random.randint(150, 300),
        'inventory_alerts': np.random.randint(5, 25),
        'delivery_delays': np.random.randint(2, 15),
        'carbon_efficiency': np.random.uniform(0.75, 0.95),
        'cost_savings': np.random.uniform(15, 35),
        'customer_satisfaction': np.random.uniform(0.85, 0.98)
    }

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 SynchroChain</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Driven Supply Chain Optimization System</p>', unsafe_allow_html=True)
    
    # Load orchestrator
    orchestrator = load_orchestrator()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🤖 AI Predictions", "📊 Analytics", "⚙️ Settings"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🤖 AI Predictions":
        show_ai_predictions(orchestrator)
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """Show the main dashboard with real-time metrics."""
    
    st.header("📊 Real-Time Supply Chain Overview")
    
    # Get real-time data
    metrics = simulate_real_time_data()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Orders",
            value=f"{metrics['active_orders']:,}",
            delta=f"+{np.random.randint(5, 20)}"
        )
    
    with col2:
        st.metric(
            label="Inventory Alerts",
            value=metrics['inventory_alerts'],
            delta=f"-{np.random.randint(1, 5)}"
        )
    
    with col3:
        st.metric(
            label="Delivery Delays",
            value=metrics['delivery_delays'],
            delta=f"-{np.random.randint(1, 3)}"
        )
    
    with col4:
        st.metric(
            label="Cost Savings",
            value=f"{metrics['cost_savings']:.1f}%",
            delta=f"+{np.random.uniform(1, 3):.1f}%"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Order Volume Trend")
        
        # Generate sample time series data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        orders = np.random.poisson(200, len(dates)) + np.sin(np.arange(len(dates)) * 0.2) * 50
        
        df_orders = pd.DataFrame({'Date': dates, 'Orders': orders})
        
        fig = px.line(df_orders, x='Date', y='Orders', 
                     title="Daily Order Volume (Last 30 Days)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 AI Decision Distribution")
        
        # Sample decision data
        decisions = ['Standard Shipping', 'Expedited Shipping', 'Priority Shipping']
        values = [45, 35, 20]  # Percentages
        
        fig = px.pie(values=values, names=decisions, 
                    title="AI Fulfillment Decisions (Today)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance indicators
    st.header("🎯 Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card success-card">
            <h4>Carbon Efficiency</h4>
            <h2>{metrics['carbon_efficiency']:.1%}</h2>
            <p>↑ 5.2% from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card success-card">
            <h4>Customer Satisfaction</h4>
            <h2>{metrics['customer_satisfaction']:.1%}</h2>
            <p>↑ 3.1% from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card warning-card">
            <h4>Inventory Turnover</h4>
            <h2>12.3x</h2>
            <p>↓ 0.8x from last month</p>
        </div>
        """, unsafe_allow_html=True)

def show_ai_predictions(orchestrator):
    """Show AI prediction interface."""
    
    st.header("🤖 AI-Powered Predictions")
    
    if orchestrator is None:
        st.error("AI models are not available. Please check the system configuration.")
        return
    
    # User session input
    st.subheader("👤 User Session Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Simulate User Browsing Behavior:**")
        
        # Product selection
        sessions, products = create_sample_data()
        
        selected_products = st.multiselect(
            "Select products viewed:",
            products,
            default=products[:3]
        )
        
        # Actions
        actions = []
        for product in selected_products:
            action = st.selectbox(
                f"Action for {product}:",
                ['view', 'add_to_cart', 'search'],
                key=f"action_{product}"
            )
            actions.append({'product_name': product, 'action': action, 'timestamp': datetime.now().isoformat()})
    
    with col2:
        st.write("**Supply Chain Context:**")
        
        warehouse = st.selectbox(
            "Warehouse Location:",
            ['US_WEST', 'US_EAST', 'EU_CENTRAL', 'ASIA_PACIFIC']
        )
        
        inventory_level = st.slider("Current Inventory Level:", 0, 500, 150)
        delay_rate = st.slider("Historical Delay Rate:", 0.0, 1.0, 0.15, 0.01)
        carbon_factor = st.slider("Carbon Cost Factor:", 0.0, 1.0, 0.3, 0.01)
        
        supply_context = {
            'warehouse_location': warehouse,
            'current_inventory': inventory_level,
            'historical_delay_rate': delay_rate,
            'carbon_cost_factor': carbon_factor
        }
    
    # Run prediction
    if st.button("🚀 Run AI Prediction", type="primary"):
        with st.spinner("Running AI pipeline..."):
            
            # Simulate processing time
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Get predictions
            results = orchestrator.orchestrate_full_pipeline(actions, supply_context)
            
            # Display results
            st.success("✅ AI Pipeline Completed Successfully!")
            
            # Results layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🧠 Intent Analysis")
                intent_preds = results['intent_predictions']
                
                intent_df = pd.DataFrame({
                    'Intent Level': ['Low', 'Medium', 'High'],
                    'Probability': [
                        intent_preds['intent_low'],
                        intent_preds['intent_medium'], 
                        intent_preds['intent_high']
                    ]
                })
                
                fig = px.bar(intent_df, x='Intent Level', y='Probability',
                           title="Purchase Intent Prediction")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Urgency Score", f"{intent_preds['urgency']:.2f}")
            
            with col2:
                st.subheader("🏭 Supply Chain Risk")
                supply_preds = results['supply_chain_predictions']
                
                risk_df = pd.DataFrame({
                    'Risk Type': ['Delay Risk', 'Inventory Risk', 'Carbon Cost'],
                    'Level': [
                        supply_preds['delay_risk'],
                        supply_preds['inventory_risk'],
                        supply_preds['carbon_cost']
                    ]
                })
                
                fig = px.bar(risk_df, x='Risk Type', y='Level',
                           title="Supply Chain Risk Assessment")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                st.subheader("🎯 Final Decision")
                decision = results['final_decision']
                
                # Decision display
                if decision['action'] == 0:
                    card_class = "success-card"
                elif decision['action'] == 1:
                    card_class = "warning-card"
                else:
                    card_class = "danger-card"
                
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h4>Recommended Action</h4>
                    <h2>{decision['action_name']}</h2>
                    <p>Confidence: {decision['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("**Reasoning:**")
                st.write(decision['reasoning'])
                
                st.metric("Processing Time", f"{results['processing_time_seconds']:.3f}s")
            
            # Detailed results
            with st.expander("📋 Detailed Results"):
                st.json(results)

def show_analytics():
    """Show analytics and historical performance."""
    
    st.header("📊 Analytics & Performance")
    
    # Performance metrics over time
    st.subheader("📈 Historical Performance")
    
    # Generate sample historical data
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
    
    # Create sample metrics
    accuracy_data = 0.65 + np.random.normal(0, 0.05, len(dates)).cumsum() * 0.001
    accuracy_data = np.clip(accuracy_data, 0.6, 0.8)
    
    cost_savings = 15 + np.random.normal(0, 2, len(dates)).cumsum() * 0.1
    cost_savings = np.clip(cost_savings, 10, 40)
    
    delivery_time = 3.5 - np.random.normal(0, 0.1, len(dates)).cumsum() * 0.01
    delivery_time = np.clip(delivery_time, 2.0, 5.0)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('AI Prediction Accuracy', 'Cost Savings %', 'Average Delivery Time (days)', 'Customer Satisfaction'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add traces
    fig.add_trace(go.Scatter(x=dates, y=accuracy_data, name='Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=cost_savings, name='Cost Savings'), row=1, col=2)
    fig.add_trace(go.Scatter(x=dates, y=delivery_time, name='Delivery Time'), row=2, col=1)
    
    # Customer satisfaction
    satisfaction = 0.85 + np.random.normal(0, 0.02, len(dates)).cumsum() * 0.001
    satisfaction = np.clip(satisfaction, 0.8, 0.95)
    fig.add_trace(go.Scatter(x=dates, y=satisfaction, name='Satisfaction'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance comparison
    st.subheader("🏆 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Intent Transformer performance
        st.write("**Intent Transformer Results:**")
        
        performance_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [0.69, 0.67, 0.69, 0.68],
            'Baseline': [0.33, 0.33, 0.33, 0.33]
        }
        
        df_perf = pd.DataFrame(performance_data)
        
        fig = px.bar(df_perf, x='Metric', y=['Value', 'Baseline'], 
                    title="Intent Transformer vs Baseline", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # PPO Agent performance
        st.write("**PPO Agent Results:**")
        
        ppo_data = {
            'Metric': ['Avg Reward', 'Cost Efficiency', 'Risk Mitigation', 'Policy Entropy'],
            'Value': [842.72, 3.118, 0.522, 0.652]
        }
        
        df_ppo = pd.DataFrame(ppo_data)
        
        fig = px.bar(df_ppo, x='Metric', y='Value',
                    title="PPO Agent Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Show system settings and configuration."""
    
    st.header("⚙️ System Settings")
    
    # Model configuration
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Intent Transformer Settings:**")
        
        max_seq_length = st.slider("Max Sequence Length:", 4, 16, 8)
        learning_rate = st.select_slider(
            "Learning Rate:",
            options=[1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
            value=3e-4,
            format_func=lambda x: f"{x:.0e}"
        )
        batch_size = st.selectbox("Batch Size:", [8, 16, 32, 64], index=2)
        
    with col2:
        st.write("**PPO Agent Settings:**")
        
        ppo_epochs = st.slider("PPO Epochs:", 50, 200, 100)
        rollout_steps = st.slider("Rollout Steps:", 1024, 4096, 2048)
        gamma = st.slider("Discount Factor (γ):", 0.9, 0.999, 0.99, 0.001)
    
    # System monitoring
    st.subheader("📊 System Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CPU Usage", "45%", "↓ 5%")
        st.metric("Memory Usage", "2.1 GB", "↑ 0.2 GB")
    
    with col2:
        st.metric("Model Load Time", "1.2s", "↓ 0.3s")
        st.metric("Prediction Latency", "0.009s", "↓ 0.001s")
    
    with col3:
        st.metric("Requests/Hour", "1,247", "↑ 156")
        st.metric("Error Rate", "0.02%", "↓ 0.01%")
    
    # Configuration export/import
    st.subheader("💾 Configuration Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Configuration"):
            config = {
                'intent_transformer': {
                    'max_seq_length': max_seq_length,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                },
                'ppo': {
                    'epochs': ppo_epochs,
                    'rollout_steps': rollout_steps,
                    'gamma': gamma
                }
            }
            
            st.download_button(
                label="Download config.json",
                data=json.dumps(config, indent=2),
                file_name="synchrochain_config.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("📤 Import Configuration", type=['json'])
        if uploaded_file is not None:
            config = json.load(uploaded_file)
            st.success("Configuration imported successfully!")
            st.json(config)

if __name__ == "__main__":
    main()

