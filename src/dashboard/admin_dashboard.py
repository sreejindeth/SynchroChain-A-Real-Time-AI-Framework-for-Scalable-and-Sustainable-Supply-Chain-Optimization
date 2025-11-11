"""
Admin Dashboard Module for SynchroChain AI Dashboard
Comprehensive admin view with live monitoring and model performance
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

# Add core directory to path for login import
core_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core')
sys.path.insert(0, core_dir)
from login import get_user_stats

def admin_dashboard():
    """Main admin dashboard function."""
    st.markdown("### 🎛️ Admin Dashboard")
    st.markdown("---")
    
    # Add basic styling for admin dashboard
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
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Select Page",
            ["Live Dashboard", "Model Performance", "System Analytics", "User Management"],
            index=0
        )
    
    # Route to appropriate page
    if page == "Live Dashboard":
        live_dashboard_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "System Analytics":
        system_analytics_page()
    elif page == "User Management":
        user_management_page()

def live_dashboard_page():
    """Live dashboard with real-time metrics."""
    st.markdown("#### 📈 Live System Metrics")
    
    # Load current predictions if available
    current_predictions = load_current_predictions()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Intent Score",
            value=f"{current_predictions.get('intent_score', 0.0):.3f}",
            delta=f"{current_predictions.get('intent_delta', 0.0):.3f}",
            help="Current user purchase intent prediction"
        )
    
    with col2:
        st.metric(
            label="⚠️ Delay Risk",
            value=f"{current_predictions.get('delay_risk', 0.0):.3f}",
            delta=f"{current_predictions.get('delay_delta', 0.0):.3f}",
            help="Predicted supply chain delay risk"
        )
    
    with col3:
        st.metric(
            label="🤖 RL Decision",
            value=current_predictions.get('rl_decision', 'Normal'),
            delta=current_predictions.get('rl_confidence', 0.0),
            help="Current reinforcement learning agent decision"
        )
    
    with col4:
        st.metric(
            label="📊 System Status",
            value="🟢 Online",
            delta="+2.3%",
            help="Overall system health status"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Intent Score Trends")
        intent_data = generate_intent_trends()
        fig_intent = px.line(
            intent_data, 
            x='timestamp', 
            y='intent_score',
            title="Intent Score Over Time",
            color_discrete_sequence=['#667eea']
        )
        fig_intent.update_layout(height=300)
        st.plotly_chart(fig_intent, use_container_width=True)
    
    with col2:
        st.markdown("#### ⚠️ Delay Risk Distribution")
        delay_data = generate_delay_risk_data()
        fig_delay = px.histogram(
            delay_data,
            x='delay_risk',
            nbins=20,
            title="Delay Risk Distribution",
            color_discrete_sequence=['#ff6b6b']
        )
        fig_delay.update_layout(height=300)
        st.plotly_chart(fig_delay, use_container_width=True)
    
    # Real-time activity
    st.markdown("#### 🔄 Real-time Activity")
    activity_data = generate_activity_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Recent Predictions")
        st.dataframe(activity_data.head(10), use_container_width=True)
    
    with col2:
        st.markdown("##### System Alerts")
        alerts = generate_system_alerts()
        for alert in alerts:
            if alert['severity'] == 'high':
                st.error(f"🔴 {alert['message']}")
            elif alert['severity'] == 'medium':
                st.warning(f"🟡 {alert['message']}")
            else:
                st.info(f"🔵 {alert['message']}")

def model_performance_page():
    """Model performance monitoring page."""
    st.markdown("#### 🧠 Model Performance Analysis")
    
    # Tabs for different models
    tab1, tab2, tab3 = st.tabs(["Intent Transformer", "Delay Risk GNN", "PPO Agent"])
    
    with tab1:
        st.markdown("##### Intent Transformer Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy metrics
            accuracy_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Value': [0.847, 0.823, 0.856, 0.839]
            }
            st.dataframe(pd.DataFrame(accuracy_data), use_container_width=True)
        
        with col2:
            # Confusion matrix
            confusion_matrix = np.array([[45, 8, 3], [6, 52, 4], [2, 5, 48]])
            fig_conf = px.imshow(
                confusion_matrix,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Training progress
        st.markdown("##### Training Progress")
        train_data = generate_training_data('intent')
        fig_train = px.line(
            train_data,
            x='epoch',
            y=['train_loss', 'val_loss'],
            title="Training Loss Over Time"
        )
        st.plotly_chart(fig_train, use_container_width=True)
    
    with tab2:
        st.markdown("##### Delay Risk GNN Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regression metrics
            regression_data = {
                'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
                'Value': [0.0234, 0.153, 0.089, 0.912]
            }
            st.dataframe(pd.DataFrame(regression_data), use_container_width=True)
        
        with col2:
            # Actual vs Predicted
            actual_pred_data = generate_actual_vs_predicted()
            
            # Check if statsmodels is available for trendline
            try:
                import statsmodels.api as sm  # noqa: F401
                has_statsmodels = True
            except ImportError:
                has_statsmodels = False
            
            fig_scatter = px.scatter(
                actual_pred_data,
                x='actual',
                y='predicted',
                title="Actual vs Predicted Delay Risk",
                trendline="ols" if has_statsmodels else None
            )
            st.plotly_chart(fig_scatter, width='stretch')
        
        # Feature importance
        st.markdown("##### Feature Importance")
        feature_importance = generate_feature_importance()
        fig_features = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title="Feature Importance for Delay Risk Prediction"
        )
        st.plotly_chart(fig_features, use_container_width=True)
    
    with tab3:
        st.markdown("##### PPO Agent Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Training metrics
            rl_metrics = {
                'Metric': ['Avg Reward', 'Episode Length', 'Success Rate', 'Exploration Rate'],
                'Value': [12.45, 45.2, 0.78, 0.15]
            }
            st.dataframe(pd.DataFrame(rl_metrics), use_container_width=True)
        
        with col2:
            # Action distribution
            action_data = generate_action_distribution()
            fig_actions = px.pie(
                action_data,
                values='count',
                names='action',
                title="Action Distribution"
            )
            st.plotly_chart(fig_actions, use_container_width=True)
        
        # Learning curve
        st.markdown("##### Learning Curve")
        learning_data = generate_learning_curve()
        fig_learning = px.line(
            learning_data,
            x='episode',
            y='reward',
            title="Reward Over Episodes"
        )
        st.plotly_chart(fig_learning, use_container_width=True)

def system_analytics_page():
    """System analytics and insights page."""
    st.markdown("#### 📊 System Analytics")
    
    # User statistics
    user_stats = get_user_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Logins", user_stats.get('total_logins', 0))
    
    with col2:
        st.metric("Recent Logins (24h)", user_stats.get('recent_logins', 0))
    
    with col3:
        last_login = user_stats.get('last_login', 'Never')
        if last_login != 'Never':
            last_login = datetime.fromisoformat(last_login).strftime('%H:%M')
        st.metric("Last Login", last_login)
    
    # System performance metrics
    st.markdown("##### System Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU and Memory usage
        performance_data = generate_performance_data()
        fig_perf = px.line(
            performance_data,
            x='time',
            y=['cpu_usage', 'memory_usage'],
            title="System Resource Usage"
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        # Model inference times
        inference_data = generate_inference_times()
        fig_inference = px.bar(
            inference_data,
            x='model',
            y='avg_time_ms',
            title="Average Inference Time by Model"
        )
        st.plotly_chart(fig_inference, use_container_width=True)
    
    # Data quality metrics
    st.markdown("##### Data Quality Metrics")
    quality_metrics = generate_data_quality_metrics()
    st.dataframe(quality_metrics, use_container_width=True)

def user_management_page():
    """User management page."""
    st.markdown("#### 👥 User Management")
    
    # User list
    st.markdown("##### Active Users")
    users_data = load_user_data()
    st.dataframe(users_data, use_container_width=True)
    
    # Add new user form
    st.markdown("##### Add New User")
    
    with st.form("add_user_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
        
        with col2:
            new_role = st.selectbox("Role", ["admin", "user", "analyst"])
            new_email = st.text_input("Email")
        
        if st.form_submit_button("Add User"):
            if new_username and new_password and new_email:
                from login import create_new_user
                success, message = create_new_user(
                    new_username, new_password, new_role, 
                    new_username.title(), new_email
                )
                if success:
                    st.success(message)
                    st.rerun()  # Refresh the page to show updated user list
                else:
                    st.error(message)
            else:
                st.error("Please fill in all fields")

# Helper functions for data generation
def load_current_predictions():
    """Load current predictions from file."""
    predictions_file = "data/current_predictions.json"
    
    if os.path.exists(predictions_file):
        try:
            with open(predictions_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Return default values
    return {
        'intent_score': 0.0,
        'delay_risk': 0.0,
        'rl_decision': 'Normal',
        'intent_delta': 0.0,
        'delay_delta': 0.0,
        'rl_confidence': 0.0
    }

def generate_intent_trends():
    """Generate intent score trend data."""
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq='1H'
    )
    
    # Generate realistic trend data
    base_intent = 0.5
    trend = np.sin(np.arange(len(timestamps)) * 0.1) * 0.2
    noise = np.random.normal(0, 0.05, len(timestamps))
    intent_scores = base_intent + trend + noise
    intent_scores = np.clip(intent_scores, 0, 1)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'intent_score': intent_scores
    })

def generate_delay_risk_data():
    """Generate delay risk distribution data."""
    # Generate realistic delay risk data
    delay_risks = np.random.beta(2, 5, 1000)  # Beta distribution for 0-1 values
    return pd.DataFrame({'delay_risk': delay_risks})

def generate_activity_data():
    """Generate recent activity data."""
    activities = []
    for i in range(20):
        activities.append({
            'timestamp': datetime.now() - timedelta(minutes=i*5),
            'user_id': f'user_{np.random.randint(1, 10)}',
            'action': np.random.choice(['view', 'add_to_cart', 'checkout']),
            'intent_score': round(np.random.uniform(0, 1), 3),
            'delay_risk': round(np.random.uniform(0, 1), 3)
        })
    
    return pd.DataFrame(activities)

def generate_system_alerts():
    """Generate system alerts."""
    return [
        {'severity': 'low', 'message': 'Model retraining scheduled for tonight'},
        {'severity': 'medium', 'message': 'High delay risk detected in 3 orders'},
        {'severity': 'high', 'message': 'Intent transformer accuracy dropped below 80%'}
    ]

def generate_training_data(model_type):
    """Generate training progress data."""
    epochs = list(range(1, 21))
    
    if model_type == 'intent':
        train_loss = [0.8 * np.exp(-e/10) + 0.1 + np.random.normal(0, 0.02) for e in epochs]
        val_loss = [0.9 * np.exp(-e/12) + 0.15 + np.random.normal(0, 0.03) for e in epochs]
    else:
        train_loss = [0.6 * np.exp(-e/8) + 0.2 + np.random.normal(0, 0.02) for e in epochs]
        val_loss = [0.7 * np.exp(-e/10) + 0.25 + np.random.normal(0, 0.03) for e in epochs]
    
    data = []
    for i, epoch in enumerate(epochs):
        data.append({'epoch': epoch, 'train_loss': train_loss[i], 'val_loss': val_loss[i]})
    
    return pd.DataFrame(data)

def generate_actual_vs_predicted():
    """Generate actual vs predicted data for regression."""
    actual = np.random.uniform(0, 1, 100)
    predicted = actual + np.random.normal(0, 0.1, 100)
    predicted = np.clip(predicted, 0, 1)
    
    return pd.DataFrame({'actual': actual, 'predicted': predicted})

def generate_feature_importance():
    """Generate feature importance data."""
    features = ['Order Value', 'Shipping Distance', 'Supplier Reliability', 
               'Inventory Level', 'Seasonal Factor', 'Weather Risk']
    importance = np.random.uniform(0.05, 0.3, len(features))
    importance = importance / importance.sum()  # Normalize
    
    return pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance')

def generate_performance_data():
    """Generate system performance data."""
    times = pd.date_range(start=datetime.now() - timedelta(hours=6), 
                         end=datetime.now(), freq='10min')
    
    cpu_usage = 50 + 20 * np.sin(np.arange(len(times)) * 0.1) + np.random.normal(0, 5, len(times))
    memory_usage = 60 + 15 * np.cos(np.arange(len(times)) * 0.15) + np.random.normal(0, 3, len(times))
    
    return pd.DataFrame({
        'time': times,
        'cpu_usage': np.clip(cpu_usage, 0, 100),
        'memory_usage': np.clip(memory_usage, 0, 100)
    })

def generate_inference_times():
    """Generate model inference time data."""
    models = ['Intent Transformer', 'Delay Risk GNN', 'PPO Agent']
    times = [45, 120, 15]  # milliseconds
    
    return pd.DataFrame({
        'model': models,
        'avg_time_ms': times
    })

def generate_data_quality_metrics():
    """Generate data quality metrics."""
    return pd.DataFrame({
        'Metric': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness', 'Validity'],
        'Score': [0.95, 0.87, 0.92, 0.89, 0.94],
        'Status': ['Good', 'Fair', 'Good', 'Good', 'Good']
    })

def generate_action_distribution():
    """Generate PPO action distribution data."""
    actions = ['Pre-allocate', 'Restock', 'Expedite', 'Normal']
    counts = [25, 15, 10, 50]
    
    return pd.DataFrame({
        'action': actions,
        'count': counts
    })

def generate_learning_curve():
    """Generate PPO learning curve data."""
    episodes = list(range(1, 101))
    rewards = []
    base_reward = 5
    
    for ep in episodes:
        # Simulate learning curve with noise
        reward = base_reward + 10 * (1 - np.exp(-ep/30)) + np.random.normal(0, 2)
        rewards.append(reward)
    
    return pd.DataFrame({
        'episode': episodes,
        'reward': rewards
    })

def load_user_data():
    """Load user data for management."""
    return pd.DataFrame({
        'Username': ['admin', 'user', 'analyst'],
        'Role': ['admin', 'user', 'analyst'],
        'Last Login': ['2 min ago', '15 min ago', '1 hour ago'],
        'Status': ['Online', 'Online', 'Offline']
    })
