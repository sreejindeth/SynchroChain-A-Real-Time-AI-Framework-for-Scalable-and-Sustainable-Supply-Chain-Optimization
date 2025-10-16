# 🚀 SynchroChain AI Dashboard

A comprehensive Streamlit dashboard demonstrating AI-powered supply chain optimization with real-time ML predictions.

## 📋 Overview

This dashboard showcases three integrated ML models working together to optimize supply chain operations:

1. **Intent Transformer** - Predicts user purchase intent from browsing behavior
2. **Delay Risk GNN** - Predicts supply chain delay risks using graph neural networks  
3. **PPO Agent** - Makes real-time supply chain decisions using reinforcement learning

## 🎯 Features

### 🔐 Authentication System
- Role-based access control (Admin, User, Analyst)
- Secure login with credential management
- Session management and audit logging

### 👨‍💼 Admin Dashboard
- **Live Dashboard**: Real-time system metrics and monitoring
- **Model Performance**: Detailed analytics for all ML models
- **System Analytics**: Performance metrics and data quality monitoring
- **User Management**: User account management and statistics

### 🛒 User Simulation
- Interactive e-commerce store simulation
- Real-time AI predictions as you browse
- Product catalog with 8 different items
- Live intent scoring and delay risk prediction
- AI-powered recommendations and decisions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment activated

### Installation

1. **Activate virtual environment:**
   ```bash
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_dashboard.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   - Navigate to `http://localhost:8501`
   - Use demo credentials to login

## 🔑 Demo Credentials

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| **Admin** | admin | admin123 | Full system access |
| **User** | user | user123 | E-commerce simulation |
| **Analyst** | analyst | analyst123 | Analytics and monitoring |

## 📊 Dashboard Components

### Main Application (`app.py`)
- Central Streamlit application
- Authentication routing
- Model loading and caching
- Responsive UI with custom CSS

### Login System (`login.py`)
- Secure authentication
- Role-based access control
- User credential management
- Login audit logging

### Admin Dashboard (`admin_dashboard.py`)
- Live system monitoring
- Model performance analytics
- System resource monitoring
- User management interface

### User Simulation (`user_simulation.py`)
- Interactive product catalog
- Real-time AI predictions
- Shopping cart simulation
- AI recommendation system

### Model Integration (`model_integration.py`)
- Unified model management
- Mock implementations for demo
- Real-time prediction pipeline
- State management

## 🧠 ML Models Integration

### Intent Transformer
- **Input**: User browsing session (view, search, add_to_cart, etc.)
- **Output**: Intent score (0-1) and urgency level (0-1)
- **Purpose**: Predicts likelihood of purchase

### Delay Risk GNN
- **Input**: Order context (category, value, shipping mode, etc.)
- **Output**: Delay risk probability (0-1)
- **Purpose**: Predicts supply chain delays

### PPO Agent
- **Input**: Combined state (intent, urgency, delay risk, inventory, etc.)
- **Output**: Supply chain decision
- **Purpose**: Optimizes real-time decisions

## 📈 Key Metrics

### Intent Scoring
- **High Intent (>0.7)**: Strong purchase likelihood
- **Medium Intent (0.4-0.7)**: Moderate purchase likelihood  
- **Low Intent (<0.4)**: Low purchase likelihood

### Delay Risk Assessment
- **Low Risk (<0.3)**: Minimal delay probability
- **Medium Risk (0.3-0.7)**: Moderate delay probability
- **High Risk (>0.7)**: High delay probability

### RL Decisions
- **Pre-allocate inventory**: High intent + high urgency
- **Expedite shipping**: High intent + high delay risk
- **Restock items**: Medium intent + low inventory
- **Normal operation**: Standard processing

## 🎮 How to Use

### As a User
1. Login with user credentials
2. Browse the product catalog
3. Click "View", "Add to Cart", or "Like" buttons
4. Watch real-time AI predictions update
5. See AI recommendations and decisions

### As an Admin
1. Login with admin credentials
2. Navigate between different dashboard sections
3. Monitor live system metrics
4. View model performance analytics
5. Manage user accounts

## 🔧 Technical Architecture

### Frontend
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Custom CSS**: Styled components

### Backend
- **Python**: Core application logic
- **PyTorch**: ML model framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

### Data Flow
1. User interactions → Session tracking
2. Session data → Intent Transformer
3. Order context → Delay Risk GNN
4. Combined predictions → PPO Agent
5. Decisions → Real-time display

## 📁 File Structure

```
├── app.py                          # Main Streamlit application
├── login.py                        # Authentication system
├── admin_dashboard.py              # Admin interface
├── user_simulation.py              # User simulation
├── model_integration.py            # ML model integration
├── test_dashboard.py               # Test suite
├── requirements_dashboard.txt      # Python dependencies
├── DASHBOARD_README.md             # This file
└── data/                           # Data directory
    ├── user_credentials.json       # User accounts
    ├── current_predictions.json    # Live predictions
    └── login_logs.json             # Audit logs
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_dashboard.py
```

Tests cover:
- Model integration
- Login system
- Data generation
- User simulation
- Test data creation

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production Deployment
1. Configure production settings
2. Set up proper authentication
3. Deploy to cloud platform
4. Configure domain and SSL

## 🔮 Future Enhancements

- [ ] Real ML model integration (replace mocks)
- [ ] Database integration for persistent data
- [ ] Advanced analytics and reporting
- [ ] Mobile-responsive design
- [ ] Real-time notifications
- [ ] API endpoints for external integration

## 🐛 Troubleshooting

### Common Issues

1. **Unicode Encoding Errors**
   - Fixed by removing Unicode characters from print statements

2. **Model Loading Failures**
   - Uses mock models for demo purposes
   - Real models can be integrated later

3. **Session State Issues**
   - Properly initialized in main application
   - Reset functionality available

### Debug Mode
```bash
streamlit run app.py --logger.level debug
```

## 📞 Support

For issues or questions:
1. Check the test suite output
2. Review error logs in console
3. Verify all dependencies are installed
4. Ensure virtual environment is activated

## 🎉 Success!

Your SynchroChain AI Dashboard is now ready! The system demonstrates:

✅ **Complete ML Pipeline**: Intent → Delay Risk → RL Decision  
✅ **Real-time Predictions**: Live updates as users interact  
✅ **Role-based Access**: Admin and user interfaces  
✅ **Interactive Demo**: Full e-commerce simulation  
✅ **Comprehensive Testing**: All components verified  

**Start the dashboard and experience AI-powered supply chain optimization!**















