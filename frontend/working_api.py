"""
Working Flask API with Proper Login Handling
Simplified version that definitely works
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import time
from datetime import datetime
import numpy as np

app = Flask(__name__, static_folder='.')
CORS(app)

# Simple authentication
VALID_USERS = {
    'admin': {'password': 'admin123', 'role': 'admin'},
    'user': {'password': 'user123', 'role': 'user'}
}

@app.route('/')
def serve_frontend():
    """Serve the enhanced HTML file"""
    return send_from_directory('.', 'enhanced_index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Working API is running'
    })

@app.route('/api/login', methods=['POST'])
def login():
    """Login endpoint with proper error handling"""
    try:
        print("Login attempt received")
        
        # Get request data
        data = request.get_json()
        print(f"Login data: {data}")
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received'
            }), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', '').strip()
        
        print(f"Username: '{username}', Password: '{password}', Role: '{role}'")
        
        # Validate credentials
        if username in VALID_USERS:
            user_info = VALID_USERS[username]
            if user_info['password'] == password:
                print(f"Login successful for {username}")
                return jsonify({
                    'success': True,
                    'user': {
                        'username': username,
                        'role': user_info['role'],
                        'token': f"token_{username}_{int(time.time())}"
                    }
                })
            else:
                print(f"Invalid password for {username}")
                return jsonify({
                    'success': False,
                    'error': 'Invalid password'
                }), 401
        else:
            print(f"Invalid username: {username}")
            return jsonify({
                'success': False,
                'error': 'Invalid username'
            }), 401
            
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with detailed AI pipeline results"""
    try:
        data = request.get_json()
        
        # Extract input parameters
        products = data.get('products', [])
        warehouse = data.get('warehouse', 'US_WEST')
        inventory = data.get('inventory', 150)
        delay_risk = data.get('delay_risk', 15)
        session_duration = data.get('session_duration', 5)
        add_to_cart = data.get('add_to_cart', False)
        search_queries = data.get('search_queries', 0)
        
        print(f"Prediction request: {data}")
        
        # Intent Transformer Simulation
        base_intent = np.random.uniform(0.3, 0.7)
        product_bonus = len(products) * 0.05
        session_bonus = min(0.2, session_duration / 5)  # 5 minutes = 0.2 bonus
        action_bonus = 0.1 if add_to_cart else 0
        search_bonus = search_queries * 0.05
        
        intent_score = min(base_intent + product_bonus + session_bonus + action_bonus + search_bonus, 1.0)
        urgency_score = np.random.uniform(0.4, 0.9)
        if add_to_cart:
            urgency_score += 0.1
        urgency_score = min(urgency_score, 1.0)
        
        # GNN Module Simulation
        inventory_risk = 0.3 if inventory < 100 else (0.1 if inventory > 300 else 0.2)
        delay_risk_normalized = delay_risk / 100
        warehouse_risk = 0.15 if warehouse == 'ASIA_PACIFIC' else 0.1
        total_risk = (inventory_risk + delay_risk_normalized + warehouse_risk) / 3
        
        # Carbon cost calculation
        carbon_cost = np.random.uniform(0.5, 3.0)
        if warehouse == 'ASIA_PACIFIC':
            carbon_cost *= 1.5
        
        # RL Agent Decision
        if intent_score > 0.6 and total_risk < 0.3:
            decision = 'priority'
            confidence = 0.85
        elif intent_score > 0.4 and total_risk < 0.5:
            decision = 'expedited'
            confidence = 0.72
        else:
            decision = 'standard'
            confidence = 0.68
        
        # Orchestrator Processing Time
        processing_time = np.random.uniform(2, 5)
        
        return jsonify({
            'success': True,
            'intent_score': round(intent_score, 3),
            'urgency_score': round(urgency_score, 3),
            'risk_assessment': round(total_risk, 3),
            'final_decision': decision,
            'confidence': round(confidence, 3),
            'processing_time_ms': round(processing_time, 1),
            'model_used': 'simulation',
            'inventory_level': inventory,
            'carbon_cost': round(carbon_cost, 2),
            'delay_risk': delay_risk,
            'warehouse': warehouse,
            'products_count': len(products),
            'session_duration': session_duration,
            'pipeline_steps': {
                'intent_transformer': {
                    'status': 'completed',
                    'processing_time_ms': round(processing_time * 0.4, 1),
                    'accuracy': 0.69
                },
                'gnn_module': {
                    'status': 'completed',
                    'processing_time_ms': round(processing_time * 0.3, 1),
                    'mae': 0.045
                },
                'rl_agent': {
                    'status': 'completed',
                    'processing_time_ms': round(processing_time * 0.2, 1),
                    'improvement': 0.213
                },
                'orchestrator': {
                    'status': 'completed',
                    'processing_time_ms': round(processing_time * 0.1, 1),
                    'throughput': 357
                }
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics"""
    return jsonify({
        'success': True,
        'metrics': {
            'active_orders': np.random.randint(1200, 1500),
            'ai_accuracy': 69,
            'processing_speed_ms': 3.2,
            'cost_savings_percent': 25,
            'throughput_per_hour': np.random.randint(300, 400),
            'error_rate_percent': 0.02,
            'cpu_usage_percent': np.random.randint(40, 60),
            'memory_usage_gb': round(np.random.uniform(1.8, 2.5), 1)
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """Get AI pipeline status"""
    return jsonify({
        'success': True,
        'pipeline_status': {
            'intent_transformer': {'status': 'online', 'accuracy': 0.69},
            'gnn_module': {'status': 'online', 'mae': 0.045},
            'rl_agent': {'status': 'online', 'improvement': 0.213},
            'orchestrator': {'status': 'online', 'throughput_per_second': 357}
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/data-layer/status', methods=['GET'])
def get_data_layer_status():
    """Get data layer status"""
    return jsonify({
        'success': True,
        'data_layer_status': {
            'live_databases': {'status': 'online', 'total_records': 1500},
            'real_time_cache': {'status': 'online', 'cache_hit_rate': 0.85},
            'data_processing': {'status': 'online', 'processing_interval_seconds': 60},
            'feature_store': {'status': 'online', 'total_features': 1200}
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/performance', methods=['GET'])
def get_performance_data():
    """Get historical performance data"""
    days = int(request.args.get('days', 30))
    
    dates = []
    intent_accuracy = []
    gnn_performance = []
    ppo_efficiency = []
    
    for i in range(days):
        date = datetime.now()
        date = date.replace(day=max(1, date.day - i))
        dates.append(date.strftime('%Y-%m-%d'))
        
        intent_accuracy.append(np.random.uniform(60, 80))
        gnn_performance.append(np.random.uniform(70, 90))
        ppo_efficiency.append(np.random.uniform(65, 85))
    
    dates.reverse()
    intent_accuracy.reverse()
    gnn_performance.reverse()
    ppo_efficiency.reverse()
    
    return jsonify({
        'success': True,
        'data': {
            'dates': dates,
            'intent_accuracy': intent_accuracy,
            'gnn_performance': gnn_performance,
            'ppo_efficiency': ppo_efficiency
        }
    })

@app.route('/api/admin/users', methods=['GET'])
def get_users():
    """Get user list"""
    return jsonify({
        'success': True,
        'users': [
            {
                'username': 'admin',
                'role': 'admin',
                'last_login': datetime.now().isoformat(),
                'status': 'online'
            },
            {
                'username': 'user',
                'role': 'user',
                'last_login': (datetime.now()).isoformat(),
                'status': 'offline'
            }
        ]
    })

if __name__ == '__main__':
    print("Starting Working SynchroChain API...")
    print("Frontend available at: http://localhost:5000")
    print("Login credentials:")
    print("  Admin: admin / admin123")
    print("  User: user / user123")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
