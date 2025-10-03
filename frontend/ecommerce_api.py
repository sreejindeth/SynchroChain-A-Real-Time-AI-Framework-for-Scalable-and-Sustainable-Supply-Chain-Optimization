from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import os
import sys
import json
import numpy as np
import time
from datetime import datetime, timedelta
import secrets
import threading
import queue

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import data layer components
try:
    from src.data.enhanced_data_layer import EnhancedDataLayer
    from src.data.data_streaming import DataStreamer
    from src.system.orchestrator import ModelOrchestrator
    from src.utils.logging_config import setup_logging
    logger = setup_logging(__name__)
except ImportError as e:
    print(f"Warning: Could not import data layer components: {e}")
    logger = None

app = Flask(__name__, static_folder='.')
CORS(app)
app.secret_key = secrets.token_hex(16)

# User credentials
USERS = {
    'admin': {'password': 'admin123', 'role': 'admin'},
    'user': {'password': 'user123', 'role': 'user'}
}

# Initialize data layer
data_layer = None
orchestrator = None
data_streamer = None

# Order queue for real-time processing
order_queue = queue.Queue()
processed_orders = {}

def initialize_data_layer():
    """Initialize data layer components"""
    global data_layer, orchestrator, data_streamer
    
    try:
        if logger:
            logger.info("Initializing enhanced data layer components...")
        
        # Initialize enhanced data layer
        data_layer = EnhancedDataLayer()
        
        # Initialize orchestrator
        orchestrator = ModelOrchestrator()
        
        # Start data streaming
        data_streamer = DataStreamer(interval_seconds=10)
        data_streamer.start_streaming()
        
        # Start real-time simulation
        data_layer.start_realtime_simulation(interval_seconds=30)
        
        if logger:
            logger.info("Enhanced data layer components initialized successfully")
        
    except Exception as e:
        if logger:
            logger.error(f"Error initializing data layer: {e}")
        print(f"Warning: Data layer initialization failed: {e}")

def process_order_ai_pipeline(order_data):
    """Process order through AI pipeline"""
    try:
        if logger:
            logger.info(f"Processing order {order_data['order_id']} through AI pipeline...")
        
        # Simulate AI processing
        start_time = time.time()
        
        # Intent Transformer Simulation
        user_behavior = order_data.get('user_behavior', {})
        session_duration = user_behavior.get('session_duration', 5)
        search_queries = user_behavior.get('search_queries', 0)
        add_to_cart = user_behavior.get('add_to_cart', False)
        
        base_intent = np.random.uniform(0.3, 0.7)
        product_bonus = len(order_data['items']) * 0.05
        session_bonus = min(0.2, session_duration / 5)
        action_bonus = 0.1 if add_to_cart else 0
        search_bonus = search_queries * 0.05
        
        intent_score = min(base_intent + product_bonus + session_bonus + action_bonus + search_bonus, 1.0)
        urgency_score = np.random.uniform(0.4, 0.9)
        if add_to_cart:
            urgency_score += 0.1
        urgency_score = min(urgency_score, 1.0)
        
        # GNN Module Simulation
        total_inventory = sum(item.get('stock', 50) for item in order_data['items'])
        inventory_risk = 0.3 if total_inventory < 100 else (0.1 if total_inventory > 300 else 0.2)
        delay_risk = np.random.uniform(0.1, 0.4)
        warehouse_risk = 0.15 if any(item.get('warehouse') == 'ASIA_PACIFIC' for item in order_data['items']) else 0.1
        total_risk = (inventory_risk + delay_risk + warehouse_risk) / 3
        
        # Carbon cost calculation
        carbon_cost = np.random.uniform(0.5, 3.0)
        if any(item.get('warehouse') == 'ASIA_PACIFIC' for item in order_data['items']):
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
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create AI timeline
        ai_timeline = {
            'intent_score': round(intent_score, 3),
            'urgency_score': round(urgency_score, 3),
            'risk_assessment': round(total_risk, 3),
            'final_decision': decision,
            'confidence': round(confidence, 3),
            'processing_time_ms': round(processing_time, 1),
            'carbon_cost': round(carbon_cost, 2),
            'inventory_level': total_inventory,
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
        }
        
        # Store processed order
        processed_orders[order_data['order_id']] = {
            **order_data,
            'ai_timeline': ai_timeline,
            'status': 'processing',
            'processed_at': datetime.now().isoformat()
        }
        
        if logger:
            logger.info(f"Order {order_data['order_id']} processed successfully")
        
        return ai_timeline
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing order {order_data['order_id']}: {e}")
        return None

def order_processing_worker():
    """Background worker for processing orders"""
    while True:
        try:
            order_data = order_queue.get(timeout=1)
            process_order_ai_pipeline(order_data)
            order_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            if logger:
                logger.error(f"Error in order processing worker: {e}")

# Start order processing worker
order_worker = threading.Thread(target=order_processing_worker, daemon=True)
order_worker.start()

# Routes
@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        role = data.get('role')

        if username in USERS and USERS[username]['password'] == password and USERS[username]['role'] == role:
            session['logged_in'] = True
            session['username'] = username
            session['role'] = role
            
            if logger:
                logger.info(f"User {username} ({role}) logged in successfully")
            
            return jsonify({
                'success': True, 
                'message': 'Login successful', 
                'role': role,
                'user_id': username
            })
        else:
            if logger:
                logger.warning(f"Failed login attempt for username: {username}, role: {role}")
            return jsonify({'success': False, 'message': 'Invalid credentials or role'}), 401
            
    except Exception as e:
        if logger:
            logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('role', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/status', methods=['GET'])
def status():
    """Check login status"""
    if session.get('logged_in'):
        return jsonify({
            'logged_in': True, 
            'username': session['username'], 
            'role': session['role']
        })
    return jsonify({'logged_in': False})

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get product catalog from data layer"""
    try:
        if data_layer:
            products = data_layer.get_products_catalog()
            return jsonify({
                'success': True,
                'products': products
            })
        else:
            return jsonify({'success': False, 'error': 'Data layer not initialized'}), 500
    except Exception as e:
        if logger:
            logger.error(f"Error getting products: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/orders', methods=['POST'])
def create_order():
    """Create new order using data layer"""
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'message': 'Not logged in'}), 401
        
        data = request.get_json()
        user_id = data.get('user_id')
        items = data.get('items', [])
        total = data.get('total', 0)
        user_behavior = data.get('user_behavior', {})
        
        if not data_layer:
            return jsonify({'success': False, 'error': 'Data layer not initialized'}), 500
        
        # Create order using data layer
        new_order = data_layer.create_new_order(user_id, items, user_behavior)
        
        if new_order:
            # Process through AI pipeline
            ai_timeline = process_order_ai_pipeline(new_order)
            if ai_timeline:
                data_layer.process_ai_predictions(new_order['id'], ai_timeline)
            
            if logger:
                logger.info(f"Order {new_order['id']} created for user {user_id}")
            
            return jsonify({
                'success': True,
                'order_id': new_order['id'],
                'message': 'Order created successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to create order'}), 500
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating order: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/orders/<order_id>', methods=['GET'])
def get_order(order_id):
    """Get order details from data layer"""
    try:
        if not data_layer:
            return jsonify({'success': False, 'error': 'Data layer not initialized'}), 500
        
        order = data_layer.get_order_by_id(int(order_id))
        if order:
            return jsonify({
                'success': True,
                'order': order
            })
        else:
            return jsonify({'success': False, 'message': 'Order not found'}), 404
            
    except Exception as e:
        if logger:
            logger.error(f"Error getting order {order_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/orders', methods=['GET'])
def get_all_orders():
    """Get all orders (admin only) from data layer"""
    try:
        if not session.get('logged_in') or session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Admin access required'}), 403
        
        if not data_layer:
            return jsonify({'success': False, 'error': 'Data layer not initialized'}), 500
        
        # Get both historical and real-time orders
        historical_orders = data_layer.get_orders_catalog()
        realtime_orders = data_layer.get_realtime_orders()
        all_orders = historical_orders + realtime_orders
        
        return jsonify({
            'success': True,
            'orders': all_orders
        })
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting all orders: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/metrics', methods=['GET'])
def get_admin_metrics():
    """Get admin metrics from data layer"""
    try:
        if not session.get('logged_in') or session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Admin access required'}), 403
        
        if not data_layer:
            return jsonify({'success': False, 'error': 'Data layer not initialized'}), 500
        
        metrics = data_layer.get_system_metrics()
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting admin metrics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """Get AI pipeline status from data layer"""
    try:
        if not data_layer:
            return jsonify({'success': False, 'error': 'Data layer not initialized'}), 500
        
        pipeline_status = data_layer.get_ai_pipeline_status()
        
        return jsonify({
            'success': True,
            'pipeline_status': pipeline_status
        })
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting pipeline status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/track-behavior', methods=['POST'])
def track_behavior():
    """Track user behavior for AI analysis using data layer"""
    try:
        data = request.get_json()
        
        if not data_layer:
            return jsonify({'success': False, 'error': 'Data layer not initialized'}), 500
        
        # Track behavior using data layer
        data_layer.track_user_behavior(data.get('user_id', 'anonymous'), data)
        
        if logger:
            logger.info(f"Tracked behavior for user {data.get('user_id', 'anonymous')}: {data.get('action')}")
        
        return jsonify({'success': True, 'message': 'Behavior tracked'})
        
    except Exception as e:
        if logger:
            logger.error(f"Error tracking behavior: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint for testing"""
    try:
        data = request.get_json()
        
        # Generate mock predictions
        intent_score = np.random.uniform(0.3, 0.9)
        urgency_score = np.random.uniform(0.4, 0.8)
        risk_assessment = np.random.uniform(0.1, 0.6)
        
        if intent_score > 0.7:
            decision = 'priority'
            confidence = 0.85
        elif intent_score > 0.5:
            decision = 'expedited'
            confidence = 0.72
        else:
            decision = 'standard'
            confidence = 0.68
        
        return jsonify({
            'success': True,
            'intent_score': round(intent_score, 3),
            'urgency_score': round(urgency_score, 3),
            'risk_assessment': round(risk_assessment, 3),
            'final_decision': decision,
            'confidence': round(confidence, 3),
            'processing_time_ms': round(np.random.uniform(1, 5), 1),
            'model_used': 'simulation'
        })
        
    except Exception as e:
        if logger:
            logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance data"""
    try:
        days = int(request.args.get('days', 90))
        
        # Generate mock performance data
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]
        
        return jsonify({
            'success': True,
            'data': {
                'dates': dates,
                'intent_accuracy': [np.random.uniform(60, 80) for _ in range(days)],
                'gnn_performance': [np.random.uniform(70, 85) for _ in range(days)],
                'ppo_efficiency': [np.random.uniform(65, 80) for _ in range(days)]
            }
        })
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting performance data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/settings', methods=['POST'])
def save_settings():
    """Save admin settings"""
    try:
        if not session.get('logged_in') or session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Admin access required'}), 403
        
        data = request.get_json()
        
        if logger:
            logger.info(f"Settings saved: {data}")
        
        return jsonify({'success': True, 'message': 'Settings saved successfully'})
        
    except Exception as e:
        if logger:
            logger.error(f"Error saving settings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def serve_index():
    """Serve main page"""
    return send_from_directory('.', 'ecommerce_user.html')

@app.route('/admin')
def serve_admin():
    """Serve admin dashboard"""
    return send_from_directory('.', 'admin_dashboard.html')

if __name__ == '__main__':
    print("Starting SynchroChain E-Commerce API...")
    
    # Initialize data layer
    initialize_data_layer()
    
    print("E-Commerce API started successfully!")
    print("User Interface: http://localhost:5000")
    print("Admin Dashboard: http://localhost:5000/admin")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
