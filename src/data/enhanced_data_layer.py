"""
Enhanced Data Layer for SynchroChain
Integrates with actual datasets and provides real-time data flow
"""

import pandas as pd
import numpy as np
import redis
import json
import time
import os
import sys
from datetime import datetime, timedelta
import threading
import queue
from typing import Dict, List, Any, Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logging_config import setup_logging
from src.data.dataset_processor import DatasetProcessor
logger = setup_logging()

class EnhancedDataLayer:
    def __init__(self, config_path='config/config.yaml'):
        self.config_path = config_path
        self.redis_client = None
        self.dataset_processor = None
        self.products_catalog = {}
        self.customers_catalog = {}
        self.orders_catalog = {}
        self.user_behavior_data = {}
        self.real_time_orders = {}
        self.feature_store = {}
        self.model_registry = {}
        
        # Initialize components
        self._initialize_redis()
        self._initialize_dataset_processor()
        self._load_processed_data()
        
        # Real-time data queues
        self.new_orders_queue = queue.Queue()
        self.user_behavior_queue = queue.Queue()
        self.ai_predictions_queue = queue.Queue()
        
        logger.info("Enhanced Data Layer initialized successfully")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            self.redis_client = None
    
    def _initialize_dataset_processor(self):
        """Initialize dataset processor"""
        try:
            self.dataset_processor = DatasetProcessor()
            logger.info("Dataset processor initialized")
        except Exception as e:
            logger.error(f"Error initializing dataset processor: {e}")
            raise
    
    def _load_processed_data(self):
        """Load processed data from dataset processor"""
        try:
            self.products_catalog = {p['id']: p for p in self.dataset_processor.get_products_catalog()}
            self.customers_catalog = {c['id']: c for c in self.dataset_processor.get_customers_catalog()}
            self.orders_catalog = {o['id']: o for o in self.dataset_processor.get_orders_catalog()}
            self.user_behavior_data = self.dataset_processor.get_user_behavior_patterns()
            
            logger.info(f"Loaded {len(self.products_catalog)} products, "
                       f"{len(self.customers_catalog)} customers, "
                       f"{len(self.orders_catalog)} orders")
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def get_products_catalog(self):
        """Get products catalog"""
        return list(self.products_catalog.values())
    
    def get_customers_catalog(self):
        """Get customers catalog"""
        return list(self.customers_catalog.values())
    
    def get_orders_catalog(self):
        """Get orders catalog"""
        return list(self.orders_catalog.values())
    
    def search_products(self, query, category=None):
        """Search products"""
        results = []
        query = query.lower()
        
        for product in self.products_catalog.values():
            if category and product['category'].lower() != category.lower():
                continue
            
            if (query in product['name'].lower() or 
                query in product['description'].lower() or 
                query in product['category'].lower()):
                results.append(product)
        
        return results
    
    def get_products_by_category(self, category):
        """Get products by category"""
        return [p for p in self.products_catalog.values() if p['category'].lower() == category.lower()]
    
    def get_categories(self):
        """Get all categories"""
        return list(set(p['category'] for p in self.products_catalog.values()))
    
    def create_new_order(self, customer_id, items, user_behavior=None):
        """Create a new order"""
        try:
            # Generate order ID
            order_id = max(self.orders_catalog.keys()) + 1 if self.orders_catalog else 1
            
            # Calculate total
            total = sum(item['price'] * item['quantity'] for item in items)
            
            # Create order
            new_order = {
                'id': order_id,
                'customer_id': customer_id,
                'items': items,
                'total': total,
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
                'user_behavior': user_behavior or {},
                'ai_timeline': None
            }
            
            # Add to orders catalog
            self.orders_catalog[order_id] = new_order
            
            # Add to real-time orders
            self.real_time_orders[order_id] = new_order
            
            # Add to processing queue
            self.new_orders_queue.put(new_order)
            
            # Store in Redis if available
            if self.redis_client:
                self.redis_client.set(f"order:{order_id}", json.dumps(new_order, default=str))
            
            logger.info(f"Created new order {order_id} for customer {customer_id}")
            return new_order
            
        except Exception as e:
            logger.error(f"Error creating new order: {e}")
            raise
    
    def track_user_behavior(self, customer_id, behavior_data):
        """Track user behavior"""
        try:
            behavior_key = f"behavior:{customer_id}:{int(time.time())}"
            
            # Store behavior data
            behavior_record = {
                'customer_id': customer_id,
                'timestamp': datetime.now().isoformat(),
                'behavior': behavior_data
            }
            
            # Add to behavior queue
            self.user_behavior_queue.put(behavior_record)
            
            # Store in Redis if available
            if self.redis_client:
                self.redis_client.set(behavior_key, json.dumps(behavior_record, default=str))
                # Set expiration (24 hours)
                self.redis_client.expire(behavior_key, 86400)
            
            logger.info(f"Tracked behavior for customer {customer_id}")
            
        except Exception as e:
            logger.error(f"Error tracking user behavior: {e}")
    
    def get_user_behavior_history(self, customer_id, hours=24):
        """Get user behavior history"""
        try:
            if not self.redis_client:
                return []
            
            # Get behavior keys for customer
            pattern = f"behavior:{customer_id}:*"
            keys = self.redis_client.keys(pattern)
            
            behaviors = []
            for key in keys:
                behavior_data = self.redis_client.get(key)
                if behavior_data:
                    behaviors.append(json.loads(behavior_data))
            
            # Filter by time
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered_behaviors = [
                b for b in behaviors 
                if datetime.fromisoformat(b['timestamp']) > cutoff_time
            ]
            
            return filtered_behaviors
            
        except Exception as e:
            logger.error(f"Error getting user behavior history: {e}")
            return []
    
    def simulate_new_order(self, customer_id=None):
        """Simulate a new order based on historical data"""
        try:
            if customer_id is None:
                customer_id = np.random.choice(list(self.customers_catalog.keys()))
            
            # Simulate user behavior
            behavior = self.dataset_processor.simulate_user_behavior(customer_id)
            
            # Select products based on behavior
            viewed_products = behavior.get('products_viewed', [])
            if not viewed_products:
                viewed_products = np.random.choice(list(self.products_catalog.keys()), 
                                                 size=np.random.randint(1, 4), 
                                                 replace=False)
            
            # Create order items
            items = []
            for product_id in viewed_products[:3]:  # Limit to 3 products
                if product_id in self.products_catalog:
                    product = self.products_catalog[product_id]
                    quantity = np.random.randint(1, 4)
                    items.append({
                        'product_id': product_id,
                        'name': product['name'],
                        'price': product['price'],
                        'quantity': quantity
                    })
            
            # Create order
            order = self.create_new_order(customer_id, items, behavior)
            
            # Track behavior
            self.track_user_behavior(customer_id, behavior)
            
            return order
            
        except Exception as e:
            logger.error(f"Error simulating new order: {e}")
            return None
    
    def process_ai_predictions(self, order_id, predictions):
        """Process AI predictions for an order"""
        try:
            if order_id in self.real_time_orders:
                self.real_time_orders[order_id]['ai_timeline'] = predictions
                self.real_time_orders[order_id]['status'] = 'processing'
                
                # Add to AI predictions queue
                self.ai_predictions_queue.put({
                    'order_id': order_id,
                    'predictions': predictions,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.set(f"ai_predictions:{order_id}", 
                                        json.dumps(predictions, default=str))
                
                logger.info(f"Processed AI predictions for order {order_id}")
                
        except Exception as e:
            logger.error(f"Error processing AI predictions: {e}")
    
    def get_order_ai_timeline(self, order_id):
        """Get AI timeline for an order"""
        try:
            if order_id in self.real_time_orders:
                return self.real_time_orders[order_id].get('ai_timeline')
            
            # Check Redis
            if self.redis_client:
                predictions = self.redis_client.get(f"ai_predictions:{order_id}")
                if predictions:
                    return json.loads(predictions)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting AI timeline for order {order_id}: {e}")
            return None
    
    def get_realtime_orders(self):
        """Get real-time orders"""
        return list(self.real_time_orders.values())
    
    def get_order_by_id(self, order_id):
        """Get order by ID"""
        if order_id in self.real_time_orders:
            return self.real_time_orders[order_id]
        elif order_id in self.orders_catalog:
            return self.orders_catalog[order_id]
        else:
            return None
    
    def update_order_status(self, order_id, status):
        """Update order status"""
        try:
            if order_id in self.real_time_orders:
                self.real_time_orders[order_id]['status'] = status
                self.real_time_orders[order_id]['updated_at'] = datetime.now().isoformat()
                
                # Update in Redis
                if self.redis_client:
                    self.redis_client.set(f"order:{order_id}", 
                                        json.dumps(self.real_time_orders[order_id], default=str))
                
                logger.info(f"Updated order {order_id} status to {status}")
                
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
    
    def get_system_metrics(self):
        """Get system metrics"""
        try:
            total_orders = len(self.orders_catalog) + len(self.real_time_orders)
            total_revenue = sum(o['total'] for o in self.orders_catalog.values()) + \
                          sum(o['total'] for o in self.real_time_orders.values())
            
            # Calculate AI metrics
            ai_orders = [o for o in self.real_time_orders.values() if o.get('ai_timeline')]
            avg_processing_time = np.mean([o['ai_timeline']['processing_time_ms'] 
                                         for o in ai_orders 
                                         if 'processing_time_ms' in o['ai_timeline']]) if ai_orders else 0
            
            return {
                'total_orders': total_orders,
                'total_revenue': total_revenue,
                'real_time_orders': len(self.real_time_orders),
                'ai_processed_orders': len(ai_orders),
                'avg_processing_time': avg_processing_time,
                'products_count': len(self.products_catalog),
                'customers_count': len(self.customers_catalog)
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def get_ai_pipeline_status(self):
        """Get AI pipeline status"""
        try:
            # Simulate pipeline status based on recent activity
            recent_orders = [o for o in self.real_time_orders.values() 
                           if datetime.fromisoformat(o['created_at']) > 
                           datetime.now() - timedelta(hours=1)]
            
            return {
                'intent_transformer': {
                    'status': 'online',
                    'accuracy': 0.69,
                    'urgency': 0.65,
                    'processing_time_ms': 2.1,
                    'recent_predictions': len(recent_orders)
                },
                'gnn_module': {
                    'status': 'online',
                    'mae': 0.045,
                    'risk': 0.25,
                    'processing_time_ms': 1.8,
                    'recent_predictions': len(recent_orders)
                },
                'rl_agent': {
                    'status': 'online',
                    'confidence': 0.78,
                    'improvement': 0.213,
                    'processing_time_ms': 1.2,
                    'recent_predictions': len(recent_orders)
                },
                'orchestrator': {
                    'status': 'online',
                    'throughput_per_second': 357,
                    'processing_time_ms': 0.5,
                    'recent_predictions': len(recent_orders)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting AI pipeline status: {e}")
            return {}
    
    def store_behavior_data(self, key, data):
        """Store behavior data in Redis"""
        try:
            if self.redis_client:
                self.redis_client.set(key, json.dumps(data, default=str))
                # Set expiration (24 hours)
                self.redis_client.expire(key, 86400)
        except Exception as e:
            logger.error(f"Error storing behavior data: {e}")
    
    def get_behavior_data(self, key):
        """Get behavior data from Redis"""
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting behavior data: {e}")
            return None
    
    def start_realtime_simulation(self, interval_seconds=30):
        """Start real-time simulation of new orders"""
        def simulation_loop():
            while True:
                try:
                    # Simulate new order
                    new_order = self.simulate_new_order()
                    if new_order:
                        logger.info(f"Simulated new order: {new_order['id']}")
                    
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in simulation loop: {e}")
                    time.sleep(interval_seconds)
        
        # Start simulation in background thread
        simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
        simulation_thread.start()
        logger.info(f"Started real-time simulation with {interval_seconds}s interval")

if __name__ == '__main__':
    # Test the enhanced data layer
    data_layer = EnhancedDataLayer()
    
    print("Enhanced Data Layer Test")
    print(f"Products: {len(data_layer.get_products_catalog())}")
    print(f"Customers: {len(data_layer.get_customers_catalog())}")
    print(f"Categories: {data_layer.get_categories()}")
    
    # Test search
    search_results = data_layer.search_products('watch')
    print(f"Search results for 'watch': {len(search_results)}")
    
    # Test new order
    new_order = data_layer.simulate_new_order()
    print(f"Simulated new order: {new_order['id'] if new_order else 'Failed'}")
    
    # Test metrics
    metrics = data_layer.get_system_metrics()
    print(f"System metrics: {metrics}")
    
    # Test AI pipeline status
    pipeline_status = data_layer.get_ai_pipeline_status()
    print(f"AI pipeline status: {pipeline_status['intent_transformer']['status']}")
