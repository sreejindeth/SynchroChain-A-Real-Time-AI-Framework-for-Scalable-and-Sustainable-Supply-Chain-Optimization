"""
Dataset Processor for SynchroChain
Processes raw datasets and creates real-time data flow
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import random
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logging_config import setup_logging
logger = setup_logging()

class DatasetProcessor:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.supply_chain_df = None
        self.access_logs_df = None
        self.products_catalog = {}
        self.customers_catalog = {}
        self.orders_catalog = {}
        
        # Load datasets
        self.load_datasets()
        self.process_datasets()
    
    def load_datasets(self):
        """Load raw datasets"""
        try:
            logger.info("Loading raw datasets...")
            
            # Load supply chain dataset
            supply_chain_path = os.path.join(self.data_dir, 'DataCoSupplyChainDataset.csv')
            if os.path.exists(supply_chain_path):
                self.supply_chain_df = pd.read_csv(supply_chain_path, encoding='latin-1')
                logger.info(f"Loaded supply chain dataset: {len(self.supply_chain_df)} records")
            else:
                logger.warning("Supply chain dataset not found")
            
            # Load access logs dataset
            access_logs_path = os.path.join(self.data_dir, 'tokenized_access_logs.csv')
            if os.path.exists(access_logs_path):
                self.access_logs_df = pd.read_csv(access_logs_path)
                logger.info(f"Loaded access logs dataset: {len(self.access_logs_df)} records")
            else:
                logger.warning("Access logs dataset not found")
                
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def process_datasets(self):
        """Process datasets to create catalogs"""
        try:
            logger.info("Processing datasets to create catalogs...")
            
            if self.supply_chain_df is not None:
                self._process_supply_chain_data()
            
            if self.access_logs_df is not None:
                self._process_access_logs_data()
                
            logger.info("Dataset processing completed")
            
        except Exception as e:
            logger.error(f"Error processing datasets: {e}")
            raise
    
    def _process_supply_chain_data(self):
        """Process supply chain dataset"""
        df = self.supply_chain_df
        
        # Create products catalog
        products = df[['Product Card Id', 'Product Name', 'Product Price', 'Product Category Id', 
                      'Category Name', 'Product Description', 'Product Image']].drop_duplicates()
        
        for _, row in products.iterrows():
            product_id = int(row['Product Card Id'])
            self.products_catalog[product_id] = {
                'id': product_id,
                'name': row['Product Name'],
                'price': float(row['Product Price']),
                'category_id': int(row['Product Category Id']),
                'category': row['Category Name'],
                'description': row['Product Description'] if pd.notna(row['Product Description']) else '',
                'image': row['Product Image'] if pd.notna(row['Product Image']) else '',
                'warehouse': self._get_warehouse_from_region(row.get('Order Region', 'US_WEST')),
                'stock': random.randint(10, 100)  # Simulate stock levels
            }
        
        # Create customers catalog
        customers = df[['Customer Id', 'Customer Fname', 'Customer Lname', 'Customer Email', 
                       'Customer City', 'Customer Country', 'Customer Segment']].drop_duplicates()
        
        for _, row in customers.iterrows():
            customer_id = int(row['Customer Id'])
            self.customers_catalog[customer_id] = {
                'id': customer_id,
                'first_name': row['Customer Fname'],
                'last_name': row['Customer Lname'],
                'email': row['Customer Email'],
                'city': row['Customer City'],
                'country': row['Customer Country'],
                'segment': row['Customer Segment']
            }
        
        # Create orders catalog (sample)
        orders = df[['Order Id', 'Order Customer Id', 'order date (DateOrders)', 
                    'Order Status', 'Order Item Total', 'Product Card Id', 
                    'Order Item Quantity', 'Late_delivery_risk', 'Shipping Mode']].drop_duplicates()
        
        for _, row in orders.iterrows():
            order_id = int(row['Order Id'])
            if order_id not in self.orders_catalog:
                self.orders_catalog[order_id] = {
                    'id': order_id,
                    'customer_id': int(row['Order Customer Id']),
                    'order_date': row['order date (DateOrders)'],
                    'status': row['Order Status'],
                    'total': float(row['Order Item Total']),
                    'items': [],
                    'late_delivery_risk': int(row['Late_delivery_risk']),
                    'shipping_mode': row['Shipping Mode']
                }
            
            # Add item to order
            product_id = int(row['Product Card Id'])
            if product_id in self.products_catalog:
                self.orders_catalog[order_id]['items'].append({
                    'product_id': product_id,
                    'quantity': int(row['Order Item Quantity']),
                    'price': self.products_catalog[product_id]['price']
                })
        
        logger.info(f"Created catalogs: {len(self.products_catalog)} products, "
                   f"{len(self.customers_catalog)} customers, {len(self.orders_catalog)} orders")
    
    def _process_access_logs_data(self):
        """Process access logs dataset"""
        df = self.access_logs_df
        
        # Extract user behavior patterns
        self.user_behavior_patterns = {}
        
        # Group by IP to simulate user sessions
        for ip, group in df.groupby('ip'):
            sessions = self._create_user_sessions(group)
            self.user_behavior_patterns[ip] = sessions
        
        logger.info(f"Processed {len(self.user_behavior_patterns)} user behavior patterns")
    
    def _create_user_sessions(self, user_data):
        """Create user sessions from access logs"""
        sessions = []
        current_session = {
            'session_id': f"session_{int(time.time())}_{random.randint(1000, 9999)}",
            'start_time': None,
            'end_time': None,
            'products_viewed': [],
            'categories_viewed': [],
            'search_queries': 0,
            'add_to_cart': False,
            'purchase_intent': 0.0
        }
        
        for _, row in user_data.iterrows():
            if current_session['start_time'] is None:
                current_session['start_time'] = row['Date']
            
            current_session['end_time'] = row['Date']
            current_session['products_viewed'].append(row['Product'])
            current_session['categories_viewed'].append(row['Category'])
            
            # Simulate search behavior
            if 'search' in row['url'].lower():
                current_session['search_queries'] += 1
            
            # Simulate add to cart (random based on product views)
            if random.random() < 0.1:  # 10% chance
                current_session['add_to_cart'] = True
            
            # Calculate purchase intent based on session length and actions
            session_duration = len(current_session['products_viewed'])
            current_session['purchase_intent'] = min(0.9, session_duration * 0.1 + 
                                                   current_session['search_queries'] * 0.05)
        
        sessions.append(current_session)
        return sessions
    
    def _get_warehouse_from_region(self, region):
        """Map region to warehouse"""
        region_mapping = {
            'Southeast Asia': 'ASIA_PACIFIC',
            'South Asia': 'ASIA_PACIFIC',
            'Eastern Asia': 'ASIA_PACIFIC',
            'Oceania': 'ASIA_PACIFIC',
            'Pacific Asia': 'ASIA_PACIFIC',
            'Europe': 'EUROPE',
            'North America': 'US_WEST',
            'US': 'US_WEST'
        }
        return region_mapping.get(region, 'US_WEST')
    
    def get_products_catalog(self):
        """Get products catalog"""
        return list(self.products_catalog.values())
    
    def get_customers_catalog(self):
        """Get customers catalog"""
        return list(self.customers_catalog.values())
    
    def get_orders_catalog(self):
        """Get orders catalog"""
        return list(self.orders_catalog.values())
    
    def get_user_behavior_patterns(self):
        """Get user behavior patterns"""
        return self.user_behavior_patterns
    
    def simulate_new_order(self, customer_id=None, products=None):
        """Simulate a new order based on historical data"""
        if customer_id is None:
            customer_id = random.choice(list(self.customers_catalog.keys()))
        
        if products is None:
            # Select random products
            num_products = random.randint(1, 3)
            products = random.sample(list(self.products_catalog.keys()), num_products)
        
        # Create new order
        order_id = max(self.orders_catalog.keys()) + 1 if self.orders_catalog else 1
        order_date = datetime.now().strftime('%m/%d/%Y %H:%M')
        
        new_order = {
            'id': order_id,
            'customer_id': customer_id,
            'order_date': order_date,
            'status': 'pending',
            'total': 0.0,
            'items': [],
            'late_delivery_risk': random.randint(0, 1),
            'shipping_mode': random.choice(['Standard Class', 'First Class', 'Second Class'])
        }
        
        # Add products to order
        for product_id in products:
            if product_id in self.products_catalog:
                product = self.products_catalog[product_id]
                quantity = random.randint(1, 3)
                item_total = product['price'] * quantity
                
                new_order['items'].append({
                    'product_id': product_id,
                    'quantity': quantity,
                    'price': product['price']
                })
                new_order['total'] += item_total
        
        return new_order
    
    def simulate_user_behavior(self, customer_id=None):
        """Simulate user behavior based on historical patterns"""
        if customer_id is None:
            customer_id = random.choice(list(self.customers_catalog.keys()))
        
        # Get random behavior pattern
        if self.user_behavior_patterns:
            random_ip = random.choice(list(self.user_behavior_patterns.keys()))
            behavior_patterns = self.user_behavior_patterns[random_ip]
            session = random.choice(behavior_patterns)
        else:
            # Create default behavior
            session = {
                'session_id': f"session_{int(time.time())}_{random.randint(1000, 9999)}",
                'start_time': datetime.now().strftime('%m/%d/%Y %H:%M'),
                'end_time': datetime.now().strftime('%m/%d/%Y %H:%M'),
                'products_viewed': random.sample(list(self.products_catalog.keys()), random.randint(1, 5)),
                'categories_viewed': random.sample(list(set(p['category'] for p in self.products_catalog.values())), random.randint(1, 3)),
                'search_queries': random.randint(0, 5),
                'add_to_cart': random.choice([True, False]),
                'purchase_intent': random.uniform(0.1, 0.9)
            }
        
        return {
            'customer_id': customer_id,
            'session_id': session['session_id'],
            'session_duration': random.randint(1, 30),  # minutes
            'products_viewed': session['products_viewed'][:3],  # Limit to 3 products
            'categories_viewed': session['categories_viewed'][:2],  # Limit to 2 categories
            'search_queries': session['search_queries'],
            'add_to_cart': session['add_to_cart'],
            'purchase_intent': session['purchase_intent'],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_products_by_category(self, category):
        """Get products by category"""
        return [p for p in self.products_catalog.values() if p['category'].lower() == category.lower()]
    
    def search_products(self, query):
        """Search products by name or description"""
        query = query.lower()
        results = []
        
        for product in self.products_catalog.values():
            if (query in product['name'].lower() or 
                query in product['description'].lower() or 
                query in product['category'].lower()):
                results.append(product)
        
        return results
    
    def get_categories(self):
        """Get all product categories"""
        return list(set(p['category'] for p in self.products_catalog.values()))
    
    def save_processed_data(self, output_dir='data/processed'):
        """Save processed data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save products catalog
        with open(os.path.join(output_dir, 'products_catalog.json'), 'w') as f:
            json.dump(self.products_catalog, f, indent=2, default=str)
        
        # Save customers catalog
        with open(os.path.join(output_dir, 'customers_catalog.json'), 'w') as f:
            json.dump(self.customers_catalog, f, indent=2, default=str)
        
        # Save orders catalog
        with open(os.path.join(output_dir, 'orders_catalog.json'), 'w') as f:
            json.dump(self.orders_catalog, f, indent=2, default=str)
        
        logger.info(f"Processed data saved to {output_dir}")

if __name__ == '__main__':
    # Test the dataset processor
    processor = DatasetProcessor()
    
    print(f"Products: {len(processor.get_products_catalog())}")
    print(f"Customers: {len(processor.get_customers_catalog())}")
    print(f"Orders: {len(processor.get_orders_catalog())}")
    print(f"Categories: {processor.get_categories()}")
    
    # Test search
    search_results = processor.search_products('watch')
    print(f"Search results for 'watch': {len(search_results)}")
    
    # Test new order simulation
    new_order = processor.simulate_new_order()
    print(f"Simulated new order: {new_order['id']} with {len(new_order['items'])} items")
    
    # Test user behavior simulation
    behavior = processor.simulate_user_behavior()
    print(f"Simulated user behavior: {behavior['session_id']}")
