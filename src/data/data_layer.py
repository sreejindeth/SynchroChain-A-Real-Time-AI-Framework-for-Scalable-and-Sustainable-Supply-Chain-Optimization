"""
SynchroChain Data Layer Architecture
Complete implementation of the data pipeline:
Live DBs → Real-time Cache → Processed Logs → Feature Store/Model Registry
"""

import redis
import sqlite3
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import asyncio
import aiofiles

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data layer components"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    db_path: str = "data/live_databases"
    cache_ttl: int = 3600  # 1 hour
    batch_size: int = 1000
    processing_interval: int = 60  # 1 minute

class LiveTransactionalDB:
    """Simulates live transactional databases (Orders, Users, Inventory)"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self._init_databases()
        
    def _init_databases(self):
        """Initialize SQLite databases for different data types"""
        
        # Orders Database
        self.orders_db = sqlite3.connect(self.db_path / "orders.db")
        self.orders_db.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                product_id TEXT,
                quantity INTEGER,
                price DECIMAL(10,2),
                warehouse_id TEXT,
                shipping_method TEXT,
                order_timestamp TIMESTAMP,
                status TEXT
            )
        """)
        
        # Users Database
        self.users_db = sqlite3.connect(self.db_path / "users.db")
        self.users_db.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER,
                products_viewed TEXT,
                actions_performed TEXT,
                session_duration INTEGER,
                timestamp TIMESTAMP,
                intent_signals TEXT
            )
        """)
        
        # Inventory Database
        self.inventory_db = sqlite3.connect(self.db_path / "inventory.db")
        self.inventory_db.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                product_id TEXT,
                warehouse_id TEXT,
                current_stock INTEGER,
                reserved_stock INTEGER,
                last_updated TIMESTAMP,
                carbon_footprint DECIMAL(5,2),
                PRIMARY KEY (product_id, warehouse_id)
            )
        """)
        
        # Supply Chain Database
        self.supply_db = sqlite3.connect(self.db_path / "supply_chain.db")
        self.supply_db.execute("""
            CREATE TABLE IF NOT EXISTS shipping_routes (
                route_id TEXT PRIMARY KEY,
                from_warehouse TEXT,
                to_location TEXT,
                estimated_days INTEGER,
                delay_risk DECIMAL(3,2),
                carbon_cost DECIMAL(5,2),
                last_updated TIMESTAMP
            )
        """)
        
        logger.info("Live transactional databases initialized")
    
    def generate_sample_data(self):
        """Generate sample transactional data"""
        
        # Sample orders
        orders_data = []
        for i in range(100):
            orders_data.append((
                i + 1,
                np.random.randint(1, 50),
                f"product_{np.random.randint(1, 20)}",
                np.random.randint(1, 5),
                round(np.random.uniform(10, 500), 2),
                f"warehouse_{np.random.choice(['US_WEST', 'US_EAST', 'EU_CENTRAL', 'ASIA_PACIFIC'])}",
                np.random.choice(['standard', 'expedited', 'priority']),
                datetime.now() - timedelta(hours=np.random.randint(0, 24)),
                np.random.choice(['pending', 'processing', 'shipped', 'delivered'])
            ))
        
        self.orders_db.executemany(
            "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            orders_data
        )
        
        # Sample user sessions
        sessions_data = []
        for i in range(50):
            products = json.dumps([f"product_{j}" for j in np.random.randint(1, 20, 3)])
            actions = json.dumps(np.random.choice(['view', 'cart', 'search'], 5).tolist())
            intent_signals = json.dumps({
                'time_on_page': np.random.uniform(10, 300),
                'scroll_depth': np.random.uniform(0.1, 1.0),
                'click_rate': np.random.uniform(0.01, 0.1)
            })
            
            sessions_data.append((
                f"session_{i}",
                np.random.randint(1, 50),
                products,
                actions,
                np.random.randint(60, 1800),
                datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
                intent_signals
            ))
        
        self.users_db.executemany(
            "INSERT OR REPLACE INTO user_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            sessions_data
        )
        
        # Sample inventory
        inventory_data = []
        warehouses = ['US_WEST', 'US_EAST', 'EU_CENTRAL', 'ASIA_PACIFIC']
        for product_id in range(1, 21):
            for warehouse in warehouses:
                inventory_data.append((
                    f"product_{product_id}",
                    warehouse,
                    np.random.randint(0, 500),
                    np.random.randint(0, 50),
                    datetime.now(),
                    round(np.random.uniform(0.1, 2.0), 2)
                ))
        
        self.inventory_db.executemany(
            "INSERT OR REPLACE INTO inventory VALUES (?, ?, ?, ?, ?, ?)",
            inventory_data
        )
        
        # Sample shipping routes
        routes_data = []
        for i, warehouse in enumerate(warehouses):
            routes_data.append((
                f"route_{i}",
                warehouse,
                "customer_location",
                np.random.randint(1, 7),
                round(np.random.uniform(0.05, 0.3), 2),
                round(np.random.uniform(0.5, 3.0), 2),
                datetime.now()
            ))
        
        self.supply_db.executemany(
            "INSERT OR REPLACE INTO shipping_routes VALUES (?, ?, ?, ?, ?, ?, ?)",
            routes_data
        )
        
        # Commit all changes
        self.orders_db.commit()
        self.users_db.commit()
        self.inventory_db.commit()
        self.supply_db.commit()
        
        logger.info("Sample transactional data generated")

class RealTimeCache:
    """Redis-based real-time cache for streaming data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        try:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except redis.ConnectionError:
            logger.warning("Redis not available, using in-memory cache")
            self.redis_client = None
            self.memory_cache = {}
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        if self.redis_client:
            self.redis_client.setex(key, ttl or self.config.cache_ttl, json.dumps(value))
        else:
            self.memory_cache[key] = {
                'value': value,
                'expires': time.time() + (ttl or self.config.cache_ttl)
            }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.redis_client:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        else:
            if key in self.memory_cache:
                item = self.memory_cache[key]
                if time.time() < item['expires']:
                    return item['value']
                else:
                    del self.memory_cache[key]
            return None
    
    def stream_data(self, data_type: str, data: Dict[str, Any]):
        """Stream data to cache in real-time"""
        key = f"stream:{data_type}:{int(time.time() * 1000)}"
        self.set(key, data, ttl=300)  # 5 minutes TTL for streaming data
        
        # Also maintain latest data
        latest_key = f"latest:{data_type}"
        self.set(latest_key, data, ttl=self.config.cache_ttl)

class DataProcessor:
    """Processes raw data into structured logs and features"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def process_orders(self, raw_orders: List[Dict]) -> pd.DataFrame:
        """Process raw order data"""
        df = pd.DataFrame(raw_orders)
        
        # Feature engineering
        df['order_hour'] = pd.to_datetime(df['order_timestamp']).dt.hour
        df['order_day'] = pd.to_datetime(df['order_timestamp']).dt.dayofweek
        df['total_value'] = df['quantity'] * df['price']
        df['is_high_value'] = df['total_value'] > df['total_value'].quantile(0.8)
        
        return df
    
    def process_user_sessions(self, raw_sessions: List[Dict]) -> pd.DataFrame:
        """Process raw user session data"""
        df = pd.DataFrame(raw_sessions)
        
        # Parse JSON fields
        df['products_viewed'] = df['products_viewed'].apply(json.loads)
        df['actions_performed'] = df['actions_performed'].apply(json.loads)
        df['intent_signals'] = df['intent_signals'].apply(json.loads)
        
        # Feature engineering
        df['num_products_viewed'] = df['products_viewed'].apply(len)
        df['num_actions'] = df['actions_performed'].apply(len)
        df['has_cart_action'] = df['actions_performed'].apply(
            lambda x: 'cart' in x
        )
        df['intent_score'] = df['intent_signals'].apply(
            lambda x: min(1.0, (x['time_on_page'] / 300) * x['scroll_depth'] * 10)
        )
        
        return df
    
    def process_inventory(self, raw_inventory: List[Dict]) -> pd.DataFrame:
        """Process raw inventory data"""
        df = pd.DataFrame(raw_inventory)
        
        # Feature engineering
        df['stock_level'] = df['current_stock'] - df['reserved_stock']
        df['stock_ratio'] = df['current_stock'] / (df['current_stock'] + df['reserved_stock'])
        df['is_low_stock'] = df['stock_level'] < 50
        df['carbon_intensity'] = df['carbon_footprint'] / df['current_stock']
        
        return df
    
    def save_processed_data(self, data_type: str, df: pd.DataFrame):
        """Save processed data to storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.processed_path / f"{data_type}_{timestamp}.parquet"
        df.to_parquet(file_path, index=False)
        logger.info(f"Processed {data_type} data saved to {file_path}")

class FeatureStore:
    """Feature store for ML model features and model registry"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_path = Path("data/feature_store")
        self.feature_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature store database
        self.feature_db = sqlite3.connect(self.feature_path / "feature_store.db")
        self.feature_db.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id TEXT PRIMARY KEY,
                feature_name TEXT,
                feature_type TEXT,
                feature_value TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Model registry
        self.feature_db.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id TEXT PRIMARY KEY,
                model_name TEXT,
                model_version TEXT,
                model_path TEXT,
                performance_metrics TEXT,
                created_at TIMESTAMP,
                status TEXT
            )
        """)
        
        logger.info("Feature store initialized")
    
    def store_feature(self, feature_id: str, feature_name: str, 
                     feature_type: str, feature_value: Any, metadata: Dict = None):
        """Store a feature in the feature store"""
        self.feature_db.execute("""
            INSERT OR REPLACE INTO features 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            feature_id,
            feature_name,
            feature_type,
            json.dumps(feature_value),
            datetime.now(),
            datetime.now(),
            json.dumps(metadata or {})
        ))
        self.feature_db.commit()
    
    def get_feature(self, feature_id: str) -> Optional[Dict]:
        """Retrieve a feature from the feature store"""
        cursor = self.feature_db.execute(
            "SELECT * FROM features WHERE feature_id = ?", (feature_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'feature_id': row[0],
                'feature_name': row[1],
                'feature_type': row[2],
                'feature_value': json.loads(row[3]),
                'created_at': row[4],
                'updated_at': row[5],
                'metadata': json.loads(row[6])
            }
        return None
    
    def register_model(self, model_id: str, model_name: str, model_version: str,
                     model_path: str, performance_metrics: Dict, status: str = "active"):
        """Register a model in the model registry"""
        self.feature_db.execute("""
            INSERT OR REPLACE INTO model_registry 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            model_name,
            model_version,
            model_path,
            json.dumps(performance_metrics),
            datetime.now(),
            status
        ))
        self.feature_db.commit()
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """Retrieve model information from registry"""
        cursor = self.feature_db.execute(
            "SELECT * FROM model_registry WHERE model_id = ?", (model_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'model_id': row[0],
                'model_name': row[1],
                'model_version': row[2],
                'model_path': row[3],
                'performance_metrics': json.loads(row[4]),
                'created_at': row[5],
                'status': row[6]
            }
        return None

class DataLayerOrchestrator:
    """Orchestrates the complete data layer pipeline"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.live_db = LiveTransactionalDB(config)
        self.cache = RealTimeCache(config)
        self.processor = DataProcessor(config)
        self.feature_store = FeatureStore(config)
        
        self.running = False
        self.processing_thread = None
        
        logger.info("Data layer orchestrator initialized")
    
    def start_data_pipeline(self):
        """Start the complete data pipeline"""
        self.running = True
        
        # Generate initial sample data
        self.live_db.generate_sample_data()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start data streaming
        self._start_data_streaming()
        
        logger.info("Data pipeline started")
    
    def stop_data_pipeline(self):
        """Stop the data pipeline"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Data pipeline stopped")
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Extract data from live databases
                orders_data = self._extract_orders_data()
                sessions_data = self._extract_sessions_data()
                inventory_data = self._extract_inventory_data()
                
                # Process data
                processed_orders = self.processor.process_orders(orders_data)
                processed_sessions = self.processor.process_user_sessions(sessions_data)
                processed_inventory = self.processor.process_inventory(inventory_data)
                
                # Save processed data
                self.processor.save_processed_data("orders", processed_orders)
                self.processor.save_processed_data("sessions", processed_sessions)
                self.processor.save_processed_data("inventory", processed_inventory)
                
                # Store features in feature store
                self._store_features(processed_orders, processed_sessions, processed_inventory)
                
                # Update model registry
                self._update_model_registry()
                
                time.sleep(self.config.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(10)
    
    def _extract_orders_data(self) -> List[Dict]:
        """Extract orders data from live database"""
        cursor = self.live_db.orders_db.execute("SELECT * FROM orders")
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def _extract_sessions_data(self) -> List[Dict]:
        """Extract user sessions data from live database"""
        cursor = self.live_db.users_db.execute("SELECT * FROM user_sessions")
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def _extract_inventory_data(self) -> List[Dict]:
        """Extract inventory data from live database"""
        cursor = self.live_db.inventory_db.execute("SELECT * FROM inventory")
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def _store_features(self, orders_df: pd.DataFrame, sessions_df: pd.DataFrame, inventory_df: pd.DataFrame):
        """Store computed features in feature store"""
        
        # Order features
        self.feature_store.store_feature(
            "avg_order_value",
            "Average Order Value",
            "numeric",
            float(orders_df['total_value'].mean()),
            {"source": "orders", "computed_at": datetime.now().isoformat()}
        )
        
        # Session features
        self.feature_store.store_feature(
            "avg_intent_score",
            "Average Intent Score",
            "numeric",
            float(sessions_df['intent_score'].mean()),
            {"source": "sessions", "computed_at": datetime.now().isoformat()}
        )
        
        # Inventory features
        self.feature_store.store_feature(
            "total_inventory_value",
            "Total Inventory Value",
            "numeric",
            float(inventory_df['stock_level'].sum()),
            {"source": "inventory", "computed_at": datetime.now().isoformat()}
        )
    
    def _update_model_registry(self):
        """Update model registry with current model information"""
        
        # Register Intent Transformer
        self.feature_store.register_model(
            "intent_transformer_v1",
            "Intent Transformer",
            "1.0",
            "models/intent_transformer_finetuned_multi.pth",
            {"accuracy": 0.69, "mae": 0.12, "training_time": "75min"},
            "active"
        )
        
        # Register GNN Model
        self.feature_store.register_model(
            "gnn_enhanced_v1",
            "Enhanced GNN",
            "1.0",
            "models/saved/enhanced_gnn_enhanced_best.pth",
            {"mae": 0.045, "model_size": "2.1MB", "training_time": "120min"},
            "active"
        )
        
        # Register PPO Agent
        self.feature_store.register_model(
            "ppo_agent_v1",
            "PPO Agent",
            "1.0",
            "models/ppo_agent.pth",
            {"improvement": 0.213, "cost_efficiency": 3.118, "training_time": "90min"},
            "active"
        )
    
    def _start_data_streaming(self):
        """Start streaming data to real-time cache"""
        def stream_loop():
            while self.running:
                try:
                    # Stream latest orders
                    latest_orders = self._extract_orders_data()[-10:]  # Last 10 orders
                    self.cache.stream_data("orders", {"data": latest_orders, "timestamp": datetime.now().isoformat()})
                    
                    # Stream latest sessions
                    latest_sessions = self._extract_sessions_data()[-10:]  # Last 10 sessions
                    self.cache.stream_data("sessions", {"data": latest_sessions, "timestamp": datetime.now().isoformat()})
                    
                    # Stream latest inventory
                    latest_inventory = self._extract_inventory_data()
                    self.cache.stream_data("inventory", {"data": latest_inventory, "timestamp": datetime.now().isoformat()})
                    
                    time.sleep(30)  # Stream every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in streaming: {e}")
                    time.sleep(10)
        
        streaming_thread = threading.Thread(target=stream_loop)
        streaming_thread.daemon = True
        streaming_thread.start()
    
    def get_latest_features(self) -> Dict[str, Any]:
        """Get latest features from cache"""
        features = {}
        
        # Get from cache
        orders_data = self.cache.get("latest:orders")
        sessions_data = self.cache.get("latest:sessions")
        inventory_data = self.cache.get("latest:inventory")
        
        if orders_data:
            features['orders'] = orders_data
        if sessions_data:
            features['sessions'] = sessions_data
        if inventory_data:
            features['inventory'] = inventory_data
        
        return features

# Example usage and testing
if __name__ == "__main__":
    # Initialize data layer
    config = DataConfig()
    orchestrator = DataLayerOrchestrator(config)
    
    # Start the complete data pipeline
    orchestrator.start_data_pipeline()
    
    try:
        # Let it run for a bit
        time.sleep(60)
        
        # Get latest features
        features = orchestrator.get_latest_features()
        print("Latest features:", json.dumps(features, indent=2, default=str))
        
    finally:
        orchestrator.stop_data_pipeline()

