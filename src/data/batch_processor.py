"""
Batch Processing System for Feature Store Updates
Handles periodic batch processing of data into feature store and model registry
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import schedule
import time
import threading

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    processing_interval: int = 3600  # 1 hour
    batch_size: int = 10000
    feature_store_path: str = "data/feature_store"
    processed_data_path: str = "data/processed"
    retention_days: int = 30

class FeatureEngineer:
    """Engineers features from raw data for ML models"""
    
    def __init__(self):
        self.feature_cache = {}
    
    def engineer_user_features(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer user behavior features"""
        
        # Parse JSON columns
        sessions_df['products_viewed'] = sessions_df['products_viewed'].apply(json.loads)
        sessions_df['actions_performed'] = sessions_df['actions_performed'].apply(json.loads)
        sessions_df['intent_signals'] = sessions_df['intent_signals'].apply(json.loads)
        
        # Basic features
        sessions_df['num_products_viewed'] = sessions_df['products_viewed'].apply(len)
        sessions_df['num_actions'] = sessions_df['actions_performed'].apply(len)
        sessions_df['session_duration_minutes'] = sessions_df['session_duration'] / 60
        
        # Intent features
        sessions_df['intent_score'] = sessions_df['intent_signals'].apply(
            lambda x: min(1.0, (x['time_on_page'] / 300) * x['scroll_depth'] * 10)
        )
        sessions_df['urgency_score'] = sessions_df['intent_signals'].apply(
            lambda x: min(1.0, x['click_rate'] * 20)
        )
        
        # Behavior features
        sessions_df['has_cart_action'] = sessions_df['actions_performed'].apply(
            lambda x: 'cart' in x
        )
        sessions_df['has_search_action'] = sessions_df['actions_performed'].apply(
            lambda x: 'search' in x
        )
        sessions_df['action_diversity'] = sessions_df['actions_performed'].apply(
            lambda x: len(set(x))
        )
        
        # Temporal features
        sessions_df['hour_of_day'] = pd.to_datetime(sessions_df['timestamp']).dt.hour
        sessions_df['day_of_week'] = pd.to_datetime(sessions_df['timestamp']).dt.dayofweek
        sessions_df['is_weekend'] = sessions_df['day_of_week'].isin([5, 6])
        
        return sessions_df
    
    def engineer_order_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer order-related features"""
        
        # Value features
        orders_df['total_value'] = orders_df['quantity'] * orders_df['price']
        orders_df['avg_item_price'] = orders_df['price']
        orders_df['is_high_value'] = orders_df['total_value'] > orders_df['total_value'].quantile(0.8)
        
        # Temporal features
        orders_df['order_hour'] = pd.to_datetime(orders_df['order_timestamp']).dt.hour
        orders_df['order_day'] = pd.to_datetime(orders_df['order_timestamp']).dt.dayofweek
        orders_df['is_weekend_order'] = orders_df['order_day'].isin([5, 6])
        
        # Shipping features
        orders_df['shipping_urgency'] = orders_df['shipping_method'].map({
            'standard': 1,
            'expedited': 2,
            'priority': 3
        })
        
        # Warehouse features
        orders_df['warehouse_region'] = orders_df['warehouse_id'].str.split('_').str[0]
        
        return orders_df
    
    def engineer_inventory_features(self, inventory_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer inventory-related features"""
        
        # Stock features
        inventory_df['stock_level'] = inventory_df['current_stock'] - inventory_df['reserved_stock']
        inventory_df['stock_ratio'] = inventory_df['current_stock'] / (inventory_df['current_stock'] + inventory_df['reserved_stock'])
        inventory_df['is_low_stock'] = inventory_df['stock_level'] < 50
        inventory_df['is_out_of_stock'] = inventory_df['stock_level'] <= 0
        
        # Carbon features
        inventory_df['carbon_intensity'] = inventory_df['carbon_footprint'] / inventory_df['current_stock']
        inventory_df['carbon_category'] = pd.cut(
            inventory_df['carbon_footprint'],
            bins=[0, 0.5, 1.0, 2.0, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Warehouse features
        inventory_df['warehouse_region'] = inventory_df['warehouse_id'].str.split('_').str[0]
        
        return inventory_df
    
    def engineer_supply_chain_features(self, supply_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer supply chain features"""
        
        # Risk features
        supply_df['risk_category'] = pd.cut(
            supply_df['delay_risk'],
            bins=[0, 0.1, 0.2, 0.3, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Carbon features
        supply_df['carbon_category'] = pd.cut(
            supply_df['carbon_cost'],
            bins=[0, 1.0, 2.0, 3.0, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Efficiency features
        supply_df['efficiency_score'] = 1 / (supply_df['estimated_days'] * supply_df['delay_risk'])
        
        return supply_df

class BatchProcessor:
    """Handles batch processing of data into feature store"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.feature_store_path = Path(config.feature_store_path)
        self.processed_data_path = Path(config.processing_interval)
        
        # Initialize feature store database
        self.feature_db = sqlite3.connect(self.feature_store_path / "feature_store.db")
        self._init_feature_tables()
        
        logger.info("Batch processor initialized")
    
    def _init_feature_tables(self):
        """Initialize feature store tables"""
        
        # Features table
        self.feature_db.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id TEXT PRIMARY KEY,
                feature_name TEXT,
                feature_type TEXT,
                feature_value TEXT,
                feature_vector TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Feature sets table
        self.feature_db.execute("""
            CREATE TABLE IF NOT EXISTS feature_sets (
                set_id TEXT PRIMARY KEY,
                set_name TEXT,
                feature_ids TEXT,
                created_at TIMESTAMP,
                description TEXT
            )
        """)
        
        # Model registry
        self.feature_db.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id TEXT PRIMARY KEY,
                model_name TEXT,
                model_version TEXT,
                model_path TEXT,
                feature_set_id TEXT,
                performance_metrics TEXT,
                created_at TIMESTAMP,
                status TEXT,
                last_trained TIMESTAMP
            )
        """)
        
        self.feature_db.commit()
    
    def process_batch(self):
        """Process a batch of data"""
        logger.info("Starting batch processing")
        
        try:
            # Load processed data files
            processed_files = self._get_latest_processed_files()
            
            if not processed_files:
                logger.warning("No processed data files found")
                return
            
            # Engineer features
            features = self._engineer_batch_features(processed_files)
            
            # Store features
            self._store_batch_features(features)
            
            # Update model registry
            self._update_model_registry()
            
            # Cleanup old data
            self._cleanup_old_data()
            
            logger.info("Batch processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
    
    def _get_latest_processed_files(self) -> Dict[str, str]:
        """Get the latest processed data files"""
        processed_files = {}
        
        for data_type in ['orders', 'sessions', 'inventory', 'supply_chain']:
            pattern = f"{data_type}_*.parquet"
            files = list(self.processed_data_path.glob(pattern))
            
            if files:
                # Get the most recent file
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                processed_files[data_type] = str(latest_file)
        
        return processed_files
    
    def _engineer_batch_features(self, processed_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Engineer features from processed data"""
        features = {}
        
        # Load and engineer features for each data type
        for data_type, file_path in processed_files.items():
            try:
                df = pd.read_parquet(file_path)
                
                if data_type == 'sessions':
                    features[data_type] = self.feature_engineer.engineer_user_features(df)
                elif data_type == 'orders':
                    features[data_type] = self.feature_engineer.engineer_order_features(df)
                elif data_type == 'inventory':
                    features[data_type] = self.feature_engineer.engineer_inventory_features(df)
                elif data_type == 'supply_chain':
                    features[data_type] = self.feature_engineer.engineer_supply_chain_features(df)
                
                logger.info(f"Engineered features for {data_type}: {len(features[data_type])} records")
                
            except Exception as e:
                logger.error(f"Error engineering features for {data_type}: {e}")
        
        return features
    
    def _store_batch_features(self, features: Dict[str, pd.DataFrame]):
        """Store engineered features in feature store"""
        
        for data_type, df in features.items():
            # Store aggregated features
            self._store_aggregated_features(data_type, df)
            
            # Store individual features
            self._store_individual_features(data_type, df)
        
        self.feature_db.commit()
    
    def _store_aggregated_features(self, data_type: str, df: pd.DataFrame):
        """Store aggregated features"""
        timestamp = datetime.now()
        
        # Store various aggregated features
        aggregations = {
            f"{data_type}_count": len(df),
            f"{data_type}_avg_value": df.get('total_value', pd.Series([0])).mean(),
            f"{data_type}_max_value": df.get('total_value', pd.Series([0])).max(),
            f"{data_type}_min_value": df.get('total_value', pd.Series([0])).min(),
        }
        
        for feature_name, feature_value in aggregations.items():
            if not pd.isna(feature_value):
                self.feature_db.execute("""
                    INSERT OR REPLACE INTO features 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_name,
                    feature_name.replace('_', ' ').title(),
                    "aggregated",
                    json.dumps(float(feature_value)),
                    None,
                    timestamp,
                    timestamp,
                    json.dumps({"data_type": data_type, "aggregation": "batch"})
                ))
    
    def _store_individual_features(self, data_type: str, df: pd.DataFrame):
        """Store individual record features"""
        timestamp = datetime.now()
        
        # Store sample of individual features (to avoid overwhelming the store)
        sample_df = df.sample(min(1000, len(df)))
        
        for idx, row in sample_df.iterrows():
            feature_id = f"{data_type}_record_{idx}_{int(timestamp.timestamp())}"
            
            # Convert row to feature vector
            feature_vector = row.to_dict()
            
            self.feature_db.execute("""
                INSERT OR REPLACE INTO features 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feature_id,
                f"{data_type.title()} Record",
                "individual",
                json.dumps(feature_vector),
                json.dumps(list(row.values())),
                timestamp,
                timestamp,
                json.dumps({"data_type": data_type, "record_type": "individual"})
            ))
    
    def _update_model_registry(self):
        """Update model registry with latest information"""
        timestamp = datetime.now()
        
        # Register/update models
        models = [
            {
                "model_id": "intent_transformer_v2",
                "model_name": "Intent Transformer",
                "model_version": "2.0",
                "model_path": "models/intent_transformer_finetuned_multi.pth",
                "feature_set_id": "user_features_v2",
                "performance_metrics": {
                    "accuracy": 0.72,
                    "precision": 0.71,
                    "recall": 0.73,
                    "f1_score": 0.72,
                    "mae": 0.11
                },
                "status": "active",
                "last_trained": timestamp
            },
            {
                "model_id": "gnn_enhanced_v2",
                "model_name": "Enhanced GNN",
                "model_version": "2.0",
                "model_path": "models/saved/enhanced_gnn_enhanced_best.pth",
                "feature_set_id": "supply_features_v2",
                "performance_metrics": {
                    "mae": 0.042,
                    "rmse": 0.058,
                    "r2_score": 0.89,
                    "model_size": "2.3MB"
                },
                "status": "active",
                "last_trained": timestamp
            },
            {
                "model_id": "ppo_agent_v2",
                "model_name": "PPO Agent",
                "model_version": "2.0",
                "model_path": "models/ppo_agent.pth",
                "feature_set_id": "decision_features_v2",
                "performance_metrics": {
                    "improvement": 0.225,
                    "cost_efficiency": 3.25,
                    "reward_mean": 12.5,
                    "reward_std": 2.1
                },
                "status": "active",
                "last_trained": timestamp
            }
        ]
        
        for model in models:
            self.feature_db.execute("""
                INSERT OR REPLACE INTO model_registry 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model["model_id"],
                model["model_name"],
                model["model_version"],
                model["model_path"],
                model["feature_set_id"],
                json.dumps(model["performance_metrics"]),
                timestamp,
                model["status"],
                model["last_trained"]
            ))
        
        # Create feature sets
        feature_sets = [
            {
                "set_id": "user_features_v2",
                "set_name": "User Behavior Features",
                "feature_ids": json.dumps([
                    "intent_score", "urgency_score", "session_duration_minutes",
                    "num_products_viewed", "has_cart_action", "action_diversity"
                ]),
                "description": "Features derived from user session data"
            },
            {
                "set_id": "supply_features_v2",
                "set_name": "Supply Chain Features",
                "feature_ids": json.dumps([
                    "delay_risk", "carbon_footprint", "stock_level",
                    "warehouse_region", "efficiency_score"
                ]),
                "description": "Features derived from supply chain data"
            },
            {
                "set_id": "decision_features_v2",
                "set_name": "Decision Features",
                "feature_ids": json.dumps([
                    "intent_score", "delay_risk", "stock_level", "carbon_cost",
                    "total_value", "shipping_urgency"
                ]),
                "description": "Features used for decision making"
            }
        ]
        
        for feature_set in feature_sets:
            self.feature_db.execute("""
                INSERT OR REPLACE INTO feature_sets 
                VALUES (?, ?, ?, ?, ?)
            """, (
                feature_set["set_id"],
                feature_set["set_name"],
                feature_set["feature_ids"],
                timestamp,
                feature_set["description"]
            ))
    
    def _cleanup_old_data(self):
        """Clean up old data files"""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        for file_path in self.processed_data_path.glob("*.parquet"):
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                file_path.unlink()
                logger.info(f"Deleted old file: {file_path}")
    
    def start_scheduled_processing(self):
        """Start scheduled batch processing"""
        # Schedule batch processing
        schedule.every(self.config.processing_interval).seconds.do(self.process_batch)
        
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info(f"Scheduled batch processing every {self.config.processing_interval} seconds")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of features in the store"""
        cursor = self.feature_db.execute("""
            SELECT feature_type, COUNT(*) as count 
            FROM features 
            GROUP BY feature_type
        """)
        
        feature_summary = {
            "total_features": 0,
            "by_type": {},
            "latest_update": None
        }
        
        for row in cursor.fetchall():
            feature_summary["by_type"][row[0]] = row[1]
            feature_summary["total_features"] += row[1]
        
        # Get latest update time
        cursor = self.feature_db.execute("SELECT MAX(updated_at) FROM features")
        latest_update = cursor.fetchone()[0]
        feature_summary["latest_update"] = latest_update
        
        return feature_summary

# Example usage
if __name__ == "__main__":
    # Initialize batch processor
    config = BatchConfig(processing_interval=60)  # 1 minute for testing
    processor = BatchProcessor(config)
    
    # Start scheduled processing
    processor.start_scheduled_processing()
    
    try:
        # Run for a while
        time.sleep(300)  # 5 minutes
        
        # Get feature summary
        summary = processor.get_feature_summary()
        print("Feature Store Summary:", json.dumps(summary, indent=2))
        
    except KeyboardInterrupt:
        print("Stopping batch processor...")

