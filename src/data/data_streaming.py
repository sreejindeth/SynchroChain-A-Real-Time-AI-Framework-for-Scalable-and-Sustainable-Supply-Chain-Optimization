"""
Real-time Data Streaming Service
Handles continuous data flow from transactional databases to cache
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import sqlite3
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for data streaming"""
    stream_interval: int = 5  # seconds
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: int = 1

class DataStreamer:
    """Handles real-time data streaming from databases"""
    
    def __init__(self, db_path: str, config: StreamConfig = None):
        self.db_path = db_path
        self.config = config or StreamConfig()
        self.running = False
        self.stream_tasks = []
        
    async def start_streaming(self, cache_client):
        """Start all data streams"""
        self.running = True
        
        # Start individual streams
        self.stream_tasks = [
            asyncio.create_task(self._stream_orders(cache_client)),
            asyncio.create_task(self._stream_user_sessions(cache_client)),
            asyncio.create_task(self._stream_inventory(cache_client)),
            asyncio.create_task(self._stream_supply_chain(cache_client))
        ]
        
        logger.info("Data streaming started")
        
    async def stop_streaming(self):
        """Stop all data streams"""
        self.running = False
        
        # Cancel all tasks
        for task in self.stream_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.stream_tasks, return_exceptions=True)
        
        logger.info("Data streaming stopped")
    
    async def _stream_orders(self, cache_client):
        """Stream order data"""
        last_timestamp = None
        
        while self.running:
            try:
                # Get new orders since last timestamp
                orders = self._get_new_orders(last_timestamp)
                
                if orders:
                    # Stream to cache
                    stream_data = {
                        "type": "orders",
                        "data": orders,
                        "timestamp": datetime.now().isoformat(),
                        "count": len(orders)
                    }
                    
                    await cache_client.stream_data("orders", stream_data)
                    
                    # Update last timestamp
                    if orders:
                        last_timestamp = orders[-1].get('order_timestamp')
                    
                    logger.info(f"Streamed {len(orders)} orders")
                
                await asyncio.sleep(self.config.stream_interval)
                
            except Exception as e:
                logger.error(f"Error streaming orders: {e}")
                await asyncio.sleep(self.config.retry_delay)
    
    async def _stream_user_sessions(self, cache_client):
        """Stream user session data"""
        last_timestamp = None
        
        while self.running:
            try:
                # Get new sessions since last timestamp
                sessions = self._get_new_sessions(last_timestamp)
                
                if sessions:
                    # Stream to cache
                    stream_data = {
                        "type": "user_sessions",
                        "data": sessions,
                        "timestamp": datetime.now().isoformat(),
                        "count": len(sessions)
                    }
                    
                    await cache_client.stream_data("sessions", stream_data)
                    
                    # Update last timestamp
                    if sessions:
                        last_timestamp = sessions[-1].get('timestamp')
                    
                    logger.info(f"Streamed {len(sessions)} user sessions")
                
                await asyncio.sleep(self.config.stream_interval)
                
            except Exception as e:
                logger.error(f"Error streaming sessions: {e}")
                await asyncio.sleep(self.config.retry_delay)
    
    async def _stream_inventory(self, cache_client):
        """Stream inventory data"""
        while self.running:
            try:
                # Get current inventory state
                inventory = self._get_current_inventory()
                
                if inventory:
                    # Stream to cache
                    stream_data = {
                        "type": "inventory",
                        "data": inventory,
                        "timestamp": datetime.now().isoformat(),
                        "count": len(inventory)
                    }
                    
                    await cache_client.stream_data("inventory", stream_data)
                    
                    logger.info(f"Streamed {len(inventory)} inventory records")
                
                await asyncio.sleep(self.config.stream_interval * 2)  # Less frequent
                
            except Exception as e:
                logger.error(f"Error streaming inventory: {e}")
                await asyncio.sleep(self.config.retry_delay)
    
    async def _stream_supply_chain(self, cache_client):
        """Stream supply chain data"""
        while self.running:
            try:
                # Get supply chain data
                supply_data = self._get_supply_chain_data()
                
                if supply_data:
                    # Stream to cache
                    stream_data = {
                        "type": "supply_chain",
                        "data": supply_data,
                        "timestamp": datetime.now().isoformat(),
                        "count": len(supply_data)
                    }
                    
                    await cache_client.stream_data("supply_chain", stream_data)
                    
                    logger.info(f"Streamed {len(supply_data)} supply chain records")
                
                await asyncio.sleep(self.config.stream_interval * 3)  # Least frequent
                
            except Exception as e:
                logger.error(f"Error streaming supply chain: {e}")
                await asyncio.sleep(self.config.retry_delay)
    
    def _get_new_orders(self, last_timestamp: Optional[str]) -> List[Dict]:
        """Get new orders from database"""
        conn = sqlite3.connect(f"{self.db_path}/orders.db")
        
        if last_timestamp:
            query = "SELECT * FROM orders WHERE order_timestamp > ? ORDER BY order_timestamp"
            cursor = conn.execute(query, (last_timestamp,))
        else:
            query = "SELECT * FROM orders ORDER BY order_timestamp LIMIT ?"
            cursor = conn.execute(query, (self.config.batch_size,))
        
        columns = [description[0] for description in cursor.description]
        orders = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return orders
    
    def _get_new_sessions(self, last_timestamp: Optional[str]) -> List[Dict]:
        """Get new user sessions from database"""
        conn = sqlite3.connect(f"{self.db_path}/users.db")
        
        if last_timestamp:
            query = "SELECT * FROM user_sessions WHERE timestamp > ? ORDER BY timestamp"
            cursor = conn.execute(query, (last_timestamp,))
        else:
            query = "SELECT * FROM user_sessions ORDER BY timestamp LIMIT ?"
            cursor = conn.execute(query, (self.config.batch_size,))
        
        columns = [description[0] for description in cursor.description]
        sessions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return sessions
    
    def _get_current_inventory(self) -> List[Dict]:
        """Get current inventory state"""
        conn = sqlite3.connect(f"{self.db_path}/inventory.db")
        
        query = "SELECT * FROM inventory WHERE current_stock > 0"
        cursor = conn.execute(query)
        
        columns = [description[0] for description in cursor.description]
        inventory = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return inventory
    
    def _get_supply_chain_data(self) -> List[Dict]:
        """Get supply chain data"""
        conn = sqlite3.connect(f"{self.db_path}/supply_chain.db")
        
        query = "SELECT * FROM shipping_routes"
        cursor = conn.execute(query)
        
        columns = [description[0] for description in cursor.description]
        supply_data = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return supply_data

class RealTimeDataGenerator:
    """Generates real-time data for simulation"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.order_counter = 1000
        self.session_counter = 100
    
    def generate_new_order(self):
        """Generate a new order"""
        order_data = (
            self.order_counter,
            np.random.randint(1, 50),
            f"product_{np.random.randint(1, 20)}",
            np.random.randint(1, 5),
            round(np.random.uniform(10, 500), 2),
            f"warehouse_{np.random.choice(['US_WEST', 'US_EAST', 'EU_CENTRAL', 'ASIA_PACIFIC'])}",
            np.random.choice(['standard', 'expedited', 'priority']),
            datetime.now(),
            'pending'
        )
        
        conn = sqlite3.connect(f"{self.db_path}/orders.db")
        conn.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            order_data
        )
        conn.commit()
        conn.close()
        
        self.order_counter += 1
        return order_data
    
    def generate_new_session(self):
        """Generate a new user session"""
        products = json.dumps([f"product_{j}" for j in np.random.randint(1, 20, 3)])
        actions = json.dumps(np.random.choice(['view', 'cart', 'search'], 5).tolist())
        intent_signals = json.dumps({
            'time_on_page': np.random.uniform(10, 300),
            'scroll_depth': np.random.uniform(0.1, 1.0),
            'click_rate': np.random.uniform(0.01, 0.1)
        })
        
        session_data = (
            f"session_{self.session_counter}",
            np.random.randint(1, 50),
            products,
            actions,
            np.random.randint(60, 1800),
            datetime.now(),
            intent_signals
        )
        
        conn = sqlite3.connect(f"{self.db_path}/users.db")
        conn.execute(
            "INSERT INTO user_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            session_data
        )
        conn.commit()
        conn.close()
        
        self.session_counter += 1
        return session_data
    
    def update_inventory(self):
        """Update inventory levels"""
        conn = sqlite3.connect(f"{self.db_path}/inventory.db")
        
        # Randomly update some inventory levels
        cursor = conn.execute("SELECT product_id, warehouse_id FROM inventory LIMIT 10")
        for row in cursor.fetchall():
            new_stock = np.random.randint(0, 500)
            conn.execute(
                "UPDATE inventory SET current_stock = ?, last_updated = ? WHERE product_id = ? AND warehouse_id = ?",
                (new_stock, datetime.now(), row[0], row[1])
            )
        
        conn.commit()
        conn.close()

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize streaming
        streamer = DataStreamer("data/live_databases")
        
        # Mock cache client
        class MockCacheClient:
            async def stream_data(self, key, data):
                print(f"Streaming {key}: {len(data.get('data', []))} records")
        
        cache_client = MockCacheClient()
        
        # Start streaming
        await streamer.start_streaming(cache_client)
        
        # Let it run for a bit
        await asyncio.sleep(30)
        
        # Stop streaming
        await streamer.stop_streaming()
    
    asyncio.run(main())

