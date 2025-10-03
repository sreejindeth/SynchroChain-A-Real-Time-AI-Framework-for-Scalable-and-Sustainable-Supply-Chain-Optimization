# SynchroChain Data Layer Architecture

## 🏗️ **Complete Data Layer Implementation**

This document describes the complete data layer architecture that implements the full pipeline:

**Live Transactional Databases → Real-time Cache → Processed Logs → Feature Store/Model Registry**

---

## 📊 **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Live DBs      │───▶│  Real-time      │───▶│   Data          │───▶│   Feature       │
│                 │    │  Cache          │    │   Processing    │    │   Store         │
│ • Orders        │    │                 │    │                 │    │                 │
│ • Users         │    │ • Redis         │    │ • Feature       │    │ • Features      │
│ • Inventory     │    │ • Memory        │    │   Engineering  │    │ • Model         │
│ • Supply Chain  │    │ • Streaming     │    │ • Aggregation  │    │   Registry      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data          │    │   Real-time     │    │   Batch         │    │   Model         │
│   Streaming     │    │   Updates       │    │   Processing    │    │   Orchestrator  │
│                 │    │                 │    │                 │    │                 │
│ • Continuous    │    │ • 5s intervals  │    │ • 5min cycles   │    │ • AI Models     │
│ • Async         │    │ • Cache TTL     │    │ • Feature      │    │ • Predictions   │
│ • Error Handling│    │ • Fallback      │    │   Updates      │    │ • Integration    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🗄️ **1. Live Transactional Databases**

### **SQLite Databases**
- **Orders Database**: Order transactions, shipping methods, timestamps
- **Users Database**: User sessions, clickstream data, intent signals
- **Inventory Database**: Stock levels, warehouse data, carbon footprint
- **Supply Chain Database**: Shipping routes, delay risks, carbon costs

### **Key Features**
- Real-time transactional data
- Sample data generation for simulation
- Continuous data updates
- ACID compliance

### **Schema Examples**
```sql
-- Orders Table
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    product_id TEXT,
    quantity INTEGER,
    price DECIMAL(10,2),
    warehouse_id TEXT,
    shipping_method TEXT,
    order_timestamp TIMESTAMP,
    status TEXT
);

-- User Sessions Table
CREATE TABLE user_sessions (
    session_id TEXT PRIMARY KEY,
    user_id INTEGER,
    products_viewed TEXT,  -- JSON
    actions_performed TEXT, -- JSON
    session_duration INTEGER,
    timestamp TIMESTAMP,
    intent_signals TEXT     -- JSON
);
```

---

## ⚡ **2. Real-time Cache**

### **Redis Integration**
- **Primary**: Redis for high-performance caching
- **Fallback**: In-memory cache if Redis unavailable
- **TTL Management**: Configurable cache expiration
- **Streaming Support**: Real-time data flow

### **Cache Structure**
```
stream:{data_type}:{timestamp}  # Streaming data (5min TTL)
latest:{data_type}              # Latest data (1hr TTL)
features:{feature_id}           # Computed features (1hr TTL)
```

### **Key Features**
- Automatic fallback to memory cache
- Configurable TTL per data type
- Real-time streaming updates
- High-performance data access

---

## 🔄 **3. Data Processing Pipeline**

### **Feature Engineering**
- **User Features**: Intent scores, urgency levels, behavior patterns
- **Order Features**: Value calculations, temporal patterns, shipping preferences
- **Inventory Features**: Stock ratios, carbon intensity, warehouse regions
- **Supply Chain Features**: Risk categories, efficiency scores, carbon costs

### **Processing Components**
- **DataProcessor**: Main processing engine
- **FeatureEngineer**: ML feature creation
- **BatchProcessor**: Scheduled processing
- **DataStreamer**: Real-time streaming

### **Example Features**
```python
# User Intent Features
intent_score = (time_on_page / 300) * scroll_depth * 10
urgency_score = click_rate * 20
action_diversity = len(set(actions_performed))

# Inventory Features
stock_level = current_stock - reserved_stock
carbon_intensity = carbon_footprint / current_stock
is_low_stock = stock_level < 50
```

---

## 🏪 **4. Feature Store**

### **SQLite-based Storage**
- **Features Table**: Individual feature storage
- **Feature Sets**: Grouped features for models
- **Model Registry**: Model metadata and versions
- **Performance Tracking**: Model metrics and history

### **Feature Store Schema**
```sql
-- Features Table
CREATE TABLE features (
    feature_id TEXT PRIMARY KEY,
    feature_name TEXT,
    feature_type TEXT,
    feature_value TEXT,      -- JSON
    feature_vector TEXT,     -- JSON array
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    metadata TEXT            -- JSON
);

-- Model Registry
CREATE TABLE model_registry (
    model_id TEXT PRIMARY KEY,
    model_name TEXT,
    model_version TEXT,
    model_path TEXT,
    feature_set_id TEXT,
    performance_metrics TEXT, -- JSON
    created_at TIMESTAMP,
    status TEXT,
    last_trained TIMESTAMP
);
```

### **Key Features**
- Versioned feature storage
- Model registry with metadata
- Performance tracking
- Feature lineage and provenance

---

## 🤖 **5. AI Model Integration**

### **Model Orchestrator Integration**
- **Intent Transformer**: Uses user session features
- **GNN Module**: Uses supply chain graph features
- **PPO Agent**: Uses decision features
- **Orchestrator**: Coordinates all models

### **Feature Sets**
- **User Features**: Intent scores, behavior patterns
- **Supply Features**: Risk assessments, carbon costs
- **Decision Features**: Combined features for RL agent

---

## 📈 **6. Batch Processing System**

### **Scheduled Processing**
- **Interval**: Every 5 minutes (configurable)
- **Batch Size**: 10,000 records (configurable)
- **Retention**: 30 days (configurable)
- **Error Handling**: Retry logic and logging

### **Processing Pipeline**
1. **Extract**: Get latest data from databases
2. **Transform**: Engineer features and aggregations
3. **Load**: Store in feature store
4. **Update**: Refresh model registry
5. **Cleanup**: Remove old data files

---

## 🚀 **7. Real-time Data Streaming**

### **Async Streaming**
- **Orders Stream**: New order events
- **Sessions Stream**: User behavior updates
- **Inventory Stream**: Stock level changes
- **Supply Chain Stream**: Route and risk updates

### **Streaming Features**
- Continuous data flow
- Error handling and retries
- Configurable intervals
- Real-time cache updates

---

## 🔧 **8. Configuration & Setup**

### **DataConfig**
```python
@dataclass
class DataConfig:
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    db_path: str = "data/live_databases"
    cache_ttl: int = 3600  # 1 hour
    batch_size: int = 1000
    processing_interval: int = 60  # 1 minute
```

### **BatchConfig**
```python
@dataclass
class BatchConfig:
    processing_interval: int = 3600  # 1 hour
    batch_size: int = 10000
    feature_store_path: str = "data/feature_store"
    processed_data_path: str = "data/processed"
    retention_days: int = 30
```

---

## 🎯 **9. Usage Examples**

### **Starting Data Layer**
```python
from src.data.data_layer import DataLayerOrchestrator, DataConfig

# Initialize
config = DataConfig()
orchestrator = DataLayerOrchestrator(config)

# Start pipeline
orchestrator.start_data_pipeline()

# Get latest features
features = orchestrator.get_latest_features()
```

### **Batch Processing**
```python
from src.data.batch_processor import BatchProcessor, BatchConfig

# Initialize
config = BatchConfig()
processor = BatchProcessor(config)

# Start scheduled processing
processor.start_scheduled_processing()

# Get feature summary
summary = processor.get_feature_summary()
```

### **Data Streaming**
```python
from src.data.data_streaming import DataStreamer

# Initialize
streamer = DataStreamer("data/live_databases")

# Start streaming
await streamer.start_streaming(cache_client)
```

---

## 📊 **10. Monitoring & Metrics**

### **Data Layer Status**
- Database connectivity
- Cache hit rates
- Processing latency
- Feature store updates
- Model registry status

### **Performance Metrics**
- Throughput (records/second)
- Latency (processing time)
- Error rates
- Resource utilization
- Data freshness

---

## 🔒 **11. Security & Reliability**

### **Data Security**
- SQLite file permissions
- Redis authentication (if configured)
- Feature store access control
- Model registry versioning

### **Reliability Features**
- Automatic fallback mechanisms
- Error handling and retries
- Data validation
- Backup and recovery
- Monitoring and alerting

---

## 🚀 **12. Deployment**

### **Launch Scripts**
- `scripts/launch_data_layer.py`: Complete data layer
- `frontend/enhanced_api.py`: API with data integration
- `launch_enhanced_frontend.bat`: Windows launcher

### **Dependencies**
```
redis>=4.0.0
schedule>=1.2.0
pyarrow>=10.0.0
pandas>=1.3.0
numpy>=1.21.0
```

### **Directory Structure**
```
data/
├── live_databases/          # SQLite databases
│   ├── orders.db
│   ├── users.db
│   ├── inventory.db
│   └── supply_chain.db
├── processed/               # Processed data files
│   ├── orders_*.parquet
│   ├── sessions_*.parquet
│   └── inventory_*.parquet
└── feature_store/          # Feature store
    ├── feature_store.db
    └── model_registry.db
```

---

## 🎉 **Summary**

The SynchroChain data layer provides:

✅ **Complete Data Pipeline**: Live DBs → Cache → Processing → Feature Store  
✅ **Real-time Streaming**: Continuous data flow with 5-second updates  
✅ **Feature Engineering**: ML-ready features for all AI models  
✅ **Model Registry**: Versioned model storage and metadata  
✅ **Batch Processing**: Scheduled feature updates every 5 minutes  
✅ **High Availability**: Redis + memory cache fallback  
✅ **Production Ready**: Error handling, monitoring, and scalability  

This implementation fulfills your requirement for a complete data layer that streams raw data into real-time cache, processes it into logs, stores it in processed data storage, and batch-loads it into the feature store/model registry.

