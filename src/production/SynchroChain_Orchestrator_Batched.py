"""
SynchroChain Model Orchestrator - Batched & Async Production Version
Implements micro-batching and async concurrency for high throughput
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable, Awaitable
from datetime import datetime
import asyncio
from dataclasses import dataclass
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SynchroChainOrchestrator')

@dataclass
class SupplyChainRequest:
    """Data structure for supply chain requests."""
    session_id: str
    user_session: List[str]
    order_context: Dict[str, Any]
    timestamp: datetime
    priority: str = "normal"

@dataclass
class SupplyChainResponse:
    """Data structure for supply chain responses."""
    session_id: str
    predictions: Dict[str, float]
    decisions: Dict[str, Any]
    confidence: Dict[str, str]
    processing_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class MicroBatcher:
    """Micro-batching utility for efficient model inference."""
    
    def __init__(self, max_batch_size: int, max_delay_ms: int,
                 process_batch_fn: Callable[[list], Awaitable[list]]):
        self.max_batch_size = max_batch_size
        self.max_delay = max_delay_ms / 1000.0
        self.process_batch_fn = process_batch_fn
        self._cv = asyncio.Condition()
        self._queue = []
        self._runner_task = None
        
    def start(self):
        """Start the batch processor."""
        if self._runner_task is None:
            try:
                self._runner_task = asyncio.create_task(self._runner())
            except RuntimeError:
                # No event loop yet, will be started on first submit
                pass
    
    async def submit(self, item):
        """Submit an item for batched processing."""
        # Ensure runner is started
        if self._runner_task is None:
            self._runner_task = asyncio.create_task(self._runner())
        
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        async with self._cv:
            self._queue.append((item, fut))
            if len(self._queue) >= self.max_batch_size:
                self._cv.notify_all()
        return await fut
    
    async def _runner(self):
        """Background batch processor."""
        while True:
            async with self._cv:
                if not self._queue:
                    await self._cv.wait()
                await asyncio.sleep(self.max_delay)
                batch = self._queue[:self.max_batch_size]
                del self._queue[:len(batch)]
            
            if not batch:
                continue
            
            items = [x for (x, _) in batch]
            try:
                results = await self.process_batch_fn(items)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                results = [e] * len(items)
            
            for (_, fut), res in zip(batch, results):
                if isinstance(res, Exception):
                    fut.set_exception(res)
                else:
                    fut.set_result(res)

class ModelOrchestrator:
    """
    Central Model Orchestrator for SynchroChain Supply Chain AI System
    With micro-batching and async concurrency for high throughput
    """
    
    def __init__(self):
        """Initialize the orchestrator with all trained models."""
        logger.info("🚀 Initializing SynchroChain Model Orchestrator (Batched)...")
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        # Load all models
        self.load_models()
        
        # Initialize micro-batchers
        self.intent_batcher = MicroBatcher(
            max_batch_size=32, 
            max_delay_ms=10, 
            process_batch_fn=self._process_intent_batch
        )
        self.gnn_batcher = MicroBatcher(
            max_batch_size=32, 
            max_delay_ms=10, 
            process_batch_fn=self._process_gnn_batch
        )
        
        logger.info("✅ Model Orchestrator initialized successfully!")
    
    def initialize_batchers(self):
        """Initialize batch processors (called before first use)."""
        self.intent_batcher.start()
        self.gnn_batcher.start()
    
    def load_models(self):
        """Load all trained models and components."""
        logger.info("🔄 Loading AI models and components...")
        
        try:
            # Try to import from production system
            from SynchroChain_Production_System import (
                IntentTransformer, DelayRiskGNN, InventoryRuleEngine, 
                CarbonRuleEngine, PPOAgent
            )
            
            self.intent_transformer = IntentTransformer()
            self.delay_risk_gnn = DelayRiskGNN()
            self.inventory_estimator = InventoryRuleEngine()
            self.carbon_estimator = CarbonRuleEngine()
            self.ppo_agent = PPOAgent()
            logger.info("   ✅ All models loaded from production system")
            
        except Exception as e:
            logger.warning(f"   Production models not available: {e}")
            logger.info("   Creating optimized mock models for benchmarking...")
            self.create_mock_models()
    
    def create_mock_models(self):
        """Create optimized mock models for benchmarking."""
        
        class MockIntentTransformer:
            def predict(self, user_session):
                intent_score = min(1.0, len([a for a in user_session if a == 'add_to_cart']) * 0.3)
                urgency_level = min(1.0, len(user_session) * 0.1)
                return intent_score, urgency_level
            
            def predict_batch(self, user_sessions):
                """Batch prediction for efficiency."""
                return [self.predict(s) for s in user_sessions]
        
        class MockDelayRiskGNN:
            def predict(self, order_context):
                base_risk = 0.1
                if order_context.get('shipping_mode') == 'Same Day':
                    base_risk += 0.3
                if order_context.get('is_international', False):
                    base_risk += 0.2
                return min(1.0, base_risk)
            
            def predict_batch(self, order_contexts):
                """Batch prediction for efficiency."""
                return [self.predict(ctx) for ctx in order_contexts]
        
        class MockInventoryEstimator:
            def estimate(self, order_context):
                category = order_context.get('category_name', 'Unknown')
                if category in ['Electronics', 'Technology']:
                    return 0.3
                elif category in ['Fashion', 'Clothing']:
                    return 0.7
                else:
                    return 0.5
        
        class MockCarbonEstimator:
            def estimate(self, order_context):
                shipping_mode = order_context.get('shipping_mode', 'Standard Class')
                if shipping_mode == 'Same Day':
                    return 0.8
                elif shipping_mode == 'First Class':
                    return 0.6
                else:
                    return 0.4
        
        class MockPPOAgent:
            def decide(self, state):
                intent_score = state.get('intent_score', 0)
                urgency = state.get('urgency_level', 0)
                delay_risk = state.get('delay_risk', 0)
                
                action = {
                    'pre_allocate': intent_score > 0.7 and urgency > 0.6,
                    'restock': state.get('inventory_level', 0.5) < 0.3,
                    'shipping_mode': 'Same Day' if urgency > 0.8 and delay_risk < 0.3 else 'Standard Class',
                    'priority': 'High' if intent_score > 0.8 else 'Normal'
                }
                return action
        
        self.intent_transformer = MockIntentTransformer()
        self.delay_risk_gnn = MockDelayRiskGNN()
        self.inventory_estimator = MockInventoryEstimator()
        self.carbon_estimator = MockCarbonEstimator()
        self.ppo_agent = MockPPOAgent()
        
        logger.info("   ✅ Optimized mock models created")
    
    async def _process_intent_batch(self, sessions: list) -> list:
        """Process a batch of intent predictions."""
        if hasattr(self.intent_transformer, "predict_batch"):
            preds = self.intent_transformer.predict_batch(sessions)
            return [(float(i), float(u)) for (i, u) in preds]
        else:
            # Fallback to individual predictions
            return [(float(i), float(u)) for s in sessions 
                    for i, u in [self.intent_transformer.predict(s)]]
    
    async def _process_gnn_batch(self, order_contexts: list) -> list:
        """Process a batch of GNN predictions."""
        if hasattr(self.delay_risk_gnn, "predict_batch"):
            preds = self.delay_risk_gnn.predict_batch(order_contexts)
            return [float(p) for p in preds]
        else:
            # Fallback to individual predictions
            return [float(self.delay_risk_gnn.predict(ctx)) for ctx in order_contexts]
    
    async def process_supply_chain_request(self, request: SupplyChainRequest) -> SupplyChainResponse:
        """Process a complete supply chain request through the AI pipeline."""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            # Step 1: Intent Prediction (batched)
            intent_score, urgency_level = await self.predict_intent(request.user_session)
            
            # Step 2: Supply Chain Risk Assessment (batched)
            delay_risk = await self.predict_delay_risk(request.order_context)
            
            # Step 3: Rule-Based Estimates (fast, no batching needed)
            inventory_level = await self.estimate_inventory_level(request.order_context)
            carbon_cost = await self.estimate_carbon_cost(request.order_context)
            
            # Step 4: State Aggregation
            state = self.aggregate_state(
                intent_score, urgency_level, delay_risk, 
                inventory_level, carbon_cost, request.order_context
            )
            
            # Step 5: RL Decision Making
            decisions = await self.make_optimal_decisions(state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=True)
            
            # Create response
            response = SupplyChainResponse(
                session_id=request.session_id,
                predictions={
                    'intent_score': intent_score,
                    'urgency_level': urgency_level,
                    'delay_risk': delay_risk,
                    'inventory_level': inventory_level,
                    'carbon_cost': carbon_cost
                },
                decisions=decisions,
                confidence=self.calculate_confidence(intent_score, delay_risk),
                processing_time=processing_time,
                timestamp=datetime.now(),
                success=True
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=False)
            
            return SupplyChainResponse(
                session_id=request.session_id,
                predictions={},
                decisions={},
                confidence={},
                processing_time=processing_time,
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    async def predict_intent(self, user_session: List[str]) -> Tuple[float, float]:
        """Predict user intent and urgency from session data (batched)."""
        try:
            intent_score, urgency_level = await self.intent_batcher.submit(user_session)
            return float(intent_score), float(urgency_level)
        except Exception as e:
            logger.error(f"Intent prediction failed: {e}")
            return 0.0, 0.0
    
    async def predict_delay_risk(self, order_context: Dict[str, Any]) -> float:
        """Predict delay risk using the trained GNN (batched)."""
        try:
            delay_risk = await self.gnn_batcher.submit(order_context)
            return float(delay_risk)
        except Exception as e:
            logger.error(f"Delay risk prediction failed: {e}")
            return 0.5
    
    async def estimate_inventory_level(self, order_context: Dict[str, Any]) -> float:
        """Estimate inventory level using rule-based system."""
        try:
            inventory_level = self.inventory_estimator.estimate(order_context)
            return float(inventory_level)
        except Exception as e:
            logger.error(f"Inventory estimation failed: {e}")
            return 0.5
    
    async def estimate_carbon_cost(self, order_context: Dict[str, Any]) -> float:
        """Estimate carbon cost using rule-based system."""
        try:
            carbon_cost = self.carbon_estimator.estimate(order_context)
            return float(carbon_cost)
        except Exception as e:
            logger.error(f"Carbon cost estimation failed: {e}")
            return 0.5
    
    def aggregate_state(self, intent_score: float, urgency_level: float, 
                       delay_risk: float, inventory_level: float, 
                       carbon_cost: float, order_context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate all predictions into a complete state vector."""
        return {
            'intent_score': intent_score,
            'urgency_level': urgency_level,
            'delay_risk': delay_risk,
            'inventory_level': inventory_level,
            'carbon_cost': carbon_cost,
            'order_value': order_context.get('order_value', 0),
            'product_category': order_context.get('category_name', 'Unknown'),
            'shipping_mode': order_context.get('shipping_mode', 'Standard Class'),
            'is_international': order_context.get('is_international', False)
        }
    
    async def make_optimal_decisions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make optimal decisions using the PPO agent."""
        try:
            decisions = self.ppo_agent.decide(state)
            decisions = self.validate_decisions(decisions, state)
            return decisions
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return self.get_fallback_decisions(state)
    
    def validate_decisions(self, decisions: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust decisions based on business rules."""
        if state['intent_score'] < 0.3 and decisions['shipping_mode'] == 'Same Day':
            decisions['shipping_mode'] = 'Standard Class'
        
        if state['delay_risk'] > 0.8 and decisions['priority'] == 'Normal':
            decisions['priority'] = 'High'
        
        return decisions
    
    def get_fallback_decisions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback decisions when PPO agent fails."""
        return {
            'pre_allocate': state['intent_score'] > 0.7,
            'restock': state['inventory_level'] < 0.3,
            'shipping_mode': 'Standard Class',
            'priority': 'Normal'
        }
    
    def calculate_confidence(self, intent_score: float, delay_risk: float) -> Dict[str, str]:
        """Calculate confidence levels for predictions."""
        return {
            'intent': 'high' if intent_score > 0.7 else 'medium' if intent_score > 0.4 else 'low',
            'delay_risk': 'high' if delay_risk > 0.7 else 'medium' if delay_risk > 0.4 else 'low',
            'overall': 'high' if intent_score > 0.6 and delay_risk < 0.5 else 'medium'
        }
    
    def update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics."""
        self.metrics['total_processing_time'] += processing_time
        self.metrics['average_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['total_requests']
        )
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        success_rate = (
            self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1) * 100
        )
        
        return {
            'total_requests': self.metrics['total_requests'],
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': f"{success_rate:.2f}%",
            'average_processing_time': f"{self.metrics['average_processing_time']:.3f}s",
            'throughput_per_second': f"{1/self.metrics['average_processing_time']:.1f}" if self.metrics['average_processing_time'] > 0 else "0"
        }

