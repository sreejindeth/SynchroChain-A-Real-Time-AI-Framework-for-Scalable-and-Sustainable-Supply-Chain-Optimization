# src/system/orchestrator.py
"""
Model Orchestrator for SynchroChain Supply Chain AI System
Coordinates Intent Transformer → GNN → PPO Agent pipeline
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import yaml
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import models
from src.models.intent_transformer.intent_transformer_lightweight import LightweightIntentTransformer
from src.models.gnn.enhanced_supply_gnn import EnhancedSupplyGNN
from src.models.agents.ppo_trainer import PPOPolicy
from src.utils.logging_config import setup_logging

class ModelOrchestrator:
    """
    Orchestrates the three-layer AI pipeline:
    1. Intent Transformer: User behavior → Intent predictions
    2. GNN: Supply chain graph → Risk/inventory predictions  
    3. PPO Agent: Combined inputs → Fulfillment decisions
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the orchestrator with all trained models."""
        self.project_root = project_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info("Initializing SynchroChain Model Orchestrator...")
        
        # Model paths
        self.intent_model_path = os.path.join(project_root, 'models', 'intent_transformer_finetuned_multi.pth')
        self.gnn_model_path = os.path.join(project_root, 'models', 'saved', 'enhanced_gnn_enhanced_best.pth')
        self.ppo_model_path = os.path.join(project_root, 'models', 'ppo_agent.pth')
        
        # Load models
        self.intent_model = self._load_intent_model()
        self.gnn_model = self._load_gnn_model()
        self.ppo_agent = self._load_ppo_agent()
        
        # Load preprocessing components
        self.intent_encoders = self._load_intent_encoders()
        self.gnn_preprocessors = self._load_gnn_preprocessors()
        
        self.logger.info("Model Orchestrator initialized successfully!")
        
    def _load_intent_model(self) -> LightweightIntentTransformer:
        """Load the trained Intent Transformer model."""
        try:
            config = self.config['models']['intent_transformer']
            model = LightweightIntentTransformer(
                vocab_size=5000,  # From preprocessing
                max_seq_length=config['max_seq_length'],
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
            
            if os.path.exists(self.intent_model_path):
                checkpoint = torch.load(self.intent_model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Intent Transformer loaded from {self.intent_model_path}")
            else:
                self.logger.warning(f"Intent model not found at {self.intent_model_path}")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load Intent Transformer: {e}")
            return None
    
    def _load_gnn_model(self) -> EnhancedSupplyGNN:
        """Load the trained GNN model."""
        try:
            config = self.config['models']['gnn']
            model = EnhancedSupplyGNN(
                input_dim=7,  # From preprocessing
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                dropout=config['dropout']
            )
            
            if os.path.exists(self.gnn_model_path):
                checkpoint = torch.load(self.gnn_model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
                self.logger.info(f"GNN model loaded from {self.gnn_model_path}")
            else:
                self.logger.warning(f"GNN model not found at {self.gnn_model_path}")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load GNN model: {e}")
            return None
    
    def _load_ppo_agent(self) -> PPOPolicy:
        """Load the trained PPO agent."""
        try:
            # PPO agent expects 4D state space from original training
            model = PPOPolicy(state_dim=4, action_dim=3, hidden_dim=64)
            
            if os.path.exists(self.ppo_model_path):
                model.load_state_dict(torch.load(self.ppo_model_path, map_location=self.device))
                self.logger.info(f"PPO Agent loaded from {self.ppo_model_path}")
            else:
                self.logger.warning(f"PPO model not found at {self.ppo_model_path}")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load PPO Agent: {e}")
            return None
    
    def _load_intent_encoders(self) -> Dict[str, Any]:
        """Load Intent Transformer preprocessing components."""
        try:
            encoders_path = os.path.join(project_root, 'models', 'intent_encoders.pkl')
            if os.path.exists(encoders_path):
                with open(encoders_path, 'rb') as f:
                    encoders = pickle.load(f)
                self.logger.info("Intent encoders loaded successfully")
                return encoders
            else:
                self.logger.warning(f"Intent encoders not found at {encoders_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load intent encoders: {e}")
            return {}
    
    def _load_gnn_preprocessors(self) -> Dict[str, Any]:
        """Load GNN preprocessing components."""
        try:
            preprocessors = {}
            
            # Load scaler
            scaler_path = os.path.join(project_root, 'models', 'saved', 'gnn_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    preprocessors['scaler'] = pickle.load(f)
            
            # Load label encoder
            le_path = os.path.join(project_root, 'models', 'saved', 'gnn_le_node_type.pkl')
            if os.path.exists(le_path):
                with open(le_path, 'rb') as f:
                    preprocessors['label_encoder'] = pickle.load(f)
            
            # Load node mapping
            node_map_path = os.path.join(project_root, 'models', 'saved', 'gnn_node_id_to_idx.pkl')
            if os.path.exists(node_map_path):
                with open(node_map_path, 'rb') as f:
                    preprocessors['node_mapping'] = pickle.load(f)
            
            self.logger.info("GNN preprocessors loaded successfully")
            return preprocessors
            
        except Exception as e:
            self.logger.error(f"Failed to load GNN preprocessors: {e}")
            return {}
    
    def predict_intent(self, user_session: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Predict user intent from browsing session.
        
        Args:
            user_session: List of user actions [{'product_name': str, 'action': str, 'timestamp': str}]
            
        Returns:
            Dict with intent predictions: {'intent_low': float, 'intent_medium': float, 'intent_high': float, 'urgency': float}
        """
        if not self.intent_model or not self.intent_encoders:
            self.logger.error("Intent model or encoders not available")
            return {'intent_low': 0.33, 'intent_medium': 0.33, 'intent_high': 0.34, 'urgency': 0.5}
        
        try:
            # Preprocess session data
            processed_session = self._preprocess_user_session(user_session)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.intent_model(processed_session)
                
                # Extract predictions
                intent_logits = predictions['intent_logits'].squeeze()
                urgency = predictions['urgency'].squeeze().item()
                
                # Convert to probabilities
                intent_probs = torch.softmax(intent_logits, dim=0)
                
                return {
                    'intent_low': intent_probs[0].item(),
                    'intent_medium': intent_probs[1].item(), 
                    'intent_high': intent_probs[2].item(),
                    'urgency': urgency,
                    'navigation_depth': predictions.get('navigation_depth', torch.tensor(0.5)).item()
                }
                
        except Exception as e:
            self.logger.error(f"Intent prediction failed: {e}")
            return {'intent_low': 0.33, 'intent_medium': 0.33, 'intent_high': 0.34, 'urgency': 0.5}
    
    def predict_supply_chain_risk(self, supply_chain_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict supply chain risks using GNN.
        
        Args:
            supply_chain_context: Dict with supply chain information
            
        Returns:
            Dict with risk predictions: {'delay_risk': float, 'inventory_risk': float, 'carbon_cost': float}
        """
        if not self.gnn_model or not self.gnn_preprocessors:
            self.logger.error("GNN model or preprocessors not available")
            return {'delay_risk': 0.5, 'inventory_risk': 0.5, 'carbon_cost': 0.5}
        
        try:
            # Preprocess supply chain data
            processed_data = self._preprocess_supply_chain_data(supply_chain_context)
            
            # Get GNN predictions
            with torch.no_grad():
                predictions = self.gnn_model(processed_data)
                
                return {
                    'delay_risk': predictions[1].item() if len(predictions) > 1 else 0.5,
                    'inventory_risk': predictions[0].item() if len(predictions) > 0 else 0.5,
                    'carbon_cost': predictions[2].item() if len(predictions) > 2 else 0.5
                }
                
        except Exception as e:
            self.logger.error(f"GNN prediction failed: {e}")
            return {'delay_risk': 0.5, 'inventory_risk': 0.5, 'carbon_cost': 0.5}
    
    def make_fulfillment_decision(self, intent_predictions: Dict[str, float], 
                                supply_predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Make fulfillment decision using PPO agent.
        
        Args:
            intent_predictions: Output from predict_intent()
            supply_predictions: Output from predict_supply_chain_risk()
            
        Returns:
            Dict with decision: {'action': int, 'action_name': str, 'confidence': float, 'reasoning': str}
        """
        if not self.ppo_agent:
            self.logger.error("PPO agent not available")
            return self._fallback_decision(intent_predictions, supply_predictions)
        
        try:
            # Combine predictions into PPO state
            state = self._create_ppo_state(intent_predictions, supply_predictions)
            
            # Get PPO decision
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                logits, value = self.ppo_agent(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][action].item()
            
            # Map action to decision
            action_names = ['Standard Shipping', 'Expedited Shipping', 'Priority Shipping']
            reasoning = self._generate_reasoning(action, intent_predictions, supply_predictions, confidence)
            
            return {
                'action': action,
                'action_name': action_names[action],
                'confidence': confidence,
                'reasoning': reasoning,
                'value_estimate': value.item(),
                'state_used': state
            }
            
        except Exception as e:
            self.logger.error(f"PPO decision failed: {e}")
            return self._fallback_decision(intent_predictions, supply_predictions)
    
    def orchestrate_full_pipeline(self, user_session: List[Dict[str, Any]], 
                                 supply_chain_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete AI pipeline: Intent → GNN → PPO → Decision.
        
        Args:
            user_session: User browsing behavior
            supply_chain_context: Supply chain state information
            
        Returns:
            Complete pipeline results with all predictions and final decision
        """
        start_time = datetime.now()
        
        self.logger.info("Starting SynchroChain AI Pipeline...")
        
        # Step 1: Intent Prediction
        self.logger.info("Step 1: Analyzing user intent...")
        intent_predictions = self.predict_intent(user_session)
        
        # Step 2: Supply Chain Risk Assessment
        self.logger.info("Step 2: Assessing supply chain risks...")
        supply_predictions = self.predict_supply_chain_risk(supply_chain_context)
        
        # Step 3: Fulfillment Decision
        self.logger.info("Step 3: Making fulfillment decision...")
        decision = self.make_fulfillment_decision(intent_predictions, supply_predictions)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Compile results
        results = {
            'timestamp': end_time.isoformat(),
            'processing_time_seconds': processing_time,
            'intent_predictions': intent_predictions,
            'supply_chain_predictions': supply_predictions,
            'final_decision': decision,
            'pipeline_summary': {
                'user_intent_level': self._get_intent_level(intent_predictions),
                'supply_risk_level': self._get_risk_level(supply_predictions),
                'recommended_action': decision['action_name'],
                'confidence_score': decision['confidence']
            }
        }
        
        self.logger.info(f"Pipeline completed in {processing_time:.3f}s")
        self.logger.info(f"Decision: {decision['action_name']} (confidence: {decision['confidence']:.3f})")
        
        return results
    
    def _preprocess_user_session(self, user_session: List[Dict[str, Any]]) -> torch.Tensor:
        """Preprocess user session for Intent Transformer."""
        # Simplified preprocessing - in production, use the actual encoders
        max_seq_length = self.config['models']['intent_transformer']['max_seq_length']
        
        # Create dummy sequence (replace with actual preprocessing)
        sequence = torch.randint(0, 1000, (1, max_seq_length)).to(self.device)
        return sequence
    
    def _preprocess_supply_chain_data(self, context: Dict[str, Any]) -> Any:
        """Preprocess supply chain data for GNN."""
        # Simplified preprocessing - in production, create actual graph data
        from torch_geometric.data import Data
        
        # Create dummy graph data
        x = torch.randn(10, 7).to(self.device)  # 10 nodes, 7 features
        edge_index = torch.randint(0, 10, (2, 20)).to(self.device)  # 20 edges
        
        return Data(x=x, edge_index=edge_index)
    
    def _create_ppo_state(self, intent_preds: Dict[str, float], supply_preds: Dict[str, float]) -> List[float]:
        """Create state vector for PPO agent."""
        # Combine predictions into 4D state vector as expected by trained PPO
        intent_signal = intent_preds['intent_high']  # Use high intent as primary signal
        delay_risk = supply_preds['delay_risk']
        inventory_level = 1.0 - supply_preds['inventory_risk']  # Invert risk to get level
        cost_factor = supply_preds['carbon_cost']
        
        return [intent_signal, delay_risk, inventory_level, cost_factor]
    
    def _fallback_decision(self, intent_preds: Dict[str, float], supply_preds: Dict[str, float]) -> Dict[str, Any]:
        """Fallback rule-based decision when PPO is unavailable."""
        # Simple rule-based logic
        if intent_preds['intent_high'] > 0.6 or supply_preds['delay_risk'] > 0.7:
            action = 2  # Priority
        elif intent_preds['intent_medium'] > 0.5 or supply_preds['delay_risk'] > 0.4:
            action = 1  # Expedited
        else:
            action = 0  # Standard
        
        action_names = ['Standard Shipping', 'Expedited Shipping', 'Priority Shipping']
        
        return {
            'action': action,
            'action_name': action_names[action],
            'confidence': 0.8,
            'reasoning': 'Rule-based fallback decision',
            'value_estimate': 0.0,
            'state_used': [0, 0, 0, 0]
        }
    
    def _generate_reasoning(self, action: int, intent_preds: Dict[str, float], 
                          supply_preds: Dict[str, float], confidence: float) -> str:
        """Generate human-readable reasoning for the decision."""
        intent_level = self._get_intent_level(intent_preds)
        risk_level = self._get_risk_level(supply_preds)
        
        action_names = ['standard', 'expedited', 'priority']
        
        reasoning = f"Selected {action_names[action]} shipping based on {intent_level} user intent "
        reasoning += f"and {risk_level} supply chain risk (confidence: {confidence:.1%})"
        
        return reasoning
    
    def _get_intent_level(self, intent_preds: Dict[str, float]) -> str:
        """Get human-readable intent level."""
        if intent_preds['intent_high'] > 0.5:
            return "high"
        elif intent_preds['intent_medium'] > 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_risk_level(self, supply_preds: Dict[str, float]) -> str:
        """Get human-readable risk level."""
        avg_risk = (supply_preds['delay_risk'] + supply_preds['inventory_risk']) / 2
        if avg_risk > 0.6:
            return "high"
        elif avg_risk > 0.3:
            return "medium"
        else:
            return "low"

def main():
    """Demo the orchestrator."""
    print("SynchroChain Model Orchestrator Demo")
    print("="*50)
    
    # Initialize orchestrator
    orchestrator = ModelOrchestrator()
    
    # Example user session
    user_session = [
        {'product_name': 'Wireless Headphones', 'action': 'view', 'timestamp': '2024-01-01 10:00:00'},
        {'product_name': 'Smartphone Case', 'action': 'view', 'timestamp': '2024-01-01 10:05:00'},
        {'product_name': 'Wireless Headphones', 'action': 'add_to_cart', 'timestamp': '2024-01-01 10:10:00'}
    ]
    
    # Example supply chain context
    supply_context = {
        'product_id': 'WH001',
        'warehouse_location': 'US_WEST',
        'current_inventory': 150,
        'historical_delay_rate': 0.15,
        'carbon_cost_factor': 0.3
    }
    
    # Run full pipeline
    results = orchestrator.orchestrate_full_pipeline(user_session, supply_context)
    
    # Display results
    print("\nPIPELINE RESULTS")
    print("="*50)
    print(f"Processing Time: {results['processing_time_seconds']:.3f}s")
    print(f"User Intent: {results['pipeline_summary']['user_intent_level']}")
    print(f"Supply Risk: {results['pipeline_summary']['supply_risk_level']}")
    print(f"Decision: {results['pipeline_summary']['recommended_action']}")
    print(f"Confidence: {results['pipeline_summary']['confidence_score']:.1%}")
    print(f"Reasoning: {results['final_decision']['reasoning']}")

if __name__ == '__main__':
    main()
