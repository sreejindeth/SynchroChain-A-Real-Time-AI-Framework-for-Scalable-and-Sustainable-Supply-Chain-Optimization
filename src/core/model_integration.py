"""
Model Integration Module for SynchroChain AI Dashboard
Integrates all ML models (Intent Transformer, GNN, PPO) with the dashboard
"""
import torch
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    """Manages all ML models and their integration."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models."""
        try:
            # Load Intent Transformer
            self._load_intent_transformer()
            
            # Load GNN model
            self._load_gnn_model()
            
            # Load PPO Agent
            self._load_ppo_agent()
            
            # Check which models are genuine vs mock
            genuine_models = []
            mock_models = []
            
            if not isinstance(self.models.get('intent_transformer'), MockIntentTransformer):
                genuine_models.append('Intent Transformer')
            else:
                mock_models.append('Intent Transformer')
            
            if not isinstance(self.models.get('gnn'), MockDelayRiskGNN):
                genuine_models.append('Delay Risk GNN')
            else:
                mock_models.append('Delay Risk GNN')
            
            if not isinstance(self.models.get('ppo'), MockPPOAgent):
                genuine_models.append('PPO Agent')
            else:
                mock_models.append('PPO Agent')
            
            print("="*60)
            print("Model Loading Status:")
            print("="*60)
            if genuine_models:
                print(f"✅ Genuine Models Loaded: {', '.join(genuine_models)}")
            if mock_models:
                print(f"⚠️  Mock Models (Fallback): {', '.join(mock_models)}")
            print("="*60)
            
        except Exception as e:
            print(f"⚠️  Some models failed to load: {e}")
            print("⚠️  Using mock models as fallback")
            # Initialize with mock models for demo
            self._initialize_mock_models()
    
    def _load_intent_transformer(self):
        """Load the Intent Transformer model."""
        try:
            # Try to load actual model (LATEST VERSION)
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'production'))
            from SynchroChain_Production_System import IntentTransformer
            
            model_path = 'models/smart_balanced_intent_transformer.pth'
            encoder_path = 'models/intent_encoders.pkl'
            
            if os.path.exists(model_path):
                # Load actual production IntentTransformer
                self.models['intent_transformer'] = IntentTransformer(
                    model_path=model_path,
                    encoder_path=encoder_path
                )
                self.encoders['intent'] = self.models['intent_transformer'].encoders
                print("✅ Intent Transformer loaded (actual model)")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        except Exception as e:
            print(f"⚠️  Intent Transformer: {e}")
            print("⚠️  Falling back to mock Intent Transformer")
            print("⚠️  WARNING: Using fabricated values instead of genuine model!")
            self.models['intent_transformer'] = MockIntentTransformer()
            self.encoders['intent'] = self._get_default_encoders()
    
    def _load_gnn_model(self):
        """Load the GNN classification model."""
        try:
            # Updated to use classification model
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'production'))
            from SynchroChain_Production_System import DelayRiskGNN
            
            model_path = 'models/gnn_classification.pth'
            
            if os.path.exists(model_path):
                # Use actual classification model from production system
                self.models['gnn'] = DelayRiskGNN(
                    model_path=model_path,
                    scaler_path='models/gnn_scalers.pkl',
                    encoder_path='models/gnn_label_encoders.pkl'
                )
                print("GNN classification model loaded")
            else:
                raise FileNotFoundError(f"GNN classification model not found: {model_path}")
                
        except Exception as e:
            print(f"⚠️  GNN model: {e}")
            print("⚠️  Falling back to mock Delay Risk GNN")
            print("⚠️  WARNING: Using fabricated values instead of genuine model!")
            # Fallback to mock if actual model loading fails
            self.models['gnn'] = MockDelayRiskGNN()
            self.scalers['gnn'] = {}
    
    def _load_ppo_agent(self):
        """Load the PPO agent."""
        try:
            model_path = 'models/ppo_agent_final.pth'
            
            if os.path.exists(model_path):
                # Import and load the actual PPO agent from production system
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'production'))
                from SynchroChain_Production_System import PPOAgent
                
                ppo_agent = PPOAgent(model_path=model_path)
                if ppo_agent.model is not None:
                    self.models['ppo'] = ppo_agent
                    print("✅ PPO agent loaded (genuine model)")
                else:
                    raise RuntimeError("PPO model failed to load")
            else:
                raise FileNotFoundError(f"PPO model file not found: {model_path}")
                
        except Exception as e:
            print(f"⚠️  PPO agent: {e}")
            print("⚠️  Falling back to mock PPO Agent")
            self.models['ppo'] = MockPPOAgent()
    
    def _load_encoders(self, encoder_path):
        """Load encoders from file."""
        try:
            with open(encoder_path, 'rb') as f:
                return pickle.load(f)
        except:
            return self._get_default_encoders()
    
    def _load_scalers(self, scaler_path):
        """Load scalers from file."""
        try:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    
    def _get_default_encoders(self):
        """Get default encoders."""
        return {
            'action_to_idx': {
                'view': 0, 'search': 1, 'add_to_cart': 2, 
                'like': 3, 'checkout': 4, 'remove': 5
            }
        }
    
    def _initialize_mock_models(self):
        """Initialize mock models for demo purposes."""
        self.models['intent_transformer'] = MockIntentTransformer()
        self.models['gnn'] = MockDelayRiskGNN()
        self.models['ppo'] = MockPPOAgent()
        self.encoders['intent'] = self._get_default_encoders()
        self.scalers['gnn'] = {}
    
    def predict_intent(self, user_session: List[str]) -> Tuple[float, float]:
        """Predict user intent and urgency."""
        return self.models['intent_transformer'].predict(user_session)
    
    def predict_delay_risk(self, order_context: Dict) -> float:
        """Predict delay risk for an order."""
        return self.models['gnn'].predict(order_context)
    
    def get_rl_decision(self, state: Dict) -> Dict:
        """Get RL agent decision."""
        return self.models['ppo'].decide(state)
    
    def process_user_session(self, user_session: List[str], order_context: Dict) -> Dict:
        """Process a complete user session and return all predictions."""
        # Get intent prediction
        intent_score, urgency = self.predict_intent(user_session)
        
        # Get delay risk prediction
        delay_risk = self.predict_delay_risk(order_context)
        
        # Create state for RL agent
        # Normalize order_value to 0-1 range (max reasonable order value: 2000)
        raw_order_value = order_context.get('order_value', 0)
        normalized_order_value = min(1.0, raw_order_value / 2000.0) if raw_order_value > 0 else 0.0
        
        state = {
            'intent_score': intent_score,
            'urgency': urgency,
            'delay_risk': delay_risk,
            'inventory_level': order_context.get('inventory_level', 0.5),
            'carbon_cost': order_context.get('carbon_cost', 0.5),
            'order_value': normalized_order_value,  # Normalized to 0-1
            'customer_priority': order_context.get('customer_priority', 0.5),
            'supplier_reliability': order_context.get('supplier_reliability', 0.8)
        }
        
        # Get RL decision
        rl_decision = self.get_rl_decision(state)
        
        # Compile results
        result = {
            'intent_score': intent_score,
            'urgency': urgency,
            'delay_risk': delay_risk,
            'rl_decision': rl_decision,
            'state': state,
            'timestamp': datetime.now().isoformat()
        }
        
        return result

class MockIntentTransformer:
    """Mock Intent Transformer for demo purposes."""
    
    def __init__(self):
        self.encoders = {
            'action_to_idx': {
                'view': 0, 'search': 1, 'add_to_cart': 2, 
                'like': 3, 'checkout': 4, 'remove': 5
            }
        }
    
    def predict(self, user_session: List[str]) -> Tuple[float, float]:
        """Predict intent and urgency from session."""
        if not user_session:
            return 0.0, 0.0
        
        # Calculate intent score based on actions
        intent_score = 0.0
        urgency = 0.0
        
        # Base score from session length
        intent_score += min(len(user_session) * 0.08, 0.4)
        
        # Action-based scoring
        action_weights = {
            'view': 0.05,
            'search': 0.1,
            'add_to_cart': 0.3,
            'like': 0.08,
            'checkout': 0.4,
            'remove': -0.1
        }
        
        for action in user_session:
            if action in action_weights:
                intent_score += action_weights[action]
        
        # Urgency calculation
        urgency = min(1.0, len([a for a in user_session if a == 'add_to_cart']) * 0.2)
        urgency += min(0.3, len(user_session) * 0.02)
        
        # Add some randomness for realism
        intent_score += np.random.normal(0, 0.02)
        urgency += np.random.normal(0, 0.01)
        
        # Clamp values
        intent_score = max(0.0, min(1.0, intent_score))
        urgency = max(0.0, min(1.0, urgency))
        
        return intent_score, urgency

class MockDelayRiskGNN:
    """Mock Delay Risk GNN for demo purposes."""
    
    def predict(self, order_context: Dict) -> float:
        """Predict delay risk for an order."""
        base_risk = 0.1
        
        # Category-based risk
        category_risk = {
            'Electronics': 0.3,
            'Sports': 0.1,
            'Home': 0.2,
            'Fashion': 0.05
        }
        
        category = order_context.get('category_name', 'Unknown')
        if category in category_risk:
            base_risk += category_risk[category]
        
        # Order value risk
        order_value = order_context.get('order_value', 0)
        if order_value > 500:
            base_risk += 0.1
        elif order_value > 1000:
            base_risk += 0.2
        
        # Shipping mode risk
        shipping_mode = order_context.get('shipping_mode', 'Standard')
        if shipping_mode == 'Same Day':
            base_risk += 0.2
        elif shipping_mode == 'International':
            base_risk += 0.3
        
        # Quantity risk
        quantity = order_context.get('order_quantity', 1)
        if quantity > 5:
            base_risk += 0.1
        
        # Add randomness
        base_risk += np.random.normal(0, 0.05)
        
        return max(0.0, min(1.0, base_risk))

class MockPPOAgent:
    """Mock PPO Agent for demo purposes."""
    
    def __init__(self):
        self.action_map = {
            0: 'Pre-allocate inventory',
            1: 'Restock items',
            2: 'Expedite shipping',
            3: 'Normal operation'
        }
    
    def decide(self, state: Dict) -> str:
        """Make supply chain decision based on state."""
        intent_score = state.get('intent_score', 0)
        urgency = state.get('urgency', 0)
        delay_risk = state.get('delay_risk', 0)
        inventory_level = state.get('inventory_level', 0.5)
        order_value = state.get('order_value', 0)
        
        # Decision logic
        if intent_score > 0.7 and urgency > 0.6:
            return "Pre-allocate inventory"
        elif intent_score > 0.5 and delay_risk > 0.6:
            return "Expedite shipping"
        elif intent_score > 0.4 and inventory_level < 0.3:
            return "Restock items"
        elif order_value > 1000 and intent_score > 0.6:
            return "Expedite shipping"
        else:
            return "Normal operation"

def create_sample_order_context():
    """Create a sample order context for testing."""
    return {
        'category_name': np.random.choice(['Electronics', 'Sports', 'Home', 'Fashion']),
        'order_value': np.random.uniform(50, 1000),
        'order_quantity': np.random.randint(1, 10),
        'shipping_mode': np.random.choice(['Standard', 'Express', 'Same Day', 'International']),
        'inventory_level': np.random.uniform(0.2, 0.9),
        'carbon_cost': np.random.uniform(0.1, 0.8),
        'customer_priority': np.random.uniform(0.3, 1.0),
        'supplier_reliability': np.random.uniform(0.6, 1.0)
    }

def test_model_integration():
    """Test the model integration."""
    print("Testing Model Integration...")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Test with sample data
    user_session = ['view', 'search', 'add_to_cart', 'like', 'add_to_cart']
    order_context = create_sample_order_context()
    
    # Get predictions
    result = model_manager.process_user_session(user_session, order_context)
    
    print("Test Results:")
    print(f"Intent Score: {result['intent_score']:.3f}")
    print(f"Urgency: {result['urgency']:.3f}")
    print(f"Delay Risk: {result['delay_risk']:.3f}")
    print(f"RL Decision: {result['rl_decision']}")
    
    return result

if __name__ == "__main__":
    test_model_integration()
