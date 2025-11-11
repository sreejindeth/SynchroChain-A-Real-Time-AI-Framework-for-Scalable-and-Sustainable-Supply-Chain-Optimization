"""
SynchroChain Production System - Hybrid AI Architecture
Combines working ML models with rule-based estimators for complete supply chain optimization

UPDATED: DelayRiskGNN now uses classification model (gnn_classification.pth)
Returns continuous probability (0-1) suitable for PPO agent without retraining.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class IntentTransformer:
    """Production Intent Transformer for user behavior prediction."""
    
    def __init__(self, model_path='models/smart_balanced_intent_transformer.pth', 
                 encoder_path='models/intent_encoders.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.encoders = self._load_encoders(encoder_path)
        self.intent_labels = ['High Intent', 'Medium Intent', 'Low Intent']
        self.intent_map = {'high': 0, 'medium': 1, 'low': 2}
        
    def _load_model(self, model_path):
        """Load the trained Smart Intent Transformer model."""
        try:
            # Import the model architecture
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
            from Intent_Transformer_Smart_Balanced import SmartIntentTransformer
            
            # Infer vocab_size from multiple sources and use the maximum
            vocab_size_sources = []
            
            # Source 1: Saved model's embedding layer
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if 'embedding.weight' in checkpoint:
                        ckpt_vocab = checkpoint['embedding.weight'].shape[0]
                        vocab_size_sources.append(('checkpoint', ckpt_vocab))
                        print(f"   📊 Checkpoint embedding vocab_size={ckpt_vocab}")
                except Exception as e:
                    print(f"   ⚠️ Could not infer vocab_size from checkpoint: {e}")
            
            # Source 2: Encoders file
            try:
                encoder_path = 'models/intent_encoders.pkl'
                with open(encoder_path, 'rb') as f:
                    temp_encoders = pickle.load(f)
                action_to_idx = temp_encoders.get('action_to_idx', {})
                if action_to_idx:
                    enc_vocab = len(action_to_idx)
                    vocab_size_sources.append(('encoders', enc_vocab))
                    print(f"   📊 Encoders vocab_size={enc_vocab}")
            except Exception as e:
                print(f"   ⚠️ Could not load encoders: {e}")
            
            # Use the maximum vocab_size found, or default to 6
            if vocab_size_sources:
                vocab_size = max(size for _, size in vocab_size_sources)
                source_info = ', '.join(f"{src}={size}" for src, size in vocab_size_sources)
                print(f"   ✅ Using vocab_size={vocab_size} (max of: {source_info})")
            else:
                vocab_size = 6  # Safe default for typical actions
                print(f"   ⚠️ Using default vocab_size={vocab_size}")
            
            # Create model with correct architecture
            model = SmartIntentTransformer(
                vocab_size=vocab_size,
                d_model=112,
                nhead=4,
                num_layers=3,
                dropout=0.35,
                num_features=3  # [duration, unique_pages, action_density]
            ).to(self.device)
            
            # Load trained weights (use strict=False to handle vocab_size mismatch)
            if os.path.exists(model_path):
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if missing or unexpected:
                        print(f"   ⚠️ Partial load: missing={len(missing)}, unexpected={len(unexpected)}")
                        if 'embedding.weight' in missing or 'embedding.weight' in unexpected:
                            print(f"   ⚠️ Embedding size mismatch - using initialized weights")
                    model.eval()
                    print(f"✅ Loaded Intent Transformer from {model_path}")
                    return model
                except Exception as e:
                    print(f"⚠️ Error loading model: {e}")
                    return None
            else:
                print(f"⚠️ Model file not found: {model_path}, using placeholder")
                return None
        except Exception as e:
            print(f"⚠️ Error loading model: {e}, using placeholder")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_encoders(self, encoder_path):
        """Load the action encoders."""
        try:
            with open(encoder_path, 'rb') as f:
                encoders = pickle.load(f)
            print(f"✅ Loaded encoders from {encoder_path}")
            return encoders
        except Exception as e:
            print(f"⚠️ Error loading encoders from {encoder_path}: {e}")
            print("⚠️ Using default encoders")
            return {'action_to_idx': {'view': 0, 'search': 1, 'add_to_cart': 2, 'like': 3, 'checkout': 4}}
    
    def _compute_session_features(self, user_session: List[str]) -> np.ndarray:
        """Compute session-level features: duration, unique_pages, action_density."""
        # These are simplified approximations since we don't have timestamps in production
        # For production, use heuristics based on session length
        
        unique_actions = len(set(user_session))
        total_actions = len(user_session) if user_session else 1
        
        # Approximate features (normalized similarly to training)
        # Session duration: approximate from action count
        session_duration_log = np.log1p(total_actions) / np.log(51)  # Normalized to ~[0, 1]
        
        # Unique pages viewed: normalized
        unique_pages = min(1.0, unique_actions / 10.0)
        
        # Action density: actions per "session unit"
        action_density = min(1.0, total_actions / 20.0)
        
        return np.array([session_duration_log, unique_pages, action_density], dtype=np.float32)
    
    def _compute_urgency(self, predicted_intent: str, add_to_cart_ratio: float) -> float:
        """
        Compute urgency from predicted intent and add_to_cart ratio.
        Matches the logic used in training data preparation.
        """
        if predicted_intent.lower() == 'high' or 'High' in predicted_intent:
            return min(0.9, 0.4 + add_to_cart_ratio)
        elif predicted_intent.lower() == 'medium' or 'Medium' in predicted_intent:
            return 0.2 + add_to_cart_ratio * 0.5
        else:  # low
            return add_to_cart_ratio * 0.4
    
    def predict(self, user_session: List[str]) -> Tuple[float, float]:
        """
        Predict user intent and urgency from session data.
        
        Args:
            user_session: List of user actions in chronological order
            
        Returns:
            intent_score: Probability of high intent (0-1)
            urgency_level: How soon the purchase might happen (0-1)
        """
        if self.model is None:
            # Fallback to rule-based estimation
            add_to_cart_count = user_session.count('add_to_cart')
            total_actions = len(user_session) if user_session else 1
            ratio = add_to_cart_count / total_actions
            
            # Simple heuristic
            intent_score = min(1.0, ratio * 3.0)
            urgency_level = min(1.0, len(user_session) * 0.1)
            return intent_score, urgency_level
        
        # Filter out 'add_to_cart' from input (prevents data leak)
        filtered_session = [a for a in user_session if a != 'add_to_cart']
        
        # Calculate add_to_cart_ratio for urgency computation (but don't use in model input)
        add_to_cart_count = user_session.count('add_to_cart')
        total_actions = len(user_session) if user_session else 1
        add_to_cart_ratio = add_to_cart_count / total_actions
        
        # Convert actions to indices
        action_to_idx = self.encoders.get('action_to_idx', {})
        action_indices = []
        for action in filtered_session:
            if action in action_to_idx:
                action_indices.append(action_to_idx[action])
            else:
                # Unknown action - use 0 (padding) or skip
                action_indices.append(0)
        
        # Pad or truncate to max length (50 from training, but using 12 for production compatibility)
        max_length = 50
        if len(action_indices) > max_length:
            action_indices = action_indices[:max_length]
        elif len(action_indices) < max_length:
            action_indices.extend([0] * (max_length - len(action_indices)))
        
        # Compute session features
        session_features = self._compute_session_features(user_session)
        
        # Convert to tensor
        sequence = torch.tensor([action_indices], dtype=torch.long).to(self.device)
        features = torch.tensor([session_features], dtype=torch.float32).to(self.device)
        
        # Predict intent
        with torch.no_grad():
            intent_logits = self.model(sequence, features)
            intent_probs = torch.softmax(intent_logits, dim=1)
            predicted_intent_idx = torch.argmax(intent_probs, dim=1).item()
            predicted_intent = self.intent_labels[predicted_intent_idx]
            
            # Get probability of high intent as intent_score
            # Intent order: ['High Intent', 'Medium Intent', 'Low Intent']
            high_intent_prob = intent_probs[0][0].item()  # Index 0 = High Intent
            intent_score = high_intent_prob
        
        # Debug output
        print(f"DEBUG Intent: Session={len(user_session)} actions, Filtered={len(filtered_session)}, add_to_cart_ratio={add_to_cart_ratio:.3f}, Predicted={predicted_intent}, Intent Score={intent_score:.3f}")
        
        # Compute urgency from predicted intent and add_to_cart_ratio
        urgency_level = self._compute_urgency(predicted_intent, add_to_cart_ratio)
        
        return float(intent_score), float(urgency_level)

class DelayRiskGNN:
    """Production Delay Risk GNN for supply chain risk prediction (Binary Classification).
    
    Uses the classification GNN model which outputs continuous probability (0-1)
    suitable for both classification metrics and PPO agent input.
    """
    
    def __init__(self, model_path='models/gnn_classification.pth',
                 scaler_path='models/gnn_scalers.pkl',
                 encoder_path='models/gnn_label_encoders.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        self.label_encoders = self._load_label_encoders(encoder_path)
        
    def _load_model(self, model_path):
        """Load the trained delay risk GNN classification model."""
        try:
            # Define model architecture (must match training)
            from sklearn.preprocessing import LabelEncoder
            
            class GraphAttentionLayer(nn.Module):
                def __init__(self, in_dim, out_dim, dropout=0.1):
                    super().__init__()
                    self.W = nn.Linear(in_dim, out_dim)
                    self.dropout = nn.Dropout(dropout)
                    nn.init.xavier_uniform_(self.W.weight)
                
                def forward(self, x):
                    return self.dropout(self.W(x))
            
            class DelayClassificationGNN(nn.Module):
                def __init__(self, input_dim=9, hidden_dim=64, num_layers=3, dropout=0.1):
                    super().__init__()
                    self.input_proj = nn.Linear(input_dim, hidden_dim)
                    self.gat_layers = nn.ModuleList([
                        GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
                        for _ in range(num_layers)
                    ])
                    self.output_layers = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    x = self.input_proj(x)
                    for gat_layer in self.gat_layers:
                        x = gat_layer(x)
                        x = torch.relu(x)
                    return self.output_layers(x)
            
            model = DelayClassificationGNN(input_dim=9, hidden_dim=64, num_layers=3, dropout=0.1)
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                print(f"✅ Loaded Delay Risk GNN Classification Model from {model_path}")
                return model
            else:
                print(f"⚠️ Model file not found: {model_path}, using placeholder")
                return None
                
        except Exception as e:
            print(f"⚠️ Error loading model: {e}, using placeholder")
            return None
    
    def _load_scaler(self, scaler_path):
        """Load the feature scaler."""
        try:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✅ Loaded scaler from {scaler_path}")
                return scaler.get('feature_scaler') if isinstance(scaler, dict) else scaler
            else:
                print("⚠️ Scaler file not found, using default")
                return None
        except Exception as e:
            print(f"⚠️ Error loading scaler: {e}")
            return None
    
    def _load_label_encoders(self, encoder_path):
        """Load label encoders for categorical features."""
        try:
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    encoders = pickle.load(f)
                print(f"✅ Loaded label encoders from {encoder_path}")
                return encoders
            else:
                print("⚠️ Label encoders not found, will use default encoding")
                return {}
        except Exception as e:
            print(f"⚠️ Error loading label encoders: {e}")
            return {}
    
    def predict(self, order_data: Dict) -> float:
        """
        Predict delay risk for an order.
        
        Args:
            order_data: Dictionary containing order information
            
        Returns:
            delay_risk: Continuous probability of delay (0.0-1.0)
                        This is the raw Sigmoid output before thresholding.
                        For binary classification: threshold at 0.5
                        For PPO agent: use continuous probability directly
        """
        if self.model is None:
            # Fallback to rule-based estimation
            return self._fallback_prediction(order_data)
        
        # Extract and encode features
        features = self._extract_features(order_data)
        
        # Scale features using StandardScaler
        # The scaler should handle all 9 features (6 numerical + 3 categorical)
        if self.scaler is not None:
            try:
                # Try scaling all 9 features
                features_scaled = self.scaler.transform([features])
                features = features_scaled
                # print(f"✅ Scaled {len(features[0])} features with StandardScaler")
            except ValueError as e:
                # Fallback: If scaler expects different number of features, use manual normalization
                print(f"⚠️ Scaler error: {e}. Using manual normalization.")
                numerical_features = features[:6]
                categorical_features = features[6:]
                
                # Manual normalization as fallback
                normalized = [
                    numerical_features[0] / 500.0,  # Product Price
                    numerical_features[1] / 10.0,   # Order Quantity
                    numerical_features[2],          # Discount Rate (already 0-1)
                    numerical_features[3],          # Profit Ratio (already 0-1)
                    numerical_features[4] / 2000.0, # Sales
                    numerical_features[5] / 7.0,    # Days for shipment
                ]
                features = np.array([normalized + list(categorical_features)])
        else:
            # No scaler available - use manual normalization
            print("⚠️ No scaler available. Using manual normalization.")
            numerical_features = features[:6]
            categorical_features = features[6:]
            normalized = [
                numerical_features[0] / 500.0,
                numerical_features[1] / 10.0,
                numerical_features[2],
                numerical_features[3],
                numerical_features[4] / 2000.0,
                numerical_features[5] / 7.0,
            ]
            features = np.array([normalized + list(categorical_features)])
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Predict (returns continuous probability 0-1)
        with torch.no_grad():
            prediction = self.model(features_tensor)
            delay_risk_prob = prediction.item()
        
        # Debug output
        print(f"DEBUG GNN: Features={[f'{f:.3f}' for f in features[0][:6]]}, Delay Risk={delay_risk_prob:.4f}")
        
        return float(delay_risk_prob)
    
    def _extract_features(self, order_data: Dict) -> List[float]:
        """Extract and encode features matching the classification model.
        
        Features (9 total):
        1. Product Price
        2. Order Item Quantity
        3. Order Item Discount Rate
        4. Order Item Profit Ratio
        5. Sales
        6. Days for shipment (scheduled) - the PLAN (not actual)
        7. Customer Segment (encoded)
        8. Shipping Mode (encoded)
        9. Market (encoded)
        """
        # Numerical features
        product_price = order_data.get('product_price', order_data.get('Product Price', 0))
        order_quantity = order_data.get('order_quantity', order_data.get('Order Item Quantity', 1))
        discount_rate = order_data.get('discount_rate', order_data.get('Order Item Discount Rate', 0))
        profit_ratio = order_data.get('profit_ratio', order_data.get('Order Item Profit Ratio', 0))
        sales = order_data.get('sales', order_data.get('Sales', 0))
        days_scheduled = order_data.get('days_for_shipment', order_data.get('Days for shipment (scheduled)', 4))
        
        # Categorical features (encode if encoders available)
        customer_segment = order_data.get('customer_segment', order_data.get('Customer Segment', 'Consumer'))
        shipping_mode = order_data.get('shipping_mode', order_data.get('Shipping Mode', 'Standard Class'))
        market = order_data.get('market', order_data.get('Market', 'Pacific Asia'))
        
        # Encode categorical features
        if self.label_encoders:
            customer_segment_encoded = self._encode_categorical('Customer Segment', customer_segment, default=0)
            shipping_mode_encoded = self._encode_categorical('Shipping Mode', shipping_mode, default=2)
            market_encoded = self._encode_categorical('Market', market, default=3)
        else:
            # Default encoding if encoders not available
            segment_map = {'Consumer': 0, 'Corporate': 1, 'Home Office': 2}
            shipping_map = {'Standard Class': 0, 'Second Class': 1, 'First Class': 2, 'Same Day': 3}
            market_map = {'Pacific Asia': 0, 'Southeast Asia': 1, 'South Asia': 2, 'Oceania': 3, 'Europe': 4}
            
            customer_segment_encoded = segment_map.get(str(customer_segment), 0)
            shipping_mode_encoded = shipping_map.get(str(shipping_mode), 0)
            market_encoded = market_map.get(str(market), 0)
        
        # Return features in the same order as training
        features = [
            float(product_price),
            float(order_quantity),
            float(discount_rate),
            float(profit_ratio),
            float(sales),
            float(days_scheduled),
            float(customer_segment_encoded),
            float(shipping_mode_encoded),
            float(market_encoded)
        ]
        
        # Debug: print raw features before scaling
        print(f"DEBUG GNN Extract: price={product_price:.2f}, qty={order_quantity}, disc={discount_rate:.3f}, profit={profit_ratio:.3f}, sales={sales:.2f}, days={days_scheduled}, seg={customer_segment_encoded}, ship={shipping_mode_encoded}, mkt={market_encoded}")
        
        return features
    
    def _encode_categorical(self, feature_name: str, value: str, default: int = 0) -> int:
        """Encode a categorical feature using saved label encoder."""
        if self.label_encoders and feature_name in self.label_encoders:
            try:
                # Handle unseen values
                if value in self.label_encoders[feature_name].classes_:
                    return int(self.label_encoders[feature_name].transform([value])[0])
                else:
                    return default
            except:
                return default
        return default
    
    def _fallback_prediction(self, order_data: Dict) -> float:
        """Fallback rule-based prediction if model not loaded."""
        delay_risk = 0.1  # Base risk
        if order_data.get('shipping_mode') == 'Same Day':
            delay_risk += 0.3
        if order_data.get('is_international', False):
            delay_risk += 0.2
        if order_data.get('order_value', 0) > 1000:
            delay_risk += 0.1
        return min(1.0, delay_risk)

class InventoryRuleEngine:
    """Rule-based inventory level estimator."""
    
    def __init__(self):
        self.category_lead_times = {
            'Electronics': 14, 'Technology': 14, 'Computers': 14,
            'Fashion': 7, 'Clothing': 7, 'Apparel': 7,
            'Home': 21, 'Furniture': 21, 'Garden': 21,
            'Sports': 10, 'Outdoor': 10
        }
        
    def estimate(self, order_data: Dict) -> float:
        """
        Estimate inventory level using business rules.
        
        Args:
            order_data: Dictionary containing order information
            
        Returns:
            inventory_level: Estimated inventory level (0-1)
        """
        category = order_data.get('category_name', 'Unknown')
        order_quantity = order_data.get('order_quantity', 1)
        order_value = order_data.get('order_value', 0)
        
        # Base inventory level
        inventory_level = 0.5
        
        # Adjust based on category
        if category in ['Electronics', 'Technology']:
            inventory_level = 0.3  # Low inventory for electronics
        elif category in ['Fashion', 'Clothing']:
            inventory_level = 0.7  # High inventory for fashion
        elif category in ['Home', 'Furniture']:
            inventory_level = 0.4  # Medium inventory for home goods
        
        # Adjust based on order characteristics
        if order_quantity > 5:
            inventory_level -= 0.2  # Large orders reduce inventory
        if order_value > 500:
            inventory_level -= 0.1  # High-value orders reduce inventory
        
        # Adjust based on lead time
        lead_time = self.category_lead_times.get(category, 10)
        if lead_time > 14:
            inventory_level += 0.2  # Longer lead times = higher inventory
        
        return max(0.0, min(1.0, inventory_level))

class CarbonRuleEngine:
    """Rule-based carbon cost estimator."""
    
    def __init__(self):
        self.shipping_emissions = {
            'Same Day': 3.0,
            'First Class': 2.5,
            'Second Class': 2.0,
            'Standard Class': 1.0
        }
        
        self.category_weights = {
            'Electronics': 2.0,
            'Fashion': 0.5,
            'Home': 5.0,
            'Sports': 1.5
        }
    
    def estimate(self, order_data: Dict) -> float:
        """
        Estimate carbon cost using business rules.
        
        Args:
            order_data: Dictionary containing order information
            
        Returns:
            carbon_cost: Estimated carbon cost (0-1)
        """
        shipping_mode = order_data.get('shipping_mode', 'Standard Class')
        category = order_data.get('category_name', 'Unknown')
        order_quantity = order_data.get('order_quantity', 1)
        is_international = order_data.get('is_international', False)
        
        # Base carbon cost
        carbon_cost = 0.3
        
        # Adjust based on shipping mode
        shipping_factor = self.shipping_emissions.get(shipping_mode, 1.0)
        carbon_cost *= shipping_factor
        
        # Adjust based on product weight
        weight_factor = self.category_weights.get(category, 1.0)
        carbon_cost *= weight_factor
        
        # Adjust based on quantity
        carbon_cost *= (1 + order_quantity * 0.1)
        
        # Adjust based on distance
        if is_international:
            carbon_cost *= 2.0
        
        return min(1.0, carbon_cost)

class PPOAgent:
    """PPO Reinforcement Learning Agent for supply chain decisions."""
    
    def __init__(self, model_path='models/ppo_agent_final.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.action_map = {
            0: {'pre_allocate': True, 'restock': False},
            1: {'pre_allocate': False, 'restock': True},
            2: {'pre_allocate': False, 'restock': False},  # Expedite shipping
            3: {'pre_allocate': False, 'restock': False}  # Normal operation
        }
        self.use_fallback = (self.model is None)
        
    def _load_model(self, model_path):
        """Load the trained PPO agent model."""
        try:
            # Import the ImprovedPPOAgent architecture from training script
            # The trained model uses ImprovedPPOAgent (not PPOAgent from PPO_Training_Implementation)
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            # Import from the training script where ImprovedPPOAgent is defined
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'training'))
            
            try:
                # Try importing ImprovedPPOAgent from train_ppo_final.py
                from train_ppo_final import ImprovedPPOAgent
                TrainedPPOAgent = ImprovedPPOAgent
            except ImportError:
                # Fallback: Define the architecture inline to match training
                import torch.nn as nn
                class ImprovedPPOAgent(nn.Module):
                    """Improved PPO with better architecture - matches training."""
                    def __init__(self, state_dim, action_dim, hidden_dim=256):
                        super().__init__()
                        # Shared layers with LayerNorm
                        self.shared = nn.Sequential(
                            nn.Linear(state_dim, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                            nn.ReLU()
                        )
                        # Actor
                        self.actor = nn.Sequential(
                            nn.Linear(hidden_dim, action_dim),
                            nn.Softmax(dim=-1)
                        )
                        # Critic
                        self.critic = nn.Linear(hidden_dim, 1)
                    def forward(self, state):
                        shared = self.shared(state)
                        action_probs = self.actor(shared)
                        value = self.critic(shared)
                        return action_probs, value
                    def get_action(self, state):
                        with torch.no_grad():
                            probs, _ = self.forward(state)
                            # Use argmax for deterministic decisions in production
                            action = torch.argmax(probs, dim=1)
                            return action.item()
                TrainedPPOAgent = ImprovedPPOAgent
            
            # PPO model uses 8-dimensional state
            # State dim: 8 (intent, urgency, delay_risk, inventory, carbon, order_value, customer_priority, supplier_reliability)
            state_dim = 8
            action_dim = 4  # 4 discrete actions
            hidden_dim = 256  # Match training configuration (ImprovedPPOAgent uses 256, not 128)
            
            # Create model with correct architecture
            model = TrainedPPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(self.device)
            
            # Load trained weights
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                print(f"✅ Loaded PPO Agent from {model_path}")
                return model
            else:
                print(f"⚠️ PPO model file not found: {model_path}, using rule-based fallback")
                return None
        except Exception as e:
            print(f"⚠️ Error loading PPO model: {e}, using rule-based fallback")
            import traceback
            traceback.print_exc()
            return None
    
    def _state_to_tensor(self, state: Dict) -> torch.Tensor:
        """Convert production state dict to model state tensor (8-dim)."""
        # Production state has 5 dimensions
        intent_score = float(state.get('intent_score', 0))
        urgency = float(state.get('urgency_level', state.get('urgency', 0)))
        delay_risk = float(state.get('delay_risk', 0))
        inventory_level = float(state.get('inventory_level', 0.5))
        carbon_cost = float(state.get('carbon_cost', 0.5))
        
        # Model expects 8 dimensions, pad with defaults
        # order_value should already be normalized (0-1), but clamp to ensure it's in range
        order_value = float(state.get('order_value', 0.5))
        order_value = max(0.0, min(1.0, order_value))  # Ensure normalized
        
        customer_priority = float(state.get('customer_priority', 0.5))
        supplier_reliability = float(state.get('supplier_reliability', 0.8))
        
        state_vector = np.array([
            intent_score,
            urgency,
            delay_risk,
            inventory_level,
            carbon_cost,
            order_value,
            customer_priority,
            supplier_reliability
        ], dtype=np.float32)
        
        return torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
    
    def _action_to_output(self, action_idx: int, state: Dict) -> Dict:
        """Convert model action index to production action format."""
        base_action = self.action_map.get(action_idx, self.action_map[3])  # Default to normal
        
        # Get base decisions
        action = {
            'pre_allocate': base_action.get('pre_allocate', False),
            'restock': base_action.get('restock', False),
            'shipping_mode': 'Standard Class',
            'priority': 'Normal'
        }
        
        # Map action to shipping mode and priority based on action type
        urgency = state.get('urgency_level', state.get('urgency', 0))
        delay_risk = state.get('delay_risk', 0)
        intent_score = state.get('intent_score', 0)
        carbon_cost = state.get('carbon_cost', 0.5)
        
        if action_idx == 2:  # Expedite shipping
            if urgency > 0.8 and delay_risk < 0.3:
                action['shipping_mode'] = 'Same Day'
            elif urgency > 0.6 and delay_risk < 0.5:
                action['shipping_mode'] = 'First Class'
            else:
                action['shipping_mode'] = 'First Class'  # Expedited option
        else:
            # Determine shipping mode from state
            if urgency > 0.8 and delay_risk < 0.3:
                action['shipping_mode'] = 'Same Day'
            elif urgency > 0.6 and delay_risk < 0.5:
                action['shipping_mode'] = 'First Class'
            elif carbon_cost < 0.3:
                action['shipping_mode'] = 'Standard Class'
            else:
                action['shipping_mode'] = 'Second Class'
        
        # Determine priority
        if intent_score > 0.8 and urgency > 0.7:
            action['priority'] = 'High'
        elif intent_score > 0.6:
            action['priority'] = 'Medium'
        else:
            action['priority'] = 'Low'
        
        return action
    
    def _rule_based_decide(self, state: Dict) -> Dict:
        """Fallback rule-based decision making (used when model not available)."""
        intent_score = state.get('intent_score', 0)
        urgency = state.get('urgency_level', state.get('urgency', 0))
        delay_risk = state.get('delay_risk', 0)
        inventory_level = state.get('inventory_level', 0.5)
        carbon_cost = state.get('carbon_cost', 0.5)
        
        # Decision logic based on state
        action = {
            'pre_allocate': False,
            'restock': False,
            'shipping_mode': 'Standard Class',
            'priority': 'Normal'
        }
        
        # Pre-allocation decision
        if intent_score > 0.7 and urgency > 0.6:
            action['pre_allocate'] = True
        
        # Restocking decision
        if inventory_level < 0.3 and intent_score > 0.5:
            action['restock'] = True
        
        # Shipping mode decision
        if urgency > 0.8 and delay_risk < 0.3:
            action['shipping_mode'] = 'Same Day'
        elif urgency > 0.6 and delay_risk < 0.5:
            action['shipping_mode'] = 'First Class'
        elif carbon_cost < 0.3:
            action['shipping_mode'] = 'Standard Class'
        else:
            action['shipping_mode'] = 'Second Class'
        
        # Priority decision
        if intent_score > 0.8 and urgency > 0.7:
            action['priority'] = 'High'
        elif intent_score > 0.6:
            action['priority'] = 'Medium'
        else:
            action['priority'] = 'Low'
        
        return action
    
    def decide(self, state: Dict) -> Dict:
        """
        Make supply chain decisions using trained PPO agent.
        
        Args:
            state: Dictionary containing all predictions and estimates
            
        Returns:
            action: Dictionary containing the decided action
        """
        if self.use_fallback or self.model is None:
            return self._rule_based_decide(state)
        
        try:
            # Convert state to tensor
            state_tensor = self._state_to_tensor(state)
            
            # Get action probabilities for debugging
            with torch.no_grad():
                probs, _ = self.model.forward(state_tensor)
                action_probs = probs.cpu().numpy()[0]
            
            # Get action from model (returns only action index)
            action_idx = self.model.get_action(state_tensor)
            
            # Debug output
            print(f"DEBUG PPO: State={[f'{v:.3f}' for v in state_tensor[0].cpu().numpy()]}, Action Probs={[f'{p:.3f}' for p in action_probs]}, Action={action_idx}")
            
            # Convert to production action format
            action = self._action_to_output(action_idx, state)
            
            return action
        except Exception as e:
            print(f"⚠️ Error in PPO decision making: {e}, falling back to rule-based")
            import traceback
            traceback.print_exc()
            return self._rule_based_decide(state)

class SynchroChainProduction:
    """Main SynchroChain production system."""
    
    def __init__(self):
        print("🚀 Initializing SynchroChain Production System...")
        
        # Initialize AI models
        self.intent_transformer = IntentTransformer()
        self.delay_risk_gnn = DelayRiskGNN()
        
        # Initialize rule-based systems
        self.inventory_estimator = InventoryRuleEngine()
        self.carbon_estimator = CarbonRuleEngine()
        
        # Initialize RL agent
        self.rl_agent = PPOAgent()
        
        print("✅ SynchroChain Production System initialized successfully!")
    
    def process_user_session(self, user_session: List[str], order_context: Dict) -> Dict:
        """
        Process a user session and return supply chain decisions.
        
        Args:
            user_session: List of user actions
            order_context: Dictionary containing order information
            
        Returns:
            result: Dictionary containing predictions and decisions
        """
        print(f"📊 Processing user session with {len(user_session)} actions...")
        
        # AI Predictions
        intent_score, urgency_level = self.intent_transformer.predict(user_session)
        delay_risk = self.delay_risk_gnn.predict(order_context)
        
        # Rule-based Estimates
        inventory_level = self.inventory_estimator.estimate(order_context)
        carbon_cost = self.carbon_estimator.estimate(order_context)
        
        # Create state for RL agent
        state = {
            'intent_score': intent_score,
            'urgency_level': urgency_level,
            'delay_risk': delay_risk,
            'inventory_level': inventory_level,
            'carbon_cost': carbon_cost
        }
        
        # RL Decision
        action = self.rl_agent.decide(state)
        
        # Compile results
        result = {
            'predictions': {
                'intent_score': intent_score,
                'urgency_level': urgency_level,
                'delay_risk': delay_risk,
                'inventory_level': inventory_level,
                'carbon_cost': carbon_cost
            },
            'decisions': action,
            'confidence': {
                'intent': 'high' if intent_score > 0.7 else 'medium' if intent_score > 0.4 else 'low',
                'delay_risk': 'high' if delay_risk > 0.7 else 'medium' if delay_risk > 0.4 else 'low'
            }
        }
        
        print(f"✅ Session processed successfully!")
        print(f"   Intent Score: {intent_score:.3f}")
        print(f"   Urgency: {urgency_level:.3f}")
        print(f"   Delay Risk: {delay_risk:.3f}")
        print(f"   Action: {action['shipping_mode']} shipping, Priority: {action['priority']}")
        
        return result
    
    def batch_process(self, sessions: List[Tuple[List[str], Dict]]) -> List[Dict]:
        """
        Process multiple user sessions in batch.
        
        Args:
            sessions: List of (user_session, order_context) tuples
            
        Returns:
            results: List of processing results
        """
        print(f"📊 Processing {len(sessions)} sessions in batch...")
        
        results = []
        for i, (user_session, order_context) in enumerate(sessions):
            print(f"   Processing session {i+1}/{len(sessions)}...")
            result = self.process_user_session(user_session, order_context)
            results.append(result)
        
        print(f"✅ Batch processing completed!")
        return results

def main():
    """Demo the SynchroChain Production System."""
    print("=" * 80)
    print("🎯 SYNCHROCHAIN PRODUCTION SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize system
    system = SynchroChainProduction()
    
    # Demo scenarios
    demo_sessions = [
        # High-intent electronics session
        (['view', 'search', 'add_to_cart', 'view', 'add_to_cart'], {
            'category_name': 'Electronics',
            'shipping_mode': 'Standard Class',
            'order_quantity': 2,
            'order_value': 800,
            'is_international': False,
            'days_for_shipping': 3,
            'days_for_shipment': 2
        }),
        
        # Low-intent fashion session
        (['view', 'search'], {
            'category_name': 'Fashion',
            'shipping_mode': 'First Class',
            'order_quantity': 1,
            'order_value': 150,
            'is_international': True,
            'days_for_shipping': 5,
            'days_for_shipment': 3
        }),
        
        # Urgent home goods session
        (['view', 'add_to_cart', 'add_to_cart', 'add_to_cart'], {
            'category_name': 'Home',
            'shipping_mode': 'Same Day',
            'order_quantity': 3,
            'order_value': 1200,
            'is_international': False,
            'days_for_shipping': 1,
            'days_for_shipment': 1
        })
    ]
    
    # Process sessions
    results = system.batch_process(demo_sessions)
    
    # Display results
    print(f"\n{'='*80}")
    print("📊 PROCESSING RESULTS")
    print(f"{'='*80}")
    
    for i, result in enumerate(results):
        print(f"\nSession {i+1}:")
        print(f"  Intent Score: {result['predictions']['intent_score']:.3f}")
        print(f"  Urgency: {result['predictions']['urgency_level']:.3f}")
        print(f"  Delay Risk: {result['predictions']['delay_risk']:.3f}")
        print(f"  Inventory Level: {result['predictions']['inventory_level']:.3f}")
        print(f"  Carbon Cost: {result['predictions']['carbon_cost']:.3f}")
        print(f"  Decision: {result['decisions']['shipping_mode']} shipping")
        print(f"  Priority: {result['decisions']['priority']}")
        print(f"  Pre-allocate: {result['decisions']['pre_allocate']}")
        print(f"  Restock: {result['decisions']['restock']}")
    
    print(f"\n🎉 SynchroChain Production System demo completed!")

if __name__ == "__main__":
    main()
