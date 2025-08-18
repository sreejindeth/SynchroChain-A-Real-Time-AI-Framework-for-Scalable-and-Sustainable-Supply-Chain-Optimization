# src/models/rl_agent.py
"""
Reinforcement Learning Agent for SynchroChain Decision Making.
This agent takes intent/urgency from the Transformer and supply state
from the GNN to make decisions like fulfillment location, shipping mode, etc.
For Phase 2, we'll implement a rule-based/heuristic agent.
A full DRL agent would be implemented in a subsequent phase.
"""
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class State:
    """Represents the state passed to the RL agent."""
    intent_score: float
    urgency_score: float
    # GNN outputs for relevant nodes (e.g., product, potential warehouses)
    product_embedding: np.ndarray # Or a summary feature
    warehouse_states: Dict[str, Dict[str, float]] # e.g., {'WH1': {'risk': 0.2, 'inventory': 0.8, 'carbon': 0.5}}
    # Global features
    global_carbon_budget: Optional[float] = None # If applicable

@dataclass
class Action:
    """Represents the action decided by the RL agent."""
    fulfill_from_warehouse: str # Identifier for the chosen warehouse
    shipping_mode: str # e.g., 'Standard Class', 'Second Class'
    offer_bundle: bool
    offer_sustainability_choice: bool
    apply_dynamic_pricing: bool
    dynamic_price_multiplier: float = 1.0 # e.g., 1.05 for +5%

class RLAgent:
    """
    A heuristic-based RL agent for demonstration.
    A full DRL agent would learn a policy network.
    """
    def __init__(self, warehouse_ids, shipping_modes):
        self.warehouse_ids = warehouse_ids
        self.shipping_modes = shipping_modes
        # Define thresholds for decision making
        self.high_intent_threshold = 0.7
        self.high_urgency_threshold = 0.8
        self.low_risk_threshold = 0.3
        self.high_inventory_threshold = 0.6
        self.low_carbon_threshold = 0.6 # Lower is better

    def act(self, state: State) -> Action:
        """
        Selects an action based on the current state using heuristics.
        In a full DRL setup, this would involve a policy network.
        """
        # 1. Select Warehouse based on risk, inventory, carbon
        best_warehouse = None
        best_score = -np.inf
        for wh_id, wh_state in state.warehouse_states.items():
            # Simple scoring: prefer low risk, high inventory, low carbon
            # Negate risk and carbon as lower is better
            score = (
                -wh_state.get('risk', 0.5) * 2 + # Weight risk heavily
                wh_state.get('inventory', 0.5) +
                -wh_state.get('carbon_cost', 0.5) +
                random.uniform(-0.1, 0.1) # Small random factor
            )
            if score > best_score:
                best_score = score
                best_warehouse = wh_id

        # 2. Select Shipping Mode
        # Default to standard
        chosen_shipping_mode = 'Standard Class'
        if state.urgency_score > self.high_urgency_threshold:
            # If urgent, consider faster (and likely more expensive/carbon) modes
            # But still check carbon if a budget is set
            # Simplification: just pick Second Class for high urgency
            chosen_shipping_mode = 'Second Class'
            # A full agent might predict or look up the actual mode based on urgency/cost tradeoff

        # 3. Offer Bundle
        offer_bundle = state.intent_score > self.high_intent_threshold

        # 4. Offer Sustainability Choice
        # Always offer if carbon data is available, or based on some logic
        offer_sustainability = any('carbon_cost' in wh_state for wh_state in state.warehouse_states.values())

        # 5. Apply Dynamic Pricing
        apply_pricing = state.intent_score > self.high_intent_threshold and \
                        any(wh_state.get('inventory', 0.5) < 0.4 for wh_state in state.warehouse_states.values())
        price_multiplier = 1.0
        if apply_pricing:
            price_multiplier = 1.05 # Simple +5%

        # 6. Construct Action
        action = Action(
            fulfill_from_warehouse=best_warehouse,
            shipping_mode=chosen_shipping_mode,
            offer_bundle=offer_bundle,
            offer_sustainability_choice=offer_sustainability,
            apply_dynamic_pricing=apply_pricing,
            dynamic_price_multiplier=price_multiplier
        )
        return action

# --- Placeholder for Future DRL Agent ---
# class DRLAgent(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         # Define policy network (e.g., Actor-Critic)
#         pass
#     def forward(self, state):
#         # Output action probabilities or Q-values
#         pass
#     def act(self, state):
#         # Select action based on policy
#         pass
#     def learn(self, experiences):
#         # Update policy based on experiences (rewards, next states etc.)
#         pass
