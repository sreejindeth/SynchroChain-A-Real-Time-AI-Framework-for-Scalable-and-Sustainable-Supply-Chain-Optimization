# src/models/agents/ppo_trainer.py
"""
PPO RL Agent for SynchroChain Supply Chain Optimization
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import gymnasium as gym
from gymnasium import spaces
import time
import yaml  # Add this missing import

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

class SupplyChainEnv(gym.Env):
    """Supply Chain Environment for RL training."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Action space: 0=Standard, 1=Expedited, 2=Priority
        self.action_space = spaces.Discrete(3)
        
        # State space: [intent_signal, delay_risk, inventory_level, cost_factor]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        # Random initial state
        self.state = np.random.uniform(0, 1, 4).astype(np.float32)
        self.step_count = 0
        self.total_reward = 0
        return self.state
    
    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        self.step_count += 1
        
        # Calculate reward based on action and state
        intent_signal, delay_risk, inventory_level, cost_factor = self.state
        
        # Reward calculation
        if action == 0:  # Standard
            reward = intent_signal * 10 - delay_risk * 5 - cost_factor * 2
        elif action == 1:  # Expedited
            reward = intent_signal * 15 - delay_risk * 3 - cost_factor * 4
        else:  # Priority
            reward = intent_signal * 20 - delay_risk * 1 - cost_factor * 6
        
        # Add inventory bonus
        if inventory_level > 0.5:
            reward += 5
        
        self.total_reward += reward
        
        # Update state (simplified dynamics)
        self.state = np.random.uniform(0, 1, 4).astype(np.float32)
        
        # Episode termination
        done = self.step_count >= 100
        
        info = {
            'intent_signal': intent_signal,
            'delay_risk': delay_risk,
            'inventory_level': inventory_level,
            'cost_factor': cost_factor
        }
        
        return self.state, reward, done, info

class PPOPolicy(nn.Module):
    """PPO Policy Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Forward pass through policy and value networks."""
        logits = self.policy_net(state)
        value = self.value_net(state)
        return logits, value
    
    def get_action(self, state):
        """Get action and log probability."""
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

class PPOTrainer:
    """PPO Training class."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Environment
        self.env = SupplyChainEnv(config)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        # Policy network
        self.policy = PPOPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=float(config['learning_rate']))
        
        # Training parameters
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.eps_clip = config['eps_clip']
        self.value_coef = config['value_coef']
        self.entropy_coef = config['entropy_coef']
        
        # Model save path
        self.model_save_path = os.path.join(project_root, 'models', 'ppo_agent.pth')
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, List]:
        """Collect rollouts for training."""
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        state = self.env.reset()
        
        for _ in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_tensor)
            
            next_state, reward, done, _ = self.env.step(action.item())
            
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(done)
            
            state = next_state
            
            if done:
                state = self.env.reset()
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'values': values,
            'dones': dones
        }
    
    def compute_returns(self, rewards: List[float], dones: List[bool], gamma: float) -> List[float]:
        """Compute discounted returns."""
        returns = []
        running_return = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                running_return = 0
            running_return = rewards[i] + gamma * running_return
            returns.insert(0, running_return)
        
        return returns
    
    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy performance with comprehensive metrics."""
        total_rewards = []
        episode_lengths = []
        action_distributions = [0, 0, 0]  # Standard, Expedited, Priority
        value_predictions = []
        policy_entropies = []
        cost_efficiency = []
        risk_mitigation = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_cost = 0
            episode_risk_handled = 0
            
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, value = self.policy(state_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum()
                    action = torch.multinomial(probs, 1).item()
                
                # Track metrics
                action_distributions[action] += 1
                value_predictions.append(value.item())
                policy_entropies.append(entropy.item())
                
                # Calculate cost and risk metrics
                intent_signal, delay_risk, inventory_level, cost_factor = state
                
                if action == 0:  # Standard
                    episode_cost += cost_factor * 2
                elif action == 1:  # Expedited  
                    episode_cost += cost_factor * 4
                else:  # Priority
                    episode_cost += cost_factor * 6
                
                # Risk mitigation score
                if delay_risk > 0.5 and action >= 1:  # Used expedited/priority for high risk
                    episode_risk_handled += 1
                elif delay_risk <= 0.5 and action == 0:  # Used standard for low risk
                    episode_risk_handled += 1
                
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            cost_efficiency.append(episode_reward / max(episode_cost, 1))  # Reward per cost
            risk_mitigation.append(episode_risk_handled / max(episode_steps, 1))  # Risk handling ratio
        
        # Normalize action distribution
        total_actions = sum(action_distributions)
        if total_actions > 0:
            action_distributions = [count / total_actions for count in action_distributions]
        
        return {
            'avg_reward': np.mean(total_rewards),
            'reward_std': np.std(total_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'avg_value_prediction': np.mean(value_predictions),
            'avg_policy_entropy': np.mean(policy_entropies),
            'action_distribution': action_distributions,
            'cost_efficiency': np.mean(cost_efficiency),
            'risk_mitigation_score': np.mean(risk_mitigation),
            'reward_variance': np.var(total_rewards),
            'max_reward': np.max(total_rewards),
            'min_reward': np.min(total_rewards)
        }

    def train(self) -> Dict[str, float]:
        """Train the PPO agent."""
        print("Starting PPO training for SynchroChain Supply Chain Optimization...")
        print("Training Configuration:")
        print(f"   - Epochs: {self.epochs}")
        print(f"   - Rollout Steps: {self.config['rollout_steps']}")
        print(f"   - Learning Rate: {float(self.config['learning_rate'])}")
        print(f"   - Device: {self.device}")
        print("-" * 60)
        
        episode_rewards = []
        policy_losses = []
        value_losses = []
        performance_history = []
        
        for epoch in range(self.epochs):
            # Collect rollouts
            rollouts = self.collect_rollouts(self.config['rollout_steps'])
            
            # Compute returns
            returns = self.compute_returns(rollouts['rewards'], rollouts['dones'], self.gamma)
            
            # Convert to tensors
            states = torch.FloatTensor(rollouts['states']).to(self.device)
            actions = torch.LongTensor(rollouts['actions']).to(self.device)
            old_log_probs = torch.FloatTensor(rollouts['log_probs']).to(self.device)
            old_values = torch.FloatTensor(rollouts['values']).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # PPO updates
            for _ in range(self.config['ppo_epochs']):
                # Get current policy outputs
                logits, values = self.policy(states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Compute ratios
                ratios = torch.exp(new_log_probs - old_log_probs)
                
                # Compute advantages
                advantages = returns - values.squeeze()
                
                # PPO loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), returns)
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
            
            # Log metrics
            episode_reward = sum(rollouts['rewards'])
            episode_rewards.append(episode_reward)
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            
            # Evaluate policy every 10 epochs
            if epoch % 10 == 0:
                eval_metrics = self.evaluate_policy(num_episodes=5)
                performance_history.append(eval_metrics)
                
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                print(f"\n=== Epoch {epoch:3d}/{self.epochs} Performance Metrics ===")
                print(f"Reward Performance:")
                print(f"  - Average Reward: {avg_reward:8.2f}")
                print(f"  - Reward Std Dev: {eval_metrics['reward_std']:8.2f}")
                print(f"  - Max Reward:     {eval_metrics['max_reward']:8.2f}")
                print(f"  - Min Reward:     {eval_metrics['min_reward']:8.2f}")
                
                print(f"Policy Performance:")
                print(f"  - Policy Entropy: {eval_metrics['avg_policy_entropy']:8.4f}")
                print(f"  - Value Prediction: {eval_metrics['avg_value_prediction']:6.2f}")
                print(f"  - Episode Length: {eval_metrics['avg_episode_length']:8.1f}")
                
                print(f"Supply Chain Metrics:")
                print(f"  - Cost Efficiency: {eval_metrics['cost_efficiency']:7.3f}")
                print(f"  - Risk Mitigation: {eval_metrics['risk_mitigation_score']*100:6.1f}%")
                
                action_dist = eval_metrics['action_distribution']
                print(f"Action Distribution:")
                print(f"  - Standard:   {action_dist[0]*100:5.1f}%")
                print(f"  - Expedited:  {action_dist[1]*100:5.1f}%")
                print(f"  - Priority:   {action_dist[2]*100:5.1f}%")
                
                print(f"Training Losses:")
                print(f"  - Policy Loss: {policy_loss.item():8.4f}")
                print(f"  - Value Loss:  {value_loss.item():8.4f}")
                print("-" * 50)
                
            elif epoch % 5 == 0:
                avg_reward = np.mean(episode_rewards[-5:]) if episode_rewards else 0
                print(f"Epoch {epoch:3d}/{self.epochs} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Policy Loss: {policy_loss.item():6.4f} | "
                      f"Value Loss: {value_loss.item():6.4f}")
        
        # Final evaluation
        print("\n" + "="*70)
        print("FINAL PERFORMANCE EVALUATION")
        print("="*70)
        final_eval = self.evaluate_policy(num_episodes=20)
        
        print(f"REWARD PERFORMANCE:")
        print(f"  Average Reward:     {final_eval['avg_reward']:10.2f}")
        print(f"  Reward Std Dev:     {final_eval['reward_std']:10.2f}")
        print(f"  Reward Variance:    {final_eval['reward_variance']:10.2f}")
        print(f"  Max Reward:         {final_eval['max_reward']:10.2f}")
        print(f"  Min Reward:         {final_eval['min_reward']:10.2f}")
        
        print(f"\nPOLICY PERFORMANCE:")
        print(f"  Policy Entropy:     {final_eval['avg_policy_entropy']:10.4f}")
        print(f"  Value Prediction:   {final_eval['avg_value_prediction']:10.2f}")
        print(f"  Episode Length:     {final_eval['avg_episode_length']:10.1f}")
        
        print(f"\nSUPPLY CHAIN EFFICIENCY:")
        print(f"  Cost Efficiency:    {final_eval['cost_efficiency']:10.3f}")
        print(f"  Risk Mitigation:    {final_eval['risk_mitigation_score']*100:9.1f}%")
        
        action_dist = final_eval['action_distribution']
        print(f"\nACTION DISTRIBUTION:")
        print(f"  Standard Shipping:  {action_dist[0]*100:9.1f}%")
        print(f"  Expedited Shipping: {action_dist[1]*100:9.1f}%")
        print(f"  Priority Shipping:  {action_dist[2]*100:9.1f}%")
        
        # Training convergence metrics
        if len(episode_rewards) >= 20:
            early_rewards = np.mean(episode_rewards[:10])
            late_rewards = np.mean(episode_rewards[-10:])
            improvement = ((late_rewards - early_rewards) / abs(early_rewards)) * 100 if early_rewards != 0 else 0
            
            print(f"\nTRAINING CONVERGENCE:")
            print(f"  Early Avg Reward:   {early_rewards:10.2f}")
            print(f"  Late Avg Reward:    {late_rewards:10.2f}")
            print(f"  Improvement:        {improvement:9.1f}%")
        
        # Save model
        torch.save(self.policy.state_dict(), self.model_save_path)
        print(f"\nMODEL SAVED TO: {self.model_save_path}")
        print("="*70)
        
        return {
            'final_reward': episode_rewards[-1] if episode_rewards else 0,
            'avg_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards) if episode_rewards else 0,
            'final_policy_loss': policy_losses[-1] if policy_losses else 0,
            'final_value_loss': value_losses[-1] if value_losses else 0,
            'final_metrics': final_eval,
            'performance_history': performance_history,
            'training_rewards': episode_rewards,
            'training_policy_losses': policy_losses,
            'training_value_losses': value_losses
        }

if __name__ == '__main__':
    # Load config and train
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = PPOTrainer(config['models']['ppo'])
    trainer.train()