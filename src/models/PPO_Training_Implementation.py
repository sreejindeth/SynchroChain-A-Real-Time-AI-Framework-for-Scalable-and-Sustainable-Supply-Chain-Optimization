"""
PPO (Proximal Policy Optimization) Training Implementation
RL environment for supply chain decision making - no traditional train/test splits needed
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pickle
import random
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class SupplyChainEnvironment:
    """RL Environment for supply chain decision making."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.state_dim = 8  # State space dimension
        self.action_dim = 4  # Action space dimension (4 discrete actions)
        
        # Environment parameters
        self.max_steps = 100
        self.current_step = 0
        self.current_state = None
        
        # Action mapping
        self.action_map = {
            0: 'pre_allocate',
            1: 'restock', 
            2: 'expedite_shipping',
            3: 'normal_operation'
        }
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        
        # Initialize random state
        self.current_state = np.array([
            random.uniform(0, 1),  # intent_score
            random.uniform(0, 1),  # urgency_level
            random.uniform(0, 1),  # delay_risk
            random.uniform(0, 1),  # inventory_level
            random.uniform(0, 1),  # carbon_cost
            random.uniform(0, 1),  # order_value
            random.uniform(0, 1),  # customer_priority
            random.uniform(0, 1),  # supplier_reliability
        ])
        
        return self.current_state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        self.current_step += 1
        
        # Get action name
        action_name = self.action_map[action]
        
        # Calculate reward based on action and state
        reward = self._calculate_reward(action_name)
        
        # Update state based on action
        self._update_state(action_name)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Info dictionary
        info = {
            'action_taken': action_name,
            'step': self.current_step,
            'state': self.current_state.copy()
        }
        
        return self.current_state.copy(), reward, done, info
    
    def _calculate_reward(self, action: str) -> float:
        """Calculate reward for the given action."""
        intent_score = self.current_state[0]
        urgency = self.current_state[1]
        delay_risk = self.current_state[2]
        inventory_level = self.current_state[3]
        carbon_cost = self.current_state[4]
        order_value = self.current_state[5]
        
        reward = 0.0
        
        if action == 'pre_allocate':
            # Reward for pre-allocation based on intent and urgency
            if intent_score > 0.7 and urgency > 0.6:
                reward += 10.0  # Good decision
            elif intent_score < 0.3:
                reward -= 5.0   # Bad decision (wasteful)
            else:
                reward += 2.0   # Neutral decision
        
        elif action == 'restock':
            # Reward for restocking based on inventory level
            if inventory_level < 0.3:
                reward += 8.0   # Good decision
            elif inventory_level > 0.8:
                reward -= 3.0   # Bad decision (unnecessary)
            else:
                reward += 1.0   # Neutral decision
        
        elif action == 'expedite_shipping':
            # Reward for expedited shipping based on urgency and delay risk
            if urgency > 0.8 and delay_risk < 0.3:
                reward += 12.0  # Excellent decision
            elif urgency < 0.4:
                reward -= 4.0   # Bad decision (unnecessary cost)
            else:
                reward += 3.0   # Neutral decision
        
        elif action == 'normal_operation':
            # Reward for normal operation (baseline)
            if intent_score < 0.4 and urgency < 0.5:
                reward += 5.0   # Good decision (conservative)
            else:
                reward += 1.0   # Neutral decision
        
        # Add cost penalties
        if action == 'pre_allocate':
            reward -= 2.0  # Storage cost
        if action == 'restock':
            reward -= 3.0  # Restocking cost
        if action == 'expedite_shipping':
            reward -= 4.0  # Expedited shipping cost
        
        # Add carbon cost penalty
        if action == 'expedite_shipping':
            reward -= carbon_cost * 2.0
        
        return reward
    
    def _update_state(self, action: str):
        """Update environment state based on action."""
        # Simulate state transitions
        if action == 'pre_allocate':
            # Pre-allocation reduces inventory but increases readiness
            self.current_state[3] = max(0, self.current_state[3] - 0.1)  # Reduce inventory
            self.current_state[1] = min(1, self.current_state[1] + 0.05)  # Increase urgency
        
        elif action == 'restock':
            # Restocking increases inventory
            self.current_state[3] = min(1, self.current_state[3] + 0.2)  # Increase inventory
            self.current_state[2] = max(0, self.current_state[2] - 0.1)  # Reduce delay risk
        
        elif action == 'expedite_shipping':
            # Expedited shipping reduces delay risk but increases carbon cost
            self.current_state[2] = max(0, self.current_state[2] - 0.2)  # Reduce delay risk
            self.current_state[4] = min(1, self.current_state[4] + 0.1)  # Increase carbon cost
        
        # Add some randomness to simulate real-world dynamics
        for i in range(len(self.current_state)):
            self.current_state[i] += random.uniform(-0.05, 0.05)
            self.current_state[i] = np.clip(self.current_state[i], 0, 1)

class PPOAgent(nn.Module):
    """PPO Agent with Actor-Critic architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPOAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        """Forward pass through the network."""
        shared_features = self.shared_net(state)
        
        # Actor output (action probabilities)
        action_probs = self.actor(shared_features)
        
        # Critic output (state value)
        state_value = self.critic(shared_features)
        
        return action_probs, state_value
    
    def get_action(self, state, deterministic=False):
        """Get action from current policy."""
        with torch.no_grad():
            action_probs, _ = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action = torch.multinomial(action_probs, 1)
            
            return action.item(), action_probs
    
    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO loss calculation."""
        action_probs, state_values = self.forward(states)
        
        # Get action probabilities for the taken actions
        action_probs_selected = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate log probabilities
        log_probs = torch.log(action_probs_selected + 1e-8)
        
        # Calculate entropy for exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        
        return log_probs, state_values.squeeze(), entropy

class PPOTrainer:
    """PPO Trainer for supply chain RL agent."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize environment and agent
        self.env = SupplyChainEnvironment(config)
        self.agent = PPOAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            hidden_dim=config.get('hidden_dim', 128)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=config['learning_rate']
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
    def collect_rollouts(self, num_rollouts: int) -> Dict:
        """Collect rollouts for PPO training."""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        for _ in range(num_rollouts):
            state = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_log_probs = []
            episode_dones = []
            
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get action from current policy
                action, action_probs = self.agent.get_action(state_tensor)
                log_prob, value = self.agent.evaluate_actions(state_tensor, torch.tensor([action]))
                
                # Take action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_values.append(value.item())
                episode_log_probs.append(log_prob.item())
                episode_dones.append(done)
                
                state = next_state
            
            # Store episode data
            states.extend(episode_states)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
            values.extend(episode_values)
            log_probs.extend(episode_log_probs)
            dones.extend(episode_dones)
            
            # Track episode metrics
            episode_reward = sum(episode_rewards)
            episode_length = len(episode_rewards)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
        
        return {
            'states': torch.FloatTensor(states).to(self.device),
            'actions': torch.LongTensor(actions).to(self.device),
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'values': torch.FloatTensor(values).to(self.device),
            'log_probs': torch.FloatTensor(log_probs).to(self.device),
            'dones': torch.BoolTensor(dones).to(self.device)
        }
    
    def compute_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute GAE advantages."""
        advantages = []
        returns = []
        
        # Compute returns and advantages
        running_return = 0
        running_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
                running_advantage = 0
            
            running_return = rewards[t] + gamma * running_return
            running_advantage = rewards[t] + gamma * values[t + 1] if t < len(values) - 1 else rewards[t]
            running_advantage = running_advantage - values[t]
            running_advantage = running_advantage + gamma * lam * running_advantage if not dones[t] else running_advantage
            
            returns.insert(0, running_return)
            advantages.insert(0, running_advantage)
        
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)
    
    def update_policy(self, rollouts: Dict):
        """Update policy using PPO."""
        states = rollouts['states']
        actions = rollouts['actions']
        rewards = rollouts['rewards']
        old_values = rollouts['values']
        old_log_probs = rollouts['log_probs']
        dones = rollouts['dones']
        
        # Compute advantages and returns
        returns, advantages = self.compute_advantages(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training
        for _ in range(self.config['ppo_epochs']):
            # Get current policy
            log_probs, values, entropy = self.agent.evaluate_actions(states, actions)
            
            # Compute ratios
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config['eps_clip'], 1 + self.config['eps_clip']) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = nn.MSELoss()(values, returns)
            
            # Total loss
            total_loss = actor_loss + self.config['value_coef'] * critic_loss - self.config['entropy_coef'] * entropy
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()
            
            self.training_losses.append(total_loss.item())
    
    def train(self, num_episodes: int):
        """Train the PPO agent."""
        print("🚀 Training PPO agent...")
        
        for episode in range(num_episodes):
            # Collect rollouts
            rollouts = self.collect_rollouts(self.config['rollout_steps'])
            
            # Update policy
            self.update_policy(rollouts)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
                print(f"   Episode {episode:3d}: Avg Reward: {avg_reward:6.2f}, Avg Length: {avg_length:5.1f}")
        
        print("✅ PPO training completed!")
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate the trained agent."""
        print("📊 Evaluating PPO agent...")
        
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, _ = self.agent.get_action(state_tensor, deterministic=True)
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        avg_length = np.mean(eval_lengths)
        
        print(f"   Evaluation Results:")
        print(f"   Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   Average Length: {avg_length:.1f}")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_length': avg_length,
            'rewards': eval_rewards,
            'lengths': eval_lengths
        }
    
    def plot_training_progress(self):
        """Plot training progress."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Moving average rewards
        window = 100
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(moving_avg)
            ax2.set_title(f'Moving Average Rewards (window={window})')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Reward')
            ax2.grid(True)
        
        # Episode lengths
        ax3.plot(self.episode_lengths)
        ax3.set_title('Episode Lengths')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Length')
        ax3.grid(True)
        
        # Training losses
        ax4.plot(self.training_losses)
        ax4.set_title('Training Losses')
        ax4.set_xlabel('Update')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/ppo_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Save the trained model."""
        os.makedirs('models', exist_ok=True)
        torch.save(self.agent.state_dict(), 'models/ppo_agent_trained.pth')
        print("💾 PPO model saved successfully!")

def main():
    """Main training function."""
    print("=" * 80)
    print("🤖 PPO TRAINING - REINFORCEMENT LEARNING (NO TRAIN/TEST SPLITS)")
    print("=" * 80)
    
    # Configuration
    config = {
        'learning_rate': 3e-4,
        'epochs': 100,
        'batch_size': 64,
        'rollout_steps': 2048,
        'ppo_epochs': 10,
        'gamma': 0.99,
        'eps_clip': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'hidden_dim': 128
    }
    
    try:
        # Initialize trainer
        trainer = PPOTrainer(config)
        
        # Train agent
        trainer.train(config['epochs'])
        
        # Evaluate agent
        eval_results = trainer.evaluate()
        
        # Plot training progress
        trainer.plot_training_progress()
        
        # Save model
        trainer.save_model()
        
        print("\n🎉 PPO training completed successfully!")
        print(f"📊 Final Evaluation Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()















