"""
Evaluate PPO Agent vs Heuristic Baseline Policy

This script compares the trained PPO agent against a rule-based heuristic baseline
to compute genuine improvement percentages for academic reporting.
"""

import numpy as np
import torch
import json
import os
import sys

# Add training scripts path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from train_ppo_final import SupplyChainEnv, ImprovedPPOAgent

# Ensure output is flushed immediately
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None


class HeuristicBaselinePolicy:
    """
    Rule-based heuristic policy matching the production fallback logic.
    This policy makes decisions based on fixed thresholds and rules.
    """
    
    def __init__(self):
        self.action_map = {
            0: 'pre_allocate',
            1: 'restock',
            2: 'expedite_shipping',
            3: 'normal_operation'
        }
    
    def get_action(self, state):
        """
        Implement heuristic policy based on state thresholds.
        Matches the logic from _rule_based_decide() in production code.
        """
        intent_score = state[0]
        urgency = state[1]
        delay_risk = state[2]
        inventory_level = state[3]
        carbon_cost = state[4]
        
        # Decision logic: prioritize actions based on conditions
        
        # 1. Check if pre-allocation is needed (highest priority)
        if intent_score > 0.7 and urgency > 0.6:
            return 0  # pre_allocate
        
        # 2. Check if restocking is needed
        if inventory_level < 0.3 and intent_score > 0.5:
            return 1  # restock
        
        # 3. Check if expedited shipping is needed
        if urgency > 0.7 or delay_risk > 0.6:
            return 2  # expedite_shipping
        
        # 4. Default: normal operation
        return 3  # normal_operation


def evaluate_policy(env, policy, num_episodes=100, policy_name="Policy"):
    """
    Evaluate a policy (PPO agent or heuristic) on the environment.
    
    Args:
        env: SupplyChainEnv instance
        policy: Policy object with get_action() method or dict
        num_episodes: Number of episodes to run
        policy_name: Name for progress display
    
    Returns:
        dict with metrics
    """
    total_rewards = []
    total_violations = 0
    episode_violations = []
    
    # Progress indicator
    progress_interval = max(1, num_episodes // 10)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            # Get action from policy
            if isinstance(policy, dict):
                # PPO agent loaded from file - use deterministic (argmax) for evaluation
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action_probs, _ = policy['model'](state_tensor)
                    action = torch.argmax(action_probs, dim=1).item()  # Deterministic
            else:
                # Heuristic baseline
                action = policy.get_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if info.get('violated', False):
                total_violations += 1
            
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        episode_violations.append(env.episode_violations)
        
        # Progress update
        if (episode + 1) % progress_interval == 0 or (episode + 1) == num_episodes:
            print(f"   Episode {episode + 1}/{num_episodes} (Avg reward so far: {np.mean(total_rewards):.2f})", flush=True)
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    violation_rate = (total_violations / (num_episodes * env.max_steps)) * 100
    
    return {
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'violation_rate_percent': float(violation_rate),
        'total_violations': int(total_violations),
        'total_episodes': num_episodes,
        'rewards': total_rewards
    }


def load_ppo_model():
    """Load the trained PPO model."""
    model_path = 'models/ppo_agent_final.pth'
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}")
        print("Looking for alternative model files...")
        # Try alternative paths
        alt_paths = [
            'models/ppo_agent_trained.pth',
            'models/ppo_agent.pth'
        ]
        found = False
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"✓ Found model at {alt_path}")
                found = True
                break
        if not found:
            raise FileNotFoundError(f"PPO model not found. Please train the model first.")
    
    print(f"📂 Loading model from: {model_path}")
    # Create model architecture
    agent = ImprovedPPOAgent(state_dim=8, action_dim=4, hidden_dim=256)
    state_dict = torch.load(model_path, map_location='cpu')
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"✓ Model loaded successfully")
    
    return {'model': agent, 'path': model_path}


def main():
    """Main evaluation function."""
    print("="*70, flush=True)
    print("PPO Agent vs Heuristic Baseline Comparison", flush=True)
    print("="*70, flush=True)
    sys.stdout.flush()
    
    try:
        # Create environment (same as training)
        print("\n🌍 Creating environment...")
        env = SupplyChainEnv(violation_penalty=-300.0)
        print("✓ Environment created")
        
        # Load PPO agent
        print("\n📥 Loading trained PPO agent...")
        try:
            ppo_policy = load_ppo_model()
            print(f"✓ PPO model loaded from {ppo_policy['path']}")
        except Exception as e:
            print(f"❌ Error loading PPO model: {e}")
            import traceback
            traceback.print_exc()
            return
    except Exception as e:
        print(f"❌ Error in setup: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create heuristic baseline
    print("\n📋 Creating heuristic baseline policy...")
    heuristic_policy = HeuristicBaselinePolicy()
    print("✓ Heuristic baseline created")
    
    # Evaluate both policies
    num_episodes = 50  # Use reasonable number for fair comparison (can increase later)
    
    print(f"\n🔬 Evaluating PPO Agent ({num_episodes} episodes)...", flush=True)
    ppo_results = evaluate_policy(env, ppo_policy, num_episodes, "PPO Agent")
    
    print(f"\n🔬 Evaluating Heuristic Baseline ({num_episodes} episodes)...", flush=True)
    heuristic_results = evaluate_policy(env, heuristic_policy, num_episodes, "Heuristic Baseline")
    
    # Compute improvement
    ppo_reward = ppo_results['avg_reward']
    heuristic_reward = heuristic_results['avg_reward']
    
    if abs(heuristic_reward) < 1e-6:
        improvement_pct = 0.0
        print("⚠️ Baseline reward is zero, cannot compute percentage improvement")
    else:
        improvement_pct = ((ppo_reward - heuristic_reward) / abs(heuristic_reward)) * 100
    
    # Print results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n📊 PPO Agent Performance:")
    print(f"   Average Reward: {ppo_reward:.2f} ± {ppo_results['std_reward']:.2f}")
    print(f"   Violation Rate: {ppo_results['violation_rate_percent']:.2f}%")
    print(f"   Total Violations: {ppo_results['total_violations']}")
    
    print(f"\n📊 Heuristic Baseline Performance:")
    print(f"   Average Reward: {heuristic_reward:.2f} ± {heuristic_results['std_reward']:.2f}")
    print(f"   Violation Rate: {heuristic_results['violation_rate_percent']:.2f}%")
    print(f"   Total Violations: {heuristic_results['total_violations']}")
    
    print(f"\n📈 Improvement:", flush=True)
    print(f"   Absolute Improvement: {ppo_reward - heuristic_reward:.2f}", flush=True)
    print(f"   Relative Improvement: {improvement_pct:.2f}%", flush=True)
    multiplier_str = f"   Multiplier: {ppo_reward / abs(heuristic_reward):.2f}x" if abs(heuristic_reward) > 1e-6 else "   Multiplier: N/A"
    print(multiplier_str, flush=True)
    
    # Save results
    results = {
        'ppo_agent': {
            'avg_reward': ppo_reward,
            'std_reward': float(ppo_results['std_reward']),
            'violation_rate_percent': float(ppo_results['violation_rate_percent']),
            'total_violations': int(ppo_results['total_violations'])
        },
        'heuristic_baseline': {
            'avg_reward': heuristic_reward,
            'std_reward': float(heuristic_results['std_reward']),
            'violation_rate_percent': float(heuristic_results['violation_rate_percent']),
            'total_violations': int(heuristic_results['total_violations'])
        },
        'comparison': {
            'improvement_percentage': float(improvement_pct),
            'absolute_improvement': float(ppo_reward - heuristic_reward),
            'reward_ratio': float(ppo_reward / abs(heuristic_reward)) if abs(heuristic_reward) > 1e-6 else None,
            'evaluation_episodes': num_episodes,
            'violation_rate_improvement': float(heuristic_results['violation_rate_percent'] - ppo_results['violation_rate_percent'])
        },
        'evaluation_info': {
            'methodology': 'Same environment, same number of episodes',
            'heuristic_policy': 'Rule-based policy matching production fallback logic',
            'ppo_model': ppo_policy['path'],
            'genuine_baseline_comparison': True
        }
    }
    
    # Create results directory if needed
    os.makedirs('results/ppo_agent', exist_ok=True)
    
    # Save comparison results
    comparison_path = 'results/ppo_agent/baseline_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {comparison_path}")
    
    # Update main metrics.json with baseline comparison
    metrics_path = 'results/ppo_agent/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Add baseline comparison to metrics
        metrics['baseline_comparison'] = results['comparison']
        metrics['heuristic_baseline_metrics'] = results['heuristic_baseline']
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Updated {metrics_path} with baseline comparison")
    
    print("\n" + "="*70, flush=True)
    print("✅ Evaluation Complete!", flush=True)
    print("="*70, flush=True)
    sys.stdout.flush()
    
    # Also write summary to file for visibility
    summary_path = 'results/ppo_agent/baseline_comparison_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("PPO Agent vs Heuristic Baseline Comparison\n")
        f.write("="*70 + "\n\n")
        f.write(f"PPO Agent Average Reward: {ppo_reward:.2f} ± {ppo_results['std_reward']:.2f}\n")
        f.write(f"PPO Agent Violation Rate: {ppo_results['violation_rate_percent']:.2f}%\n\n")
        f.write(f"Heuristic Baseline Average Reward: {heuristic_reward:.2f} ± {heuristic_results['std_reward']:.2f}\n")
        f.write(f"Heuristic Baseline Violation Rate: {heuristic_results['violation_rate_percent']:.2f}%\n\n")
        f.write(f"Improvement: {improvement_pct:.2f}%\n")
        f.write(f"Absolute Improvement: {ppo_reward - heuristic_reward:.2f}\n")
        if abs(heuristic_reward) > 1e-6:
            f.write(f"Multiplier: {ppo_reward / abs(heuristic_reward):.2f}x\n")
    
    print(f"\n📄 Summary written to: {summary_path}", flush=True)
    
    return results


if __name__ == '__main__':
    # Debug: Write to file to confirm script runs
    try:
        with open('eval_debug.txt', 'w') as f:
            f.write('Script started\n')
    except:
        pass
    
    try:
        main()
        # Debug: Confirm completion
        try:
            with open('eval_debug.txt', 'a') as f:
                f.write('Script completed\n')
        except:
            pass
    except Exception as e:
        # Debug: Write error
        try:
            with open('eval_debug.txt', 'a') as f:
                f.write(f'Error: {e}\n')
                import traceback
                traceback.print_exc(file=f)
        except:
            pass
        raise

