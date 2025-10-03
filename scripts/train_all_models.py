# scripts/train_all_models.py
"""
Master training script for all SynchroChain models
Trains: Data Preprocessing → Intent Transformer → GNN → PPO RL Agent
"""
import os
import sys
import yaml
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def train_data_preprocessing():
    """Step 1: Train data preprocessing pipeline."""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    try:
        from src.data.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        preprocessor.run()
        
        print("✅ Data preprocessing completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def train_intent_transformer():
    """Step 2: Train Intent Transformer model."""
    print("\n" + "="*60)
    print("STEP 2: INTENT TRANSFORMER TRAINING")
    print("="*60)
    
    try:
        from src.models.intent_transformer.trainer import IntentTrainer
        
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        trainer = IntentTrainer(config['models']['intent_transformer'])
        metrics = trainer.train()
        
        print("✅ Intent Transformer training completed!")
        print(f"Final Validation Accuracy: {metrics.get('best_val_accuracy', 'N/A'):.4f}")
        return True
    except Exception as e:
        print(f"❌ Intent Transformer training failed: {e}")
        return False

def train_gnn():
    """Step 3: Train GNN model."""
    print("\n" + "="*60)
    print("STEP 3: GNN TRAINING")
    print("="*60)
    
    try:
        from src.models.gnn.trainer import GNNTrainer
        
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        trainer = GNNTrainer(config['models']['gnn'])
        metrics = trainer.train()
        
        print("✅ GNN training completed!")
        print(f"Final Validation Loss: {metrics.get('best_val_loss', 'N/A'):.6f}")
        return True
    except Exception as e:
        print(f"❌ GNN training failed: {e}")
        return False

def train_ppo_agent():
    """Step 4: Train PPO RL Agent."""
    print("\n" + "="*60)
    print("STEP 4: PPO RL AGENT TRAINING")
    print("="*60)
    
    try:
        from src.models.agents.ppo_trainer import PPOTrainer
        
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        trainer = PPOTrainer(config['models']['ppo'])
        metrics = trainer.train()
        
        print("✅ PPO RL Agent training completed!")
        print(f"Final Reward: {metrics.get('final_reward', 'N/A'):.4f}")
        return True
    except Exception as e:
        print(f"❌ PPO RL Agent training failed: {e}")
        return False

def main():
    """Main training pipeline."""
    print("SynchroChain Model Training Pipeline")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    start_time = time.time()
    results = {}
    
    # Step 1: Data Preprocessing
    results['preprocessing'] = train_data_preprocessing()
    if not results['preprocessing']:
        print("\n❌ Pipeline stopped due to preprocessing failure.")
        return
    
    # Step 2: Intent Transformer
    results['intent_transformer'] = train_intent_transformer()
    if not results['intent_transformer']:
        print("\n❌ Pipeline stopped due to intent transformer failure.")
        return
    
    # Step 3: GNN
    results['gnn'] = train_gnn()
    if not results['gnn']:
        print("\n❌ Pipeline stopped due to GNN failure.")
        return
    
    # Step 4: PPO RL Agent
    results['ppo_agent'] = train_ppo_agent()
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE SUMMARY")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Data Preprocessing: {'✅' if results['preprocessing'] else '❌'}")
    print(f"Intent Transformer: {'✅' if results['intent_transformer'] else '❌'}")
    print(f"GNN Model: {'✅' if results['gnn'] else '❌'}")
    print(f"PPO RL Agent: {'✅' if results['ppo_agent'] else '❌'}")
    
    if all(results.values()):
        print("\n🎉 All models trained successfully!")
    else:
        print("\n⚠️ Some models failed to train. Check logs for details.")

if __name__ == '__main__':
    main()