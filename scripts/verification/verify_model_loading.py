"""
Verification Script for Model Loading
Checks which models are loaded and whether they are genuine or mock/fallback
"""
import os
import sys
import torch

# Add paths
sys.path.insert(0, 'src/production')
sys.path.insert(0, 'src/core')

def check_model_files():
    """Check if model files exist."""
    model_files = {
        'Intent Transformer': 'models/smart_balanced_intent_transformer.pth',
        'Intent Encoders': 'models/intent_encoders.pkl',
        'Delay Risk GNN': 'models/gnn_classification.pth',
        'GNN Scalers': 'models/gnn_scalers.pkl',
        'GNN Encoders': 'models/gnn_label_encoders.pkl',
        'PPO Agent': 'models/ppo_agent_final.pth'
    }
    
    print("="*70)
    print("Model File Existence Check")
    print("="*70)
    
    all_exist = True
    for model_name, file_path in model_files.items():
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"{status} {model_name}: {file_path}")
        if not exists:
            all_exist = False
    
    print("="*70)
    return all_exist

def check_production_models():
    """Check production system models."""
    print("\n" + "="*70)
    print("Production System Model Check")
    print("="*70)
    
    try:
        from SynchroChain_Production_System import (
            SynchroChainProduction, IntentTransformer, 
            DelayRiskGNN, PPOAgent
        )
        
        # Check Intent Transformer
        print("\n1. Intent Transformer:")
        try:
            it = IntentTransformer()
            if it.model is not None:
                print("   ✅ Genuine model loaded")
                print(f"   ✅ Model type: {type(it.model).__name__}")
            else:
                print("   ⚠️  Model not loaded (using fallback)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Check Delay Risk GNN
        print("\n2. Delay Risk GNN:")
        try:
            gnn = DelayRiskGNN()
            if gnn.model is not None:
                print("   ✅ Genuine model loaded")
                print(f"   ✅ Model type: {type(gnn.model).__name__}")
            else:
                print("   ⚠️  Model not loaded (using fallback)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Check PPO Agent
        print("\n3. PPO Agent:")
        try:
            ppo = PPOAgent()
            if ppo.model is not None:
                print("   ✅ Genuine model loaded")
                print(f"   ✅ Model type: {type(ppo.model).__name__}")
                print(f"   ✅ Using fallback: {ppo.use_fallback}")
            else:
                print("   ⚠️  Model not loaded (using fallback)")
                print(f"   ⚠️  Using fallback: {ppo.use_fallback}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Check Full System
        print("\n4. Full Production System:")
        try:
            system = SynchroChainProduction()
            print("   ✅ System initialized successfully")
            
            # Check each component
            print("\n   Component Status:")
            if hasattr(system.intent_transformer, 'model'):
                status = "✅ Genuine" if system.intent_transformer.model is not None else "⚠️  Fallback"
                print(f"   - Intent Transformer: {status}")
            
            if hasattr(system.delay_risk_gnn, 'model'):
                status = "✅ Genuine" if system.delay_risk_gnn.model is not None else "⚠️  Fallback"
                print(f"   - Delay Risk GNN: {status}")
            
            if hasattr(system.rl_agent, 'model'):
                status = "✅ Genuine" if system.rl_agent.model is not None else "⚠️  Fallback"
                print(f"   - PPO Agent: {status}")
                if hasattr(system.rl_agent, 'use_fallback'):
                    print(f"   - PPO Fallback Flag: {system.rl_agent.use_fallback}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error importing production system: {e}")
        import traceback
        traceback.print_exc()

def check_model_integration():
    """Check model integration (dashboard) models."""
    print("\n" + "="*70)
    print("Model Integration (Dashboard) Check")
    print("="*70)
    
    try:
        from model_integration import ModelManager
        
        manager = ModelManager()
        
        print("\nModel Type Check:")
        
        # Check Intent Transformer
        from model_integration import MockIntentTransformer
        it_type = type(manager.models.get('intent_transformer')).__name__
        is_mock = isinstance(manager.models.get('intent_transformer'), MockIntentTransformer)
        status = "⚠️  Mock" if is_mock else "✅ Genuine"
        print(f"   - Intent Transformer: {status} ({it_type})")
        
        # Check GNN
        from model_integration import MockDelayRiskGNN
        gnn_type = type(manager.models.get('gnn')).__name__
        is_mock = isinstance(manager.models.get('gnn'), MockDelayRiskGNN)
        status = "⚠️  Mock" if is_mock else "✅ Genuine"
        print(f"   - Delay Risk GNN: {status} ({gnn_type})")
        
        # Check PPO
        from model_integration import MockPPOAgent
        ppo_type = type(manager.models.get('ppo')).__name__
        is_mock = isinstance(manager.models.get('ppo'), MockPPOAgent)
        status = "⚠️  Mock" if is_mock else "✅ Genuine"
        print(f"   - PPO Agent: {status} ({ppo_type})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all checks."""
    print("\n" + "="*70)
    print("SynchroChain Model Loading Verification")
    print("="*70 + "\n")
    
    # Check 1: Model files exist
    all_files_exist = check_model_files()
    
    # Check 2: Production models
    check_production_models()
    
    # Check 3: Model integration
    check_model_integration()
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if all_files_exist:
        print("✅ All model files exist")
    else:
        print("⚠️  Some model files are missing (this may cause fallback to mock models)")
    
    print("\n" + "="*70)
    print("Verification Complete!")
    print("="*70)

if __name__ == "__main__":
    main()





