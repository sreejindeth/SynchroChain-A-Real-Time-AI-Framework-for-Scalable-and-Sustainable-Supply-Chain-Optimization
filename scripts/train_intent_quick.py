#!/usr/bin/env python3
"""
Quick Start Script for Training Intent Transformer
Simple command-line interface for training and evaluation.
"""
import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def train_intent_model():
    """Train the Intent Transformer model."""
    print("🚀 Starting Intent Transformer training...")
    print("-" * 70)
    
    from src.models.intent_transformer.train_intent_comprehensive import main as train_main
    
    results = train_main()
    
    if results:
        print("\n✅ Training completed successfully!")
        print(f"   Final Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"   F1-Score: {results['metrics']['f1_score']:.4f}")
        return True
    else:
        print("\n❌ Training failed!")
        return False

def evaluate_intent_model():
    """Evaluate the trained Intent Transformer model."""
    print("🎯 Starting Intent Transformer evaluation...")
    print("-" * 70)
    
    from src.models.intent_transformer.evaluate_intent_model import main as eval_main
    
    results = eval_main()
    
    if results:
        print("\n✅ Evaluation completed successfully!")
        print(f"   Test Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"   Test F1-Score: {results['metrics']['f1_score']:.4f}")
        return True
    else:
        print("\n❌ Evaluation failed!")
        return False

def check_prerequisites():
    """Check if required data files exist."""
    print("🔍 Checking prerequisites...")
    
    data_dir = os.path.join(project_root, 'data', 'processed', 'temporal_splits')
    train_path = os.path.join(data_dir, 'train', 'processed_access_logs.csv')
    
    if not os.path.exists(train_path):
        print("❌ Training data not found!")
        print(f"   Expected: {train_path}")
        print("\n💡 Please run preprocessing first:")
        print("   python src/data/preprocessing.py")
        return False
    
    print("✓ Prerequisites met!")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Intent Transformer Training & Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python scripts/train_intent_quick.py --train
  
  # Evaluate the model
  python scripts/train_intent_quick.py --evaluate
  
  # Train and then evaluate
  python scripts/train_intent_quick.py --train --evaluate
  
  # Check prerequisites only
  python scripts/train_intent_quick.py --check
        """
    )
    
    parser.add_argument('--train', action='store_true', 
                       help='Train the Intent Transformer model')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Evaluate the trained model')
    parser.add_argument('--check', action='store_true', 
                       help='Check prerequisites only')
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if not (args.train or args.evaluate or args.check):
        parser.print_help()
        return
    
    print("="*70)
    print("🤖 SYNCHROCHAIN - INTENT TRANSFORMER")
    print("="*70)
    
    # Check prerequisites
    if args.check or args.train:
        if not check_prerequisites():
            print("\n❌ Prerequisites not met. Exiting.")
            return
    
    # Training
    if args.train:
        print("\n" + "="*70)
        success = train_intent_model()
        if not success:
            print("\n❌ Stopping due to training failure.")
            return
    
    # Evaluation
    if args.evaluate:
        print("\n" + "="*70)
        model_path = os.path.join(project_root, 'models', 'intent_transformer_finetuned_multi.pth')
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("   Please train the model first using --train")
            return
        
        evaluate_intent_model()
    
    print("\n" + "="*70)
    print("🎉 ALL TASKS COMPLETED!")
    print("="*70)

if __name__ == '__main__':
    main()


