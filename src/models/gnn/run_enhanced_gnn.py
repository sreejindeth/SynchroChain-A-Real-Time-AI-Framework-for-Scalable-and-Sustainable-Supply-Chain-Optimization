# src/models/gnn/run_enhanced_gnn.py
"""
Main script to run the complete Enhanced GNN pipeline for SynchroChain
"""
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

def main():
    """Main function to run the enhanced GNN pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced GNN Pipeline for SynchroChain')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'], default='both',
                       help='Mode to run: train, evaluate, or both')
    parser.add_argument('--model-type', choices=['enhanced', 'lightweight', 'both'], default='both',
                       help='Model type to use')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced epochs for testing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("[GNN] ENHANCED SUPPLY CHAIN GNN PIPELINE")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Model Type: {args.model_type}")
    print(f"Quick Mode: {args.quick}")
    print(f"Start Time: {datetime.now()}")
    print("="*60)
    
    try:
        if args.mode in ['train', 'both']:
            print("\n[STEP 1] TRAINING ENHANCED GNN MODELS")
            print("-" * 40)
            
            from src.models.gnn.enhanced_trainer import EnhancedGNNTrainer
            
            trainer = EnhancedGNNTrainer()
            
            # Modify config for quick mode
            if args.quick:
                trainer.num_epochs = 10
                trainer.patience = 5
                print("[QUICK] Using reduced epochs (10) and patience (5)")
            
            model_types = ['enhanced', 'lightweight'] if args.model_type == 'both' else [args.model_type]
            
            training_results = {}
            for model_type in model_types:
                print(f"\n[TRAIN] Training {model_type} model...")
                try:
                    results = trainer.train_model(model_type=model_type)
                    training_results[model_type] = results
                    trainer.create_training_plots(results, model_type)
                except Exception as e:
                    print(f"[ERROR] Training {model_type} model failed: {e}")
                    continue
            
            print(f"\n[COMPLETE] Training completed for {len(training_results)} models")
        
        if args.mode in ['evaluate', 'both']:
            print("\n[STEP 2] COMPREHENSIVE MODEL EVALUATION")
            print("-" * 40)
            
            from src.models.gnn.comprehensive_evaluator import ComprehensiveGNNEvaluator
            
            evaluator = ComprehensiveGNNEvaluator()
            
            model_types = ['enhanced', 'lightweight'] if args.model_type == 'both' else [args.model_type]
            
            try:
                evaluation_results = evaluator.run_comprehensive_evaluation(model_types=model_types)
                print(f"\n[COMPLETE] Evaluation completed for {len(evaluation_results)} models")
            except Exception as e:
                print(f"[ERROR] Evaluation failed: {e}")
        
        print("\n" + "="*60)
        print("[SUCCESS] Enhanced GNN pipeline completed successfully!")
        print(f"End Time: {datetime.now()}")
        print("="*60)
        
        # Print summary
        print("\n[SUMMARY] RESULTS SUMMARY")
        print("-" * 30)
        
        if args.mode in ['train', 'both']:
            print("Training Results:")
            if 'training_results' in locals():
                for model_type, results in training_results.items():
                    print(f"  {model_type.title()} Model:")
                    print(f"    - Best Val Loss: {results['best_val_loss']:.6f}")
                    print(f"    - Training Time: {results['training_time']}")
                    if results['test_results']:
                        print(f"    - Test Loss: {results['test_results']['total']:.6f}")
        
        if args.mode in ['evaluate', 'both']:
            print("\nEvaluation Results:")
            if 'evaluation_results' in locals():
                for model_type, model_results in evaluation_results.items():
                    print(f"  {model_type.title()} Model:")
                    if 'test' in model_results:
                        metrics = model_results['test']['metrics']
                        print(f"    - Delay Risk MAE: {metrics.get('delay_risk_mae', 0):.4f}")
                        print(f"    - Shortfall Accuracy: {metrics.get('inventory_shortfall_accuracy', 0):.4f}")
                        print(f"    - Carbon Cost MAE: {metrics.get('carbon_cost_mae', 0):.4f}")
        
        print("\n[FILES] Generated Files:")
        results_dir = os.path.join(project_root, 'results')
        models_dir = os.path.join(project_root, 'models', 'saved')
        
        print(f"  Models: {models_dir}")
        print(f"  Results: {results_dir}")
        print(f"  - Training plots: gnn_training_plots_*.png")
        print(f"  - Evaluation plots: gnn_evaluation_*.png")
        print(f"  - Results JSON: gnn_training_results_*.json")
        print(f"  - Comprehensive evaluation: comprehensive_gnn_evaluation.json")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()






