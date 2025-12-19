"""
Run Training Pipeline - Steel Defect Detection System
======================================================

Usage:
    python run_pipeline.py              # Full pipeline (all images)
    python run_pipeline.py --test-run   # Quick test (10 train, 5 test images)
    python run_pipeline.py --model baseline --epochs 10   # Train baseline
    python run_pipeline.py --model transfer --epochs 20   # Train transfer learning
    python run_pipeline.py --memory-efficient  # For full dataset on limited RAM
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.steel_defect_detection_system.pipelines.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Steel Defect Detection Training Pipeline')
    parser.add_argument('--max-train', type=int, default=None,
                        help='Max training images')
    parser.add_argument('--max-test', type=int, default=None,
                        help='Max test images')
    parser.add_argument('--test-run', action='store_true',
                        help='Quick test with 10 train, 5 test images')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'transfer'],
                        help='Model type: baseline or transfer')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training (data pipeline only)')
    parser.add_argument('--memory-efficient', action='store_true',
                        help='Use disk-based approach for large datasets (saves RAM)')
    
    args = parser.parse_args()
    
    if args.test_run:
        args.max_train = 10
        args.max_test = 5
        print("[TEST] Running quick test (10 train, 5 test images)")
    
    if args.memory_efficient:
        print("[MEMORY] Using disk-based approach for large datasets")
    
    pipeline = TrainingPipeline()
    result = pipeline.run_pipeline(
        max_train_images=args.max_train,
        max_test_images=args.max_test,
        model_type=args.model,
        epochs=args.epochs,
        skip_training=args.skip_training,
        memory_efficient=args.memory_efficient
    )
    
    print("\n" + "="*60)
    print("[OK] PIPELINE COMPLETE")
    print("="*60)
    
    # Print dataset statistics
    print(f"Train patches: {result['transformation_result']['train_patches']}")
    print(f"Test patches: {result['transformation_result']['test_patches']}")
    
    if result['training_result']:
        history = result['training_result']['history']
        print(f"Final val recall: {history.get('val_recall', [0])[-1]:.4f}")
