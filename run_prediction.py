"""
Run Prediction - Steel Defect Detection System
===============================================

Usage:
    python run_prediction.py --image path/to/image.jpg
    python run_prediction.py --directory path/to/images/
    python run_prediction.py --image image.jpg --threshold 0.3
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.steel_defect_detection_system.pipelines.prediction_pipeline import (
    PredictionPipeline, PredictionConfig
)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Steel Defect Detection Prediction')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Classification threshold (default: 0.3)')
    parser.add_argument('--model', type=str, default='transfer_model_final.keras',
                        help='Model filename')
    
    args = parser.parse_args()
    
    if not args.image and not args.directory:
        print("Error: Please provide --image or --directory")
        parser.print_help()
        return
    
    # Configure
    config = PredictionConfig()
    config.optimal_threshold = args.threshold
    config.model_name = args.model
    
    # Initialize pipeline
    print("Loading model...")
    pipeline = PredictionPipeline(config)
    
    # Run prediction
    if args.image:
        print(f"\nPredicting: {args.image}")
        result = pipeline.predict_single_image(args.image)
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Image: {result['image_name']}")
        print(f"Decision: {result['decision']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Is Defective: {result['is_defective']}")
        print(f"Defective Patches: {result['n_defective_patches']}/{result['total_patches']}")
        print(f"Max Patch Prob: {result['max_patch_prob']:.3f}")
        print(f"Mean Patch Prob: {result['mean_patch_prob']:.3f}")
        print("="*50)
        
    elif args.directory:
        print(f"\nPredicting directory: {args.directory}")
        results = pipeline.predict_directory(args.directory)
        summary = pipeline.get_summary(results)
        
        print("\n" + "="*50)
        print("BATCH PREDICTION SUMMARY")
        print("="*50)
        print(f"Total Images: {summary['total_images']}")
        print(f"PASS: {summary['pass']} ({summary['pass_rate']*100:.1f}%)")
        print(f"FAIL: {summary['fail']} ({summary['fail_rate']*100:.1f}%)")
        print(f"HOLD: {summary['hold']} ({summary['hold_rate']*100:.1f}%)")
        print(f"Errors: {summary['errors']}")
        print("="*50)
        
        # Print individual results
        print("\nDetailed Results:")
        for r in results:
            if 'error' in r:
                print(f"  {r['image_path']}: ERROR - {r['error']}")
            else:
                print(f"  {r['image_name']}: {r['decision']} (conf: {r['confidence']:.3f})")


if __name__ == "__main__":
    main()
