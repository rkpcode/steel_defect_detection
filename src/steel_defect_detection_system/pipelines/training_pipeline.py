"""
Training Pipeline - Steel Defect Detection System
==================================================

Step-by-step pipeline:
1. Data Ingestion - Download/load data, create train/test split
2. Data Transformation - Patch extraction, augmentation, TF datasets
3. Model Training - Baseline CNN / Transfer Learning with class weights
4. Model Evaluation - (Phase 5 - to be added)
"""

import os
import sys
from pathlib import Path
from src.steel_defect_detection_system.logger import logger
from src.steel_defect_detection_system.exception import CustomException
from src.steel_defect_detection_system.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.steel_defect_detection_system.components.data_transformation import DataTransformation, DataTransformationConfig
from src.steel_defect_detection_system.components.data_transformation_optimized import OptimizedDataTransformation
from src.steel_defect_detection_system.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.steel_defect_detection_system.components.model_evaluation import ModelEvaluation, ModelEvaluationConfig
class TrainingPipeline:
    """
    End-to-end training pipeline for Steel Defect Detection.
    
    Executes components in sequence:
    1. Data Ingestion
    2. Data Transformation
    3. Model Training (Phase 4)
    4. Model Evaluation (Phase 5)
    """
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.transformation_config = DataTransformationConfig()
        logger.info("Training Pipeline initialized")
    
    def run_step_1_data_ingestion(self):
        """
        STEP 1: Data Ingestion
        
        - Download data from Kaggle (if needed)
        - Parse annotations
        - Create stratified train/test split
        """
        logger.info("=" * 60)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 60)
        
        try:
            ingestion = DataIngestion(self.ingestion_config)
            train_path, test_path, stats = ingestion.initiate_data_ingestion()
            
            logger.info(f"\n✅ Step 1 Complete!")
            logger.info(f"   Train CSV: {train_path}")
            logger.info(f"   Test CSV: {test_path}")
            logger.info(f"   Total images: {stats['total_images']}")
            
            return train_path, test_path, stats
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_step_2_data_transformation(self, train_path: str, test_path: str,
                                        max_train_images: int = None,
                                        max_test_images: int = None,
                                        memory_efficient: bool = False,
                                        optimized: bool = True):
        """
        STEP 2: Data Transformation
        
        - Extract patches from images (256x256, 50% overlap)
        - Generate patch-level labels from masks
        - Apply safe augmentations
        - Create TensorFlow datasets with class balancing
        
        Args:
            train_path: Path to train CSV
            test_path: Path to test CSV
            max_train_images: Limit training images (for testing)
            max_test_images: Limit test images (for testing)
            memory_efficient: Use disk-based approach for large datasets
            optimized: Use on-the-fly patch extraction (RECOMMENDED!)
        """
        logger.info("=" * 60)
        logger.info("STEP 2: DATA TRANSFORMATION")
        logger.info("=" * 60)
        
        try:
            if optimized:
                # RECOMMENDED: On-the-fly patch extraction
                logger.info("Using OPTIMIZED mode (on-the-fly patches, no disk storage)")
                transformer = OptimizedDataTransformation(self.transformation_config)
                result = transformer.initiate_data_transformation(
                    train_path=train_path,
                    test_path=test_path,
                    max_train_images=max_train_images,
                    max_test_images=max_test_images
                )
                logger.info(f"\n[OK] Step 2 Complete!")
                logger.info(f"   Train patches (estimated): {result['train_patches']}")
                logger.info(f"   Test patches (estimated): {result['test_patches']}")
            elif memory_efficient:
                logger.info("Using MEMORY-EFFICIENT mode (disk-based)")
                transformer = DataTransformation(self.transformation_config)
                result = transformer.initiate_data_transformation_memory_efficient(
                    train_path=train_path,
                    test_path=test_path,
                    max_train_images=max_train_images,
                    max_test_images=max_test_images
                )
                logger.info(f"\n[OK] Step 2 Complete!")
                logger.info(f"   Train patches: {len(result['y_train'])}")
                logger.info(f"   Test patches: {len(result['X_test'])}")
            else:
                transformer = DataTransformation(self.transformation_config)
                result = transformer.initiate_data_transformation(
                    train_path=train_path,
                    test_path=test_path,
                    max_train_images=max_train_images,
                    max_test_images=max_test_images
                )
                logger.info(f"\n[OK] Step 2 Complete!")
                logger.info(f"   Train patches: {len(result['X_train'])}")
                logger.info(f"   Test patches: {len(result['X_test'])}")
            
            logger.info(f"   Class weights: {result['class_weights']}")
            
            return result
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_step_3_model_training(self, transformation_result: dict,
                                   model_type: str = "baseline",
                                   fine_tune: bool = False,
                                   epochs: int = 10):
        """
        STEP 3: Model Training
        
        - Build baseline or transfer learning model
        - Compile with class-weighted loss
        - Train with early stopping
        - Monitor recall and F2-score
        
        Args:
            transformation_result: Output from Step 2
            model_type: "baseline" or "transfer"
            fine_tune: Whether to fine-tune (transfer only)
            epochs: Number of training epochs
        """
        logger.info("=" * 60)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 60)
        
        try:
            # Get datasets and class weights
            train_dataset = transformation_result['train_dataset']
            test_dataset = transformation_result['test_dataset']
            class_weights = transformation_result['class_weights']
            
            # Configure trainer
            trainer_config = ModelTrainerConfig()
            trainer_config.epochs = epochs
            
            trainer = ModelTrainer(trainer_config)
            
            # Train model
            result = trainer.initiate_model_training(
                train_dataset=train_dataset,
                val_dataset=test_dataset,
                class_weights=class_weights,
                model_type=model_type,
                fine_tune=fine_tune
            )
            
            logger.info(f"\n[OK] Step 3 Complete!")
            logger.info(f"   Model type: {model_type}")
            logger.info(f"   Final val recall: {result['history'].get('val_recall', [0])[-1]:.4f}")
            
            return result
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_step_4_model_evaluation(self, training_result: dict,
                                     transformation_result: dict):
        """
        STEP 4: Model Evaluation
        
        - Generate predictions on test set
        - Calculate metrics (Recall, Precision, F2)
        - Plot confusion matrix and PR curve
        - Find optimal threshold
        - Generate evaluation report
        
        Args:
            training_result: Output from Step 3
            transformation_result: Output from Step 2 (for test data)
        """
        logger.info("=" * 60)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 60)
        
        try:
            model = training_result['model']
            X_test = transformation_result['X_test']
            y_test = transformation_result['y_test']
            
            evaluator = ModelEvaluation()
            result = evaluator.initiate_model_evaluation(
                model=model,
                X_test=X_test,
                y_test=y_test
            )
            
            logger.info(f"\n[OK] Step 4 Complete!")
            logger.info(f"   Default threshold recall: {result['metrics_default']['recall']:.4f}")
            logger.info(f"   Optimal threshold: {result['optimal_threshold']:.3f}")
            logger.info(f"   Optimal recall: {result['metrics_optimal']['recall']:.4f}")
            
            return result
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_pipeline(self, max_train_images: int = None, max_test_images: int = None,
                     model_type: str = "baseline", epochs: int = 10, 
                     skip_training: bool = False, skip_evaluation: bool = False,
                     memory_efficient: bool = False):
        """
        Run complete training pipeline.
        
        Args:
            max_train_images: Limit for testing (None = all)
            max_test_images: Limit for testing (None = all)
            model_type: "baseline" or "transfer"
            epochs: Training epochs
            skip_training: Skip Step 3 (for data testing only)
            skip_evaluation: Skip Step 4 (for training only)
            memory_efficient: Use disk-based approach for large datasets
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("Steel Defect Detection System")
        logger.info("=" * 60)
        
        # Step 1: Data Ingestion
        train_path, test_path, ingestion_stats = self.run_step_1_data_ingestion()
        
        # Step 2: Data Transformation
        transformation_result = self.run_step_2_data_transformation(
            train_path=train_path,
            test_path=test_path,
            max_train_images=max_train_images,
            max_test_images=max_test_images,
            memory_efficient=memory_efficient
        )
        
        # Step 3: Model Training
        training_result = None
        if not skip_training:
            training_result = self.run_step_3_model_training(
                transformation_result=transformation_result,
                model_type=model_type,
                epochs=epochs
            )
        
        # Step 4: Model Evaluation
        evaluation_result = None
        if not skip_evaluation and training_result is not None:
            evaluation_result = self.run_step_4_model_evaluation(
                training_result=training_result,
                transformation_result=transformation_result
            )
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return {
            'ingestion_stats': ingestion_stats,
            'transformation_result': transformation_result,
            'training_result': training_result,
            'evaluation_result': evaluation_result
        }


def main():
    """Run training pipeline with optional sample size for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Steel Defect Detection Training Pipeline')
    parser.add_argument('--max-train', type=int, default=None,
                        help='Max training images (for testing)')
    parser.add_argument('--max-test', type=int, default=None,
                        help='Max test images (for testing)')
    parser.add_argument('--test-run', action='store_true',
                        help='Quick test with 10 train, 5 test images')
    
    args = parser.parse_args()
    
    # Test run with small sample
    if args.test_run:
        args.max_train = 10
        args.max_test = 5
    
    pipeline = TrainingPipeline()
    result = pipeline.run_pipeline(
        max_train_images=args.max_train,
        max_test_images=args.max_test
    )
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Train patches: {len(result['transformation_result']['X_train'])}")
    print(f"Test patches: {len(result['transformation_result']['X_test'])}")
    print(f"Class weights: {result['transformation_result']['class_weights']}")


if __name__ == "__main__":
    main()
