import numpy as np
import torch

from moe.training import kfold, robust, stratified, standard
from utils.args import parse_args, print_config
from utils.seeding import set_seed
from utils.tracking import ExpertTracker
from moe.factories.datasets_factory import DatasetFactory
from data_handlers.data_utils import DataProcessor
from utils.app_logger import AppLogger
import sys

def main():
    # Initialize logger
    logger = AppLogger(name="Guided MOE", log_dir="outputs", log_file="app.log")
    
    try:
        # Parse arguments and setup
        config = parse_args()
        print_config(config, logger)
        
        # Set device and seed
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        set_seed(config.seed)
        logger.info(f"Using device: {device}")
        
        # Choose training mode based on config
        if config.training_config.training_mode == 'kfold':
            logger.info("Starting K-Fold Cross Validation Training")
            results = kfold.kfold_training(config, logger, device, n_splits=config.n_splits)
            
            # Log summary of k-fold results
            accuracies = [fold['best_val_acc'] for fold in results]
            logger.info(f"\nK-Fold Training Summary:")
            logger.info(f"Mean Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
            
        elif config.training_config.training_mode == 'stratified':
            logger.info("Starting Stratified K-Fold Cross Validation Training")
            results = stratified.stratified_kfold_training(config, logger, device, n_splits=config.n_splits)
            
            # Log summary of stratified k-fold results
            accuracies = [fold['best_val_acc'] for fold in results]
            logger.info(f"\nStratified K-Fold Training Summary:")
            logger.info(f"Mean Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
            
        elif config.training_config.training_mode == 'standard':
            logger.info("Starting Standard Train/Test Training")
            results = standard.train_test_training(config, logger, device)
            logger.info(f"\nTraining Summary:")
            logger.info(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
            logger.info(f"Model saved at: {results['model_path']}")
        
        elif config.training_config.training_mode == 'robust':
            logger.info("Starting Robust Training")
            results = robust.robust_training(config, logger, device)
            logger.info(f"\nTraining Summary:")
            logger.info(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
            logger.info(f"Model saved at: {results['model_path']}")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error("Exiting training loop")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("Cleaning up and closing logger")

if __name__ == "__main__":
    main()