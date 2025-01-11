import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

from moe.factories.moe_factory import MoEFactory
from moe.training.trainer import evaluate, train_epoch
from utils.args import parse_args, print_config
from utils.seeding import set_seed
from utils.tracking import ExpertTracker
from moe.factories.datasets_factory import DatasetFactory
from data_handlers.data_utils import DataProcessor
from utils.model_utils import create_expert_assignments
from utils.app_logger import AppLogger
import sys

def kfold_training(config, logger, device, n_splits=5):
    """Run training with k-fold cross validation"""
    try:
        # Get dataset configuration
        dcfg = DatasetFactory.get_dataset_config(config.dataset_name, config.moe_config.architecture)    

        # Create full training dataset
        train_dataset = DatasetFactory.create_dataset(
            dataset=config.dataset_name,
            data_dir=config.data_dir,
            train=True,
            architecture=config.moe_config.architecture
        )

        # Initialize K-Fold
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        indices = list(range(len(train_dataset)))

        # Store results for each fold
        fold_results = []

        # Train and evaluate on each fold
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            logger.info(f'\nFold {fold + 1}/{n_splits}')
            logger.info('-' * 50)

            # Create data samplers and loaders for this fold
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.training_config.batch_size,
                sampler=train_sampler,
                num_workers=config.training_config.num_workers
            )

            val_loader = DataLoader(
                train_dataset,
                batch_size=config.training_config.batch_size,
                sampler=val_sampler,
                num_workers=config.training_config.num_workers
            )

            # Create new model instance for this fold
            model = MoEFactory.create_moe(config.moe_config).to(device)

            # Create optimizer
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.training_config.learning_rate,
                weight_decay=config.training_config.weight_decay
            )

            # Create expert assignments for guided version
            expert_assignments = None
            if config.moe_config.moe_type.name == 'GUIDED':
                expert_assignments = create_expert_assignments(
                    dcfg["num_classes"], 
                    config.moe_config.num_experts
                )

            # Create tracker for this fold
            fold_tracker = ExpertTracker(
                model_type=config.moe_config.moe_type.name,
                architecture=config.moe_config.architecture.name,
                base_path=config.output_dir ,
                dataset_name=config.dataset_name,
                num_experts=config.moe_config.num_experts,
                expert_label_assignments=expert_assignments,
                fold_no=fold
            )

            # Create data processor
            data_processor = DataProcessor(config.moe_config.architecture)

            # Training loop for this fold
            best_val_acc = 0
            patience_counter = 0
            best_model_path = None

            for epoch in range(config.training_config.num_epochs):
                # Train
                train_metrics = train_epoch(
                    model, train_loader, optimizer, data_processor,
                    device, fold_tracker, epoch, config.nan_check
                )

                # Evaluate
                val_metrics = evaluate(
                    model, val_loader, data_processor,
                    device, fold_tracker, epoch, config.nan_check
                )

                # Log metrics
                fold_tracker.log_metrics(epoch, train_metrics, val_metrics)

                # Save best model and handle early stopping
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    patience_counter = 0
                    # Save best model for this fold
                    best_model_path = fold_tracker.save_model(
                        model, optimizer, epoch, val_metrics, is_best=True
                    )
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= config.training_config.early_stopping_patience:
                    logger.info(f"Early stopping triggered on fold {fold + 1}")
                    break

                # Print progress
                logger.info(f"Fold {fold + 1}, Epoch {epoch + 1}:")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['accuracy']:.2f}%")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val Acc: {val_metrics['accuracy']:.2f}%")

            # Store results for this fold
            fold_results.append({
                'fold': fold + 1,
                'best_val_acc': best_val_acc,
                'model_path': best_model_path
            })

        # Print final cross-validation results
        logger.info("\nCross-validation Results:")
        logger.info('-' * 50)
        accuracies = [fold['best_val_acc'] for fold in fold_results]
        logger.info(f"Mean Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        for fold in fold_results:
            logger.info(f"Fold {fold['fold']}: {fold['best_val_acc']:.2f}%")
            logger.info(f"Best model saved at: {fold['model_path']}")

        return fold_results

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error("Exiting training loop")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
