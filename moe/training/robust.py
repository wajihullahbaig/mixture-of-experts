import torch.optim as optim

from moe.factories.moe_factory import MoEFactory
from moe.training.trainer import evaluate, train_epoch
from utils.tracking import ExpertTracker
from moe.factories.datasets_factory import DatasetFactory
from data_handlers.data_utils import DataProcessor
from utils.model_utils import create_expert_assignments
import sys

def robust_training(config, logger, device):
    """
    Robust training pipeline:
    1. Train final model on full dataset
    2. Evaluate on test set
    """
    try:
        
        # Get dataset configuration
        dcfg = DatasetFactory.get_dataset_config(config.dataset_name, config.moe_config.architecture)
        
        # Create data loaders for final training
        train_loader, test_loader = DatasetFactory.create_dataloaders(
            dataset=config.dataset_name,
            data_dir=config.data_dir,
            batch_size=config.training_config.batch_size,
            num_workers=config.training_config.num_workers,
            architecture=config.moe_config.architecture
        )
        
        # Create model
        model = MoEFactory.create_moe(config.moe_config).to(device)
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training_config.learning_rate,
            weight_decay=config.training_config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.training_config.learning_rate,
            epochs=config.training_config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # Warm-up period
            div_factor=25,
            final_div_factor=1e4
        )
        
        # Create expert assignments for guided version
        expert_assignments = None
        if config.moe_config.moe_type.name == 'GUIDED':
            expert_assignments = create_expert_assignments(
                dcfg["num_classes"], 
                config.moe_config.num_experts
            )
        
        # Create tracker
        tracker = ExpertTracker(
            model_type=config.moe_config.moe_type.name,
            architecture=config.moe_config.architecture.name,
            base_path=config.output_dir,
            dataset_name=config.dataset_name,
            num_experts=config.moe_config.num_experts,
            expert_label_assignments=expert_assignments
        )
        
        # Create data processor
        data_processor = DataProcessor(config.moe_config.architecture)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        best_model_path = None
        
        for epoch in range(config.training_config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{config.training_config.num_epochs}")
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Train
            train_metrics = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                data_processor=data_processor,
                device=device,
                tracker=tracker,
                epoch=epoch,
                nan_check=config.nan_check,
                scheduler=scheduler
            )
            
            # Validate
            val_metrics = evaluate(
                model=model,
                val_loader=test_loader,  # Using test set as validation
                data_processor=data_processor,
                device=device,
                tracker=tracker,
                epoch=epoch,
                nan_check=config.nan_check
            )
            
            # Log metrics
            tracker.log_metrics(epoch, train_metrics, val_metrics)
            
            # Save best model and handle early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                best_model_path = tracker.save_model(
                    model, optimizer, epoch, val_metrics, is_best=True
                )
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= config.training_config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Regular checkpoint
            if epoch % 10 == 0:
                tracker.save_model(model, optimizer, epoch, val_metrics)
            
            # Print progress
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Final Results
        logger.info("\nTraining Completed")
        logger.info("-" * 50)
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        logger.info(f"Best model saved at: {best_model_path}")
        
        return {
            'best_val_acc': best_val_acc,
            'model_path': best_model_path
        }
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error("Exiting training loop")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)