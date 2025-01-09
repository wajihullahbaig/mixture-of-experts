import torch
import torch.optim as optim

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

def main():
    logger = AppLogger(name="Guided MOE",log_dir="outputs",log_file="app.log")
    
    # Parse arguments
    config = parse_args()
    print_config(config,logger)
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    set_seed(config.seed)
    
    try:
        # Create data loaders
        train_loader, val_loader = DatasetFactory.create_dataloaders(
        dataset=config.dataset_name,
        data_dir=config.data_dir,
        batch_size=config.training_config.batch_size,
        num_workers=config.training_config.num_workers,
        architecture=config.moe_config.architecture
    )   
        # Create model using factory
        model = MoEFactory.create_moe(config.moe_config).to(device)
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training_config.learning_rate,
            weight_decay=config.training_config.weight_decay
        )
        
        # Create tracker
        dcfg = DatasetFactory.get_dataset_config(config.dataset_name, config.moe_config.architecture)    

        expert_assignments = None
        if config.moe_config.moe_type.name == 'GUIDED': 
            expert_assignments = create_expert_assignments(dcfg["num_classes"], config.moe_config.num_experts)

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
        for epoch in range(config.training_config.num_epochs):
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, data_processor, device, tracker, epoch, config.nan_check
            )
            
            # Evaluate
            val_metrics = evaluate(
                model, val_loader, data_processor, device, tracker, epoch,config.nan_check
            )
            
            # Log metrics
            tracker.log_metrics(epoch, train_metrics, val_metrics)
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                tracker.save_model(model, optimizer, epoch, val_metrics, is_best=True)
            
            # Regular checkpoint
            if epoch % 10 == 0:
                tracker.save_model(model, optimizer, epoch, val_metrics)
            
            # Print progress
            logger.info(f"Epoch {epoch}:")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            logger.info("-" * 50)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error("Exiting training loop")
        sys.exit(1)

if __name__ == "__main__":
    main()