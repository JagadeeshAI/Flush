from data import get_dataloaders
from utils import IrisTrainer


def main():
    """Main training pipeline"""
    print("="*60)
    print("Iris Recognition Training Pipeline")
    print("Paper: Novel Deep Learning Network with LAM")
    print("="*60)
    
    # Configuration
    NUM_CLASSES = 1000
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    PRETRAIN_EPOCHS = 200
    FINETUNE_EPOCHS = 100
    TARGET_ACC = 95.0
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, test_loader = get_dataloaders(
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = IrisTrainer(
        num_classes=NUM_CLASSES,
        device='cuda',
        learning_rate=LEARNING_RATE
    )
    
    # Run full training pipeline
    print("\nStarting training...")
    best_acc = trainer.train_full_pipeline(
        train_loader=train_loader,
        test_loader=test_loader, 
        pretrain_epochs=PRETRAIN_EPOCHS,
        finetune_epochs=FINETUNE_EPOCHS,
        target_acc=TARGET_ACC
    )
    
    print(f"\n✓ Training complete! Best accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()