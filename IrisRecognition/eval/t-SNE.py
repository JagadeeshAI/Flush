import torch
import numpy as np


from codes.data import get_dataloaders
from eval.utils import IrisMetricEvaluator


def main():
    print("="*60)
    print("Iris Recognition - Embedding Cluster Visualization Only")
    print("="*60)
    
    NUM_CLASSES = 1000
    BATCH_SIZE = 64
    CHECKPOINT_PATH = 'checkpoints/best_softmax.pth'
    N_CLASSES_VISUALIZE = 10
    
    # Load data
    print("\nLoading dataset...")
    train_loader, test_loader = get_dataloaders(
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_workers=4
    )
    
    # Initialize evaluator
    evaluator = IrisMetricEvaluator(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=NUM_CLASSES,
        device='cuda'
    )
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    train_embeddings, train_labels = evaluator.extract_embeddings(train_loader)
    test_embeddings, test_labels = evaluator.extract_embeddings(test_loader)
    
    # Combine train+test for visualization
    all_embeddings = np.vstack([train_embeddings, test_embeddings])
    all_labels = np.concatenate([train_labels, test_labels])
    
    # ONLY perform t-SNE visualization
    evaluator.visualize_subset_tsne(
        all_embeddings,
        all_labels,
        n_classes=N_CLASSES_VISUALIZE,
        save_path='tsne_5_classes.png'
    )

    print("\n✓ Done! Only clustering visualization generated.")

    
if __name__ == "__main__":
    main()