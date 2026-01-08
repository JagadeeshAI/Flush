import numpy as np
import torch
from codes.model import create_iris_recognition_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

class IrisMetricEvaluator:
    """Extract embeddings and visualize iris identity clusters."""

    def __init__(self, checkpoint_path, num_classes=1000, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load trained model
        self.model = create_iris_recognition_model(num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
    
    def extract_embeddings(self, dataloader):
        """Extract feature embeddings for all samples"""
        embeddings = []
        labels = []

        with torch.no_grad():
            for images, lbls in tqdm(dataloader, desc='Extracting embeddings'):
                images = images.to(self.device)
                feats = self.model.extract_features(images)

                embeddings.append(feats.cpu().numpy())
                labels.append(lbls.numpy())

        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)

        # L2 normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        return embeddings, labels

    def visualize_subset_tsne(self, embeddings, labels, n_classes=5, save_path='tsne_subset.png'):
        """Visualize random subset of classes using t-SNE"""
        print(f"\nVisualizing {n_classes} random classes...")

        unique_classes = np.unique(labels)
        selected_classes = np.random.choice(unique_classes, n_classes, replace=False)
        print(f"Selected classes: {selected_classes}")

        mask = np.isin(labels, selected_classes)
        subset_embeddings = embeddings[mask]
        subset_labels = labels[mask]

        print(f"Total samples: {len(subset_embeddings)}")
        for cls in selected_classes:
            print(f"  Class {cls}: {np.sum(subset_labels == cls)} samples")

        # t-SNE
        print("Computing t-SNE projection...")
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, len(subset_embeddings)//4))
        embeddings_2d = tsne.fit_transform(subset_embeddings)

        # Plot
        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for idx, cls in enumerate(selected_classes):
            m = subset_labels == cls
            plt.scatter(
                embeddings_2d[m, 0], embeddings_2d[m, 1],
                c=[colors[idx]], label=f"Person {cls}",
                alpha=0.7, s=100, edgecolors='black', linewidth=0.5
            )

        plt.title(f"t-SNE Visualization of {n_classes} Random Iris Identities", fontsize=14)
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
        plt.close()
