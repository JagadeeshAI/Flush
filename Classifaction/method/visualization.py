"""
Clean t-SNE visualization module for unlearning.
- Uses SAME samples across all epochs (fixed at epoch 0)
- Aligns t-SNE coordinates to reference layout using Procrustes
- Fixed axis limits across epochs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import procrustes
import os


class TSNEVisualizer:
    def __init__(self, output_dir="visualizations", 
                 forget_classes=None, retain_classes=None,
                 n_forget_vis=5, n_retain_vis=10):
        """Initialize t-SNE visualizer with fixed samples."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Select subset of classes to visualize
        self.forget_vis = sorted(forget_classes[:n_forget_vis]) if forget_classes else []
        self.retain_vis = sorted(retain_classes[:n_retain_vis]) if retain_classes else []
        self.all_vis_classes = self.forget_vis + self.retain_vis
        
        # Reference data - stored at epoch 0 and REUSED for all epochs
        self.reference_images = None      # Fixed input images (N, C, H, W)
        self.reference_labels = None      # Fixed labels for those images
        self.reference_coords_2d = None   # Reference t-SNE coordinates (for alignment)
        self.reference_xlim = None        # Fixed x-axis limits
        self.reference_ylim = None        # Fixed y-axis limits
        
        # Initial epoch target coordinates (computed once, then aligned)
        self.initial_target_coords = None
        
        # Target assignments
        self.target_assignment = None
        
        # Colors
        self.forget_colors = plt.cm.Reds(np.linspace(0.4, 0.95, max(len(self.forget_vis), 1)))
        self.retain_colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.retain_vis), 1)))
        
        print(f"TSNEVisualizer initialized:")
        print(f"  Forget classes: {self.forget_vis}")
        print(f"  Retain classes: {self.retain_vis}")
        print(f"  Output: {output_dir}")
    
    def set_targets(self, target_assignment):
        """Store target assignments for forget classes"""
        self.target_assignment = target_assignment
        print(f"Targets set for {len(target_assignment)} forget classes")
    
    def _get_color(self, class_id):
        """Get color for a class"""
        if class_id in self.forget_vis:
            idx = self.forget_vis.index(class_id)
            return self.forget_colors[idx]
        elif class_id in self.retain_vis:
            idx = self.retain_vis.index(class_id)
            return self.retain_colors[idx]
        return 'gray'
    
    def _select_and_store_samples(self, dataloader, device, samples_per_class=50):
        """
        Extract and store FIXED reference samples from dataloader.
        Called once at epoch 0, stores the actual image tensors.
        """
        # Collect all data from dataloader first
        all_images, all_labels = [], []
        for imgs, lbls in dataloader:
            all_images.append(imgs)
            all_labels.extend(lbls.tolist())
        
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.tensor(all_labels)
        
        # Select samples per class (deterministic order)
        selected_indices = []
        for c in self.all_vis_classes:
            class_mask = (all_labels == c)
            class_indices = torch.where(class_mask)[0].numpy()
            
            # Sort for consistent selection
            class_indices = np.sort(class_indices)
            
            if len(class_indices) > samples_per_class:
                selected = class_indices[:samples_per_class]
            else:
                selected = class_indices
            
            selected_indices.extend(selected.tolist())
        
        # Sort indices for consistent order
        selected_indices = sorted(selected_indices)
        
        # Store reference images and labels
        self.reference_images = all_images[selected_indices]  # (N, C, H, W)
        self.reference_labels = all_labels[selected_indices].numpy()  # (N,)
        
        print(f"Stored {len(selected_indices)} reference samples for visualization")
        print(f"  Classes: {sorted(set(self.reference_labels.tolist()))}")
        
        return len(selected_indices)
    
    def _extract_embeddings_from_stored(self, model, device):
        """
        Extract embeddings for STORED reference images.
        Same images, same order → can be aligned with Procrustes.
        """
        from utilities.utils import extract_features
        
        model.eval()
        embeddings = []
        
        # Process in batches for memory efficiency
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(self.reference_images), batch_size):
                batch = self.reference_images[i:i+batch_size].to(device)
                emb = extract_features(model, batch)
                embeddings.append(emb.cpu())
        
        return torch.cat(embeddings, dim=0).numpy()
    
    def _compute_tsne(self, embeddings):
        """Compute t-SNE on embeddings - fixed random_state for reproducibility"""
        perplexity = min(30, max(5, len(embeddings) // 4))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init='random',
            random_state=42  # FIXED seed
        )
        return tsne.fit_transform(embeddings)
    
    def _align_to_reference(self, coords_2d):
        """
        Align current t-SNE coordinates to reference using Procrustes analysis.
        This removes rotation, scaling, and reflection differences.
        """
        if self.reference_coords_2d is None:
            return coords_2d
        
        # Procrustes expects matching shapes
        # Returns: standardized ref, standardized coords, disparity
        _, aligned, _ = procrustes(self.reference_coords_2d, coords_2d)
        
        # Scale and shift aligned coords to match original reference scale
        ref_mean = self.reference_coords_2d.mean(axis=0)
        ref_std = self.reference_coords_2d.std()
        aligned_mean = aligned.mean(axis=0)
        aligned_std = aligned.std()
        
        # Transform aligned to match reference coordinate system
        if aligned_std > 1e-8:
            aligned = (aligned - aligned_mean) / aligned_std * ref_std + ref_mean
        
        return aligned
    
    def _compute_target_coords(self, embeddings, coords_2d, labels):
        """Compute target positions in t-SNE space using KNN interpolation"""
        if not self.target_assignment:
            return {}
        
        target_coords = {}
        for class_id, target_emb in self.target_assignment.items():
            if class_id not in self.forget_vis:
                continue
            
            target_np = target_emb.cpu().numpy().reshape(1, -1)
            
            # Find k nearest neighbors in embedding space
            distances = np.linalg.norm(embeddings - target_np, axis=1)
            k = min(5, len(distances))
            nearest_idx = np.argsort(distances)[:k]
            
            weights = 1.0 / (distances[nearest_idx] + 1e-8)
            weights /= weights.sum()
            
            target_2d = np.average(coords_2d[nearest_idx], axis=0, weights=weights)
            target_coords[class_id] = target_2d
        
        return target_coords
    
    def plot_epoch(self, epoch, model, dataloader, device, samples_per_class=50, save=True):
        """
        Plot t-SNE visualization for a specific epoch.
        
        Epoch 0: 
            - Select and store FIXED reference samples
            - Compute t-SNE as reference layout
            - Store reference coordinates and axis limits
        
        Later epochs:
            - Use SAME stored samples
            - Extract new embeddings (model has changed)
            - Compute t-SNE and ALIGN to reference using Procrustes
            - Use FIXED axis limits and target coordinates
        """
        # Epoch 0: Store reference samples
        if self.reference_images is None:
            self._select_and_store_samples(dataloader, device, samples_per_class)
        
        # Extract embeddings for stored samples (same images, same order)
        sample_emb = self._extract_embeddings_from_stored(model, device)
        sample_labels = self.reference_labels
        
        print(f"Epoch {epoch}: Computing t-SNE with {len(sample_emb)} samples...")
        
        # Compute t-SNE
        coords_2d = self._compute_tsne(sample_emb)
        
        # Epoch 0: Store reference coordinates and frame
        if self.reference_coords_2d is None:
            self.reference_coords_2d = coords_2d.copy()
            
            padding = 5
            self.reference_xlim = (coords_2d[:, 0].min() - padding, coords_2d[:, 0].max() + padding)
            self.reference_ylim = (coords_2d[:, 1].min() - padding, coords_2d[:, 1].max() + padding)
            
            self.initial_target_coords = self._compute_target_coords(sample_emb, coords_2d, sample_labels)
            
            print(f"Reference frame set: x={self.reference_xlim}, y={self.reference_ylim}")
            print(f"Target coords computed for {len(self.initial_target_coords)} forget classes")
        else:
            # Align current coords to reference
            coords_2d = self._align_to_reference(coords_2d)
        
        xlim = self.reference_xlim
        ylim = self.reference_ylim
        target_coords = self.initial_target_coords  # FIXED target positions
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor('white')
        
        # Plot retain classes
        for c in self.retain_vis:
            mask = sample_labels == c
            if mask.sum() > 0:
                ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                          c=[self._get_color(c)], label=f'R{c}',
                          s=60, alpha=0.85, marker='o',
                          edgecolors='white', linewidths=0.5)
        
        # Plot forget classes
        for c in self.forget_vis:
            mask = sample_labels == c
            if mask.sum() > 0:
                ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                          c=[self._get_color(c)], label=f'F{c}',
                          s=70, alpha=0.9, marker='o',
                          edgecolors='white', linewidths=0.5)
                
                # Arrow to target (fixed target position)
                if c in target_coords:
                    centroid = coords_2d[mask].mean(axis=0)
                    target = target_coords[c]
                    
                    ax.annotate('', xy=target, xytext=centroid,
                               arrowprops=dict(arrowstyle='->', color='black',
                                             lw=1.5, alpha=0.6))
                    ax.scatter(*target, c='gold', s=200, marker='*',
                              edgecolors='black', linewidths=1.5, zorder=100)
        
        # Fixed axes
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.set_xlabel('t-SNE dimension 1', fontsize=11)
        ax.set_ylabel('t-SNE dimension 2', fontsize=11)
        title = "(a) Original" if epoch == 0 else f"(b) After Epoch {epoch}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save:
            path = os.path.join(self.output_dir, f'epoch_{epoch:04d}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return path
        else:
            plt.show()
            return None


def create_vis_dataloader(data_dir, batch_size, class_range, ratio=1.0, split='train'):
    """Create dataloader for visualization"""
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from codes.data import get_dataloaders
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size, class_range=class_range, data_ratio=ratio
    )
    
    if split == 'train':
        return train_loader
    elif split == 'val':
        return val_loader
    else:
        return test_loader
