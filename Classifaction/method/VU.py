import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from codes.data import get_dataloaders
from codes.utils import get_model
from method.utils import (
    extract_embeddings, extract_features, compute_class_centroids, compute_voronoi_vertices,
    select_target_vertices, assign_targets_to_classes, compute_forget_loss, compute_retain_loss,
    compute_classification_loss
)
from tqdm import tqdm


class VoronoiUnlearning:
    def __init__(self, model_path, forget_classes, retain_classes, device="cuda", method="simple"):
        self.device = device
        self.forget_classes = forget_classes
        self.retain_classes = retain_classes
        self.method = method
        
        # Load model
        num_classes = len(forget_classes) + len(retain_classes)
        self.model = get_model(num_classes, model_path=model_path, device=device)
        
        # Initialize target assignment
        self.target_assignment = None
        self.retain_centroids = None
        self.forget_centroids = None
    
    def setup_targets(self, dataloader):
        """Extract centroids and compute Voronoi vertices for target assignment"""
        import gc
        
        print("Extracting embeddings...")
        embeddings, labels = extract_embeddings(self.model, dataloader, self.device)
        
        print("Computing retain class centroids...")
        self.retain_centroids = compute_class_centroids(embeddings, labels, self.retain_classes)
        
        # For advance method, also compute forget centroids
        if self.method == "advance":
            print("Computing forget class centroids for adaptive assignment...")
            self.forget_centroids = compute_class_centroids(embeddings, labels, self.forget_classes)
        
        # Clear embeddings to free memory
        del embeddings, labels
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"Computing Voronoi vertices for {len(self.retain_centroids)} retain classes...")
        vertices, degrees = compute_voronoi_vertices(self.retain_centroids, max_vertices=50)
        
        print(f"Selecting {len(self.forget_classes)} target vertices...")
        target_vertices = select_target_vertices(vertices, degrees, len(self.forget_classes))
        
        # Clear intermediate variables
        del vertices, degrees
        gc.collect()
        
        print(f"Assigning targets to forget classes using '{self.method}' method...")
        self.target_assignment = assign_targets_to_classes(
            target_vertices, self.forget_classes, self.forget_centroids, self.method
        )
        
        print(f"Setup complete: {len(self.target_assignment)} targets assigned")
        
        # Final cleanup
        del target_vertices
        gc.collect()
    
    def unlearning_step_forget(self, batch, optimizer):
        """Process forget batch and return loss"""
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        
        # Get features from penultimate layer
        embeddings = extract_features(self.model, imgs)
        
        # L_forget = Σ MSE(embedding(x_forget), assigned_vertex)
        loss_forget = compute_forget_loss(embeddings, labels, self.target_assignment)*5
        
        loss_components = {'forget': loss_forget.item()}
        
        return loss_forget, loss_components
    
    def unlearning_step_retain(self, batch, optimizer):
        """Process retain batch and return loss"""
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        
        # Get features from penultimate layer
        embeddings = extract_features(self.model, imgs)
        
        # L_retain = Σ MSE(embedding(x_retain), original_centroid) + CE loss
        # MSE loss - anchor to centroids
        loss_retain_mse = compute_retain_loss(embeddings, labels, self.retain_centroids)
        
        # CE loss - classification loss on retain samples
        logits = self.model(imgs)
        loss_retain_ce = nn.CrossEntropyLoss()(logits, labels)
        
        # Total retain loss
        loss_retain_total = loss_retain_mse + loss_retain_ce
        
        loss_components = {
            'retain_mse': loss_retain_mse.item(),
            'retain_ce': loss_retain_ce.item()
        }
        
        return loss_retain_total, loss_components
    
    def unlearn(self, data_dir, batch_size=32, epochs=5, lr=1e-4, lambda_retain=3.0):
        """Main unlearning loop"""
        # Get forget dataloaders (classes 0-49, ratio 1.0)
        forget_train_loader, forget_val_loader, _ = get_dataloaders(
            data_dir, batch_size, class_range=(0, 49), data_ratio=1.0
        )
        
        # Get retain dataloaders (classes 50-99, ratio 0.1)
        retain_train_loader, retain_val_loader, _ = get_dataloaders(
            data_dir, batch_size, class_range=(50, 99), data_ratio=0.1
        )
        
        # For advance method, we need both forget and retain data to compute centroids
        if self.method == "advance":
            # Get combined loader for computing both forget and retain centroids
            combined_val_loader, _, _ = get_dataloaders(
                data_dir, batch_size, class_range=(0, 99), data_ratio=0.1
            )
            self.setup_targets(combined_val_loader)
        else:
            # For simple method, only need retain centroids
            self.setup_targets(retain_val_loader)
        
        # Initial evaluation before unlearning
        print("\n=== Initial Evaluation ===")
        forget_acc, retain_acc = evaluate_on_ranges(self.model, data_dir, self.forget_classes, self.retain_classes, batch_size//2, self.device)
        print(f"Forget Acc: {forget_acc:.2f}% | Retain Acc: {retain_acc:.2f}%")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        print(f"\n=== Starting Voronoi Unlearning for {epochs} epochs ===")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {'forget': 0.0, 'retain_mse': 0.0, 'retain_ce': 0.0, 'count': 0}
            
            # Debug: Check loader sizes
            print(f"Forget loader batches: {len(forget_train_loader)}, Retain loader batches: {len(retain_train_loader)}")
            
            # Create iterators for alternating batches
            forget_iter = iter(forget_train_loader)
            retain_iter = iter(retain_train_loader)
            
            # Calculate total steps (use minimum of both loaders)
            total_steps = min(len(forget_train_loader), len(retain_train_loader))
            
            # Alternate between forget and retain batches
            pbar = tqdm(range(total_steps), desc=f"Epoch {epoch+1}/{epochs}")
            for step in pbar:
                # Process one forget batch
                try:
                    forget_batch = next(forget_iter)
                    loss_forget, loss_components = self.unlearning_step_forget(forget_batch, optimizer)
                    
                    # Backprop for forget loss
                    optimizer.zero_grad()
                    loss_forget.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)  # Add this
                    optimizer.step()
                    
                    for key, value in loss_components.items():
                        epoch_losses[key] = epoch_losses.get(key, 0.0) + value
                    epoch_losses['count'] += 1
                except StopIteration:
                    break
                
                # Process one retain batch
                try:
                    retain_batch = next(retain_iter)
                    loss_retain, loss_components = self.unlearning_step_retain(retain_batch, optimizer)
                    
                    # Backprop for retain loss
                    optimizer.zero_grad()
                    loss_retain.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)  # Add this
                    optimizer.step()
                    
                    for key, value in loss_components.items():
                        epoch_losses[key] = epoch_losses.get(key, 0.0) + value
                    epoch_losses['count'] += 1
                except StopIteration:
                    break
                
                # Update progress bar with current losses
                if epoch_losses['count'] > 0:
                    avg_forget = epoch_losses['forget'] / epoch_losses['count']
                    avg_retain_mse = epoch_losses['retain_mse'] / epoch_losses['count']
                    avg_retain_ce = epoch_losses['retain_ce'] / epoch_losses['count']
                    pbar.set_postfix({
                        'Forget': f"{avg_forget:.4f}", 
                        'R_MSE': f"{avg_retain_mse:.4f}",
                        'R_CE': f"{avg_retain_ce:.4f}"
                    })
            
            # Print epoch summary
            if epoch_losses['count'] > 0:
                avg_forget = epoch_losses.get('forget', 0.0) / epoch_losses['count']
                avg_retain_mse = epoch_losses.get('retain_mse', 0.0) / epoch_losses['count']
                avg_retain_ce = epoch_losses.get('retain_ce', 0.0) / epoch_losses['count']
                print(f"Epoch {epoch+1}: Forget: {avg_forget:.4f}, Retain MSE: {avg_retain_mse:.4f}, Retain CE: {avg_retain_ce:.4f}")
            
            # Evaluate after each epoch with memory optimization
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            forget_acc, retain_acc = evaluate_on_ranges(self.model, data_dir, self.forget_classes, self.retain_classes, batch_size//2, self.device)
            print(f"         Forget Acc: {forget_acc:.2f}% | Retain Acc: {retain_acc:.2f}%")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return self.model


def evaluate_on_ranges(model, data_dir, forget_classes, retain_classes, batch_size=64, device="cuda"):
    """Evaluate model on forget and retain class ranges"""
    from codes.data import get_dataloaders
    
    # Evaluate on forget classes (use ratio 1.0 for complete evaluation)
    if forget_classes:
        forget_range = (min(forget_classes), max(forget_classes))
        _, forget_val_loader, _ = get_dataloaders(data_dir, batch_size, class_range=forget_range, data_ratio=1.0)
        forget_acc = evaluate_accuracy(model, forget_val_loader, device)
    else:
        forget_acc = 0.0
    
    # Evaluate on retain classes (use ratio 0.1 for faster evaluation)
    if retain_classes:
        retain_range = (min(retain_classes), max(retain_classes))
        _, retain_val_loader, _ = get_dataloaders(data_dir, batch_size, class_range=retain_range, data_ratio=0.1)
        retain_acc = evaluate_accuracy(model, retain_val_loader, device)
    else:
        retain_acc = 0.0
    
    return forget_acc, retain_acc


def evaluate_accuracy(model, dataloader, device):
    """Compute accuracy on a dataloader"""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            predicted = outputs.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Clear memory after each batch
            del imgs, labels, outputs, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return (correct / total) * 100 if total > 0 else 0.0


def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Voronoi Unlearning Methods')
    parser.add_argument('--method', type=str, choices=['simple', 'advance'], default='simple',
                       help='Assignment method: simple (random) or advance (nearest)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Configuration
    DATA_DIR = "/media/jag/volD2/cifer100/cifer"
    MODEL_PATH = "checkpoints/best.pth"  # Pre-trained model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define forget/retain split
    forget_classes = list(range(0, 50))   # Classes 0-49
    retain_classes = list(range(50, 100)) # Classes 50-99
    
    print(f"=== Voronoi Unlearning ({args.method.upper()}) ===")
    print(f"Method: {args.method}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Create Voronoi Unlearning instance with selected method
    vu = VoronoiUnlearning(MODEL_PATH, forget_classes, retain_classes, DEVICE, method=args.method)
    
    # Run unlearning
    unlearned_model = vu.unlearn(
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_retain=3.0
    )
    
    # Save unlearned model with method name
    output_filename = f'voronoi_unlearned_{args.method}.pth'
    torch.save({
        'model_state': unlearned_model.state_dict(),
        'forget_classes': forget_classes,
        'retain_classes': retain_classes,
        'method': args.method,
    }, output_filename)
    
    print(f"Voronoi Unlearning completed! Model saved to '{output_filename}'")


if __name__ == "__main__":
    main()