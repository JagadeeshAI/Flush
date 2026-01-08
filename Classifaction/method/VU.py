import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from codes.data import get_dataloaders
from codes.utils import get_model
from utils.utils import extract_embeddings, extract_features, compute_class_centroids, evaluate_model, get_visualization_data, create_visualization_step
from method.utils import *
from tqdm import tqdm


class VoronoiUnlearning:
    def __init__(self, model_path, forget_classes, retain_classes, device="cuda", method="simple", 
                 use_regularization=False, lambda_reg=1e-3, enable_visualization=False, 
                 vis_output_dir="visualizations", use_color=True):
        self.device, self.forget_classes, self.retain_classes = device, forget_classes, retain_classes
        self.method, self.use_regularization, self.lambda_reg = method, use_regularization, lambda_reg
        self.enable_visualization = enable_visualization
        
        num_classes = len(forget_classes) + len(retain_classes)
        self.model = get_model(num_classes, model_path=model_path, device=device)
        self.target_assignment = self.retain_centroids = self.forget_centroids = self.visualizer = None
        
        if enable_visualization:
            from visual import TSNEVoronoiVisualizer
            self.visualizer = TSNEVoronoiVisualizer(self.model, forget_classes, retain_classes, 
                                                   device=device, output_dir=vis_output_dir, use_color=use_color)
    
    def setup_targets(self, dataloader):
        """Setup target assignment using classifier weights"""
        import gc
        print("Setting up targets...")
        
        embeddings, labels = extract_embeddings(self.model, dataloader, self.device)
        self.retain_centroids = compute_class_centroids(embeddings, labels, self.retain_classes)
        self.forget_centroids = compute_class_centroids(embeddings, labels, self.forget_classes)
        del embeddings, labels; gc.collect()
        
        # Extract weights and compute vertices
        retain_weights = extract_classifier_weights(self.model, self.retain_classes)
        vertices, degrees = compute_voronoi_vertices_from_weights(retain_weights, max_vertices=50)
        target_vertices = select_target_vertices(vertices, degrees, len(self.forget_classes))
        
        # Orthogonalize targets to forget weights
        forget_weights = extract_classifier_weights(self.model, self.forget_classes)
        forget_weight_vectors = torch.stack(list(forget_weights.values()))
        if target_vertices.device != forget_weight_vectors.device:
            forget_weight_vectors = forget_weight_vectors.to(target_vertices.device)
        target_vertices = orthogonalize_embeddings_to_weights(target_vertices, forget_weight_vectors)
        
        self.target_assignment = assign_targets_to_classes(target_vertices, self.forget_classes, self.forget_centroids, self.method)
        
        if self.visualizer: self.visualizer.setup_visualization(get_visualization_data(dataloader))
        
        print(f"Setup complete: {len(self.target_assignment)} targets assigned")
        del target_vertices, retain_weights, forget_weights, forget_weight_vectors; gc.collect()
    
    def compute_losses(self, batch, global_step, is_forget=True):
        """Compute losses for forget or retain batch"""
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        embeddings = extract_features(self.model, imgs)
        
        if is_forget:
            loss_main = compute_forget_loss(embeddings, labels, self.target_assignment)
            logits = self.model(imgs)
            loss_constraint = compute_auxiliary_logit_constraint(logits, self.forget_classes, self.retain_classes, margin=2.0)
            total_loss = loss_main + 0.1 * loss_constraint
            components = {'forget': loss_main.item(), 'constraint': loss_constraint.item()}
        else:
            loss_retain_mse = compute_retain_loss(embeddings, labels, self.retain_centroids)
            logits = self.model(imgs)
            loss_retain_ce = nn.CrossEntropyLoss()(logits, labels)
            total_loss = loss_retain_mse + loss_retain_ce * 0.1
            components = {'retain_mse': loss_retain_mse.item(), 'retain_ce': loss_retain_ce.item()}
        
        if self.use_regularization and global_step > 1000:
            reg_loss = compute_group_sparse_regularization(self.model, self.lambda_reg)
            total_loss += reg_loss
            components['reg'] = reg_loss.item()
        
        return total_loss, components
    
    def unlearn(self, data_dir, batch_size=32, epochs=5, lr=1e-4):
        """Main unlearning loop"""
        # Setup data loaders
        forget_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=(0, 49), data_ratio=1.0)
        retain_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=(50, 99), data_ratio=0.1)
        combined_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=(0, 99), data_ratio=0.1)
        
        self.setup_targets(combined_loader)
        
        # Initial evaluation
        print("\n=== Initial Evaluation ===")
        forget_acc, retain_acc = evaluate_model(self.model, data_dir, batch_size//2, self.forget_classes, self.retain_classes, self.device)
        print(f"Forget Acc: {forget_acc:.2f}% | Retain Acc: {retain_acc:.2f}%")
        if self.enable_visualization: create_visualization_step(self.visualizer, self.model, 0, data_dir, batch_size, self.device, self.target_assignment)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        max_steps = max(len(forget_loader), len(retain_loader)) * epochs
        forget_iter, retain_iter = iter(forget_loader), iter(retain_loader)
        losses = {'forget': 0.0, 'constraint': 0.0, 'retain_mse': 0.0, 'retain_ce': 0.0, 'reg': 0.0, 'count': 0}
        
        # Training loop
        pbar = tqdm(range(max_steps), desc="Training")
        for step in range(1, max_steps + 1):
            self.model.train()
            
            # Process forget and retain batches
            try: forget_batch = next(forget_iter)
            except StopIteration: forget_iter = iter(forget_loader); forget_batch = next(forget_iter)
            
            try: retain_batch = next(retain_iter)
            except StopIteration: retain_iter = iter(retain_loader); retain_batch = next(retain_iter)
            
            # Compute losses and optimize
            for batch, is_forget in [(forget_batch, True), (retain_batch, False)]:
                loss, comps = self.compute_losses(batch, step, is_forget)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                for k, v in comps.items(): losses[k] = losses.get(k, 0.0) + v
                losses['count'] += 1
            
            # Progress and validation
            if losses['count'] > 0:
                avg_losses = {k: f"{v/losses['count']:.4f}" for k, v in losses.items() if k != 'count'}
                pbar.set_postfix(avg_losses)
            
            if step % 100 == 0:
                f_acc, r_acc = evaluate_model(self.model, data_dir, batch_size//2, self.forget_classes, self.retain_classes, self.device)
                print(f"\nStep {step} - Forget: {f_acc:.2f}% | Retain: {r_acc:.2f}%")
                if self.enable_visualization: create_visualization_step(self.visualizer, self.model, step, data_dir, batch_size, self.device, self.target_assignment)
                losses = {'forget': 0.0, 'constraint': 0.0, 'retain_mse': 0.0, 'retain_ce': 0.0, 'reg': 0.0, 'count': 0}
            
            pbar.update(1)
        
        pbar.close()
        return self.model
    


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['simple', 'advance'], default='simple')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use-reg', choices=['yes', 'no'], default='no')
    parser.add_argument('--lambda-reg', type=float, default=0.01)
    parser.add_argument('--enable-viz', action='store_true')
    parser.add_argument('--viz-dir', default='visualizations')
    parser.add_argument('--grayscale', action='store_true')
    args = parser.parse_args()
    
    DATA_DIR, MODEL_PATH = "/media/jag/volD2/cifer100/cifer", "checkpoints/best.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    forget_classes, retain_classes = list(range(0, 50)), list(range(50, 100))
    
    print(f"=== Voronoi Unlearning ({args.method}) ===")
    vu = VoronoiUnlearning(MODEL_PATH, forget_classes, retain_classes, DEVICE, 
                          method=args.method, use_regularization=args.use_reg=='yes', lambda_reg=args.lambda_reg,
                          enable_visualization=args.enable_viz, vis_output_dir=args.viz_dir, use_color=not args.grayscale)
    
    model = vu.unlearn(DATA_DIR, args.batch_size, args.epochs, args.lr)
    torch.save({'model_state': model.state_dict(), 'forget_classes': forget_classes, 
               'retain_classes': retain_classes, 'method': args.method}, f'voronoi_unlearned_{args.method}.pth')
    print(f"Model saved to 'voronoi_unlearned_{args.method}.pth'")
    
    if args.enable_viz and vu.visualizer:
        print(f"Animation script: {vu.visualizer.create_animation_script(fps=2)}")


if __name__ == "__main__":
    main()