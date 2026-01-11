import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time

from codes.data import get_dataloaders
from codes.utils import get_model
from utilities.utils import extract_embeddings, extract_features, compute_class_centroids, evaluate_model
from method.utils import *
from tqdm import tqdm


class VoronoiUnlearning:
    def __init__(self, model_path, forget_classes, retain_classes, device="cuda", 
                 use_regularization=False, lambda_reg=1e-3):
        self.device, self.forget_classes, self.retain_classes = device, forget_classes, retain_classes
        self.use_regularization, self.lambda_reg = use_regularization, lambda_reg
        
        num_classes = len(forget_classes) + len(retain_classes)
        self.model = get_model(num_classes, model_path=model_path, device=device)
        self.target_assignment = self.retain_centroids = self.forget_centroids = None
    
    def setup_targets(self, dataloader):
        """Setup target assignment using classifier weights"""
        import gc
        print("Setting up targets...")
        
        embeddings, labels = extract_embeddings(self.model, dataloader, self.device)
        self.retain_centroids = compute_class_centroids(embeddings, labels, self.retain_classes)
        self.forget_centroids = compute_class_centroids(embeddings, labels, self.forget_classes)
        del embeddings, labels; gc.collect()
        
        # Step 1: Generate targets along forget→retain paths
        target_vertices = generate_directional_targets(self.forget_centroids, self.retain_centroids, 
                                                     len(self.forget_classes) * 4)  # 4 targets per forget class
        print(f"Generated {len(target_vertices)} directional targets along forget→retain paths")
        
        # Step 2: Assign targets to classes
        self.target_assignment = assign_targets_to_classes(target_vertices, self.forget_classes, self.forget_centroids)
        
        # Step 3: Then orthogonalize assigned targets
        forget_weights = extract_classifier_weights(self.model, self.forget_classes)
        forget_weight_vectors = torch.stack(list(forget_weights.values()))
        
        # Orthogonalize the assigned targets
        for class_id in self.target_assignment:
            assigned_target = self.target_assignment[class_id].unsqueeze(0)  # [1, embed_dim]
            if assigned_target.device != forget_weight_vectors.device:
                forget_weight_vectors = forget_weight_vectors.to(assigned_target.device)
            orthogonalized_target = orthogonalize_embeddings_to_weights(assigned_target, forget_weight_vectors)
            self.target_assignment[class_id] = orthogonalized_target.squeeze(0)
        
        
        print(f"Setup complete: {len(self.target_assignment)} targets assigned")
        del target_vertices, forget_weights, forget_weight_vectors; gc.collect()
    
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
            total_loss = loss_retain_mse + loss_retain_ce * 0.4
            components = {'retain_mse': loss_retain_mse.item(), 'retain_ce': loss_retain_ce.item()}
        
        if self.use_regularization and global_step > 1000:
            reg_loss = compute_group_sparse_regularization(self.model, self.lambda_reg)
            total_loss += reg_loss
            components['reg'] = reg_loss.item()
        
        return total_loss, components
    
    

    def unlearn(self, data_dir, batch_size=32, epochs=5, lr=1e-4):
        """Main unlearning loop"""
        # Setup data loaders
        forget_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=(0, 44), data_ratio=1.0)
        retain_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=(45, 89), data_ratio=0.1)
        combined_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=(0, 89), data_ratio=0.1)
        
        self.setup_targets(combined_loader)
        
        # Initial evaluation
        print("\n=== Initial Evaluation ===")
        forget_acc, retain_acc = evaluate_model(self.model, data_dir, batch_size//2, self.forget_classes, self.retain_classes, self.device)
        print(f"Forget Acc: {forget_acc:.2f}% | Retain Acc: {retain_acc:.2f}%")
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        max_steps = max(len(forget_loader), len(retain_loader)) * epochs
        forget_iter, retain_iter = iter(forget_loader), iter(retain_loader)
        losses = {'forget': 0.0, 'constraint': 0.0, 'retain_mse': 0.0, 'retain_ce': 0.0, 'reg': 0.0, 'count': 0}
        
        # Timing variables
        step_start_time = time.time()
        total_start_time = time.time()
        total_training_time = 0.0  # Track pure training time excluding evaluations
        
        # Training loop
        pbar = tqdm(range(max_steps), desc="Training")
        for step in range(1, max_steps + 1):
            training_step_start = time.time()
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
            
            # Add this step's training time (excluding evaluation)
            total_training_time += time.time() - training_step_start
            
            # Progress and validation
            if losses['count'] > 0:
                avg_losses = {k: f"{v/losses['count']:.4f}" for k, v in losses.items() if k != 'count'}
                pbar.set_postfix(avg_losses)
            
            if step % 25 == 0:
                step_time = time.time() - step_start_time
                eval_start = time.time()
                f_acc, r_acc = evaluate_model(self.model, data_dir, batch_size//2, self.forget_classes, self.retain_classes, self.device)
                eval_time = time.time() - eval_start
                print(f"\nStep {step} - Forget: {f_acc:.2f}% | Retain: {r_acc:.2f}% | Step Time: {step_time:.2f}s | Training Time So Far: {total_training_time:.1f}s")
                losses = {'forget': 0.0, 'constraint': 0.0, 'retain_mse': 0.0, 'retain_ce': 0.0, 'reg': 0.0, 'count': 0}
                step_start_time = time.time()
            
            pbar.update(1)
        
        pbar.close()
        
        total_time = time.time() - total_start_time
        print(f"\n=== Training Complete ===")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Pure training time (excluding evaluations): {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
        print(f"Evaluation overhead: {total_time - total_training_time:.2f}s")
        
        return self.model
    


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use-reg', choices=['yes', 'no'], default='no')
    parser.add_argument('--lambda-reg', type=float, default=0.01)
    args = parser.parse_args()
    
    DATA_DIR, MODEL_PATH = "/media/jag/volD2/cifer100/cifer", "checkpoints/best.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    forget_classes, retain_classes = list(range(0, 44)), list(range(45, 89))
    
    print(f"=== Voronoi Unlearning ===")
    vu = VoronoiUnlearning(MODEL_PATH, forget_classes, retain_classes, DEVICE, 
                          use_regularization=args.use_reg=='yes', lambda_reg=args.lambda_reg)
    
    model = vu.unlearn(DATA_DIR, args.batch_size, args.epochs, args.lr)
    torch.save({'model_state': model.state_dict(), 'forget_classes': forget_classes, 
               'retain_classes': retain_classes}, 'voronoi_unlearned.pth')
    print(f"Model saved to 'voronoi_unlearned.pth'")


if __name__ == "__main__":
    main()