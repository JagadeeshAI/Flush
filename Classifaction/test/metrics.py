import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from codes.data import get_dataloaders
from codes.utils import get_model
from tqdm import tqdm


def compute_kl_divergence(model1, model2, dataloader, device="cuda"):
    """Compute KL divergence between two models on given data"""
    model1.eval()
    model2.eval()
    
    total_kl = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Computing KL divergence"):
            imgs = imgs.to(device)
            
            # Get logits from both models
            logits1 = model1(imgs)
            logits2 = model2(imgs)
            
            # Convert to probabilities
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            
            # Compute KL divergence: KL(P1 || P2)
            kl_div = F.kl_div(F.log_softmax(logits2, dim=1), probs1, reduction='none').sum(dim=1)
            
            total_kl += kl_div.sum().item()
            total_samples += imgs.size(0)
    
    return total_kl / total_samples


class MIAAttacker:
    """Membership Inference Attack implementation with label-only access"""
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()
    
    def gap_attack(self, data_loader, num_perturbations=10, noise_std=0.1):
        """Gap Attack: Test label consistency under perturbations"""
        results = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(data_loader, desc="Gap Attack"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                # Get original predictions
                original_preds = self.model(imgs).argmax(dim=1)
                
                # Count consistent predictions under perturbations
                for i in range(imgs.size(0)):
                    img = imgs[i:i+1]
                    orig_pred = original_preds[i]
                    consistent_count = 0
                    
                    for _ in range(num_perturbations):
                        # Add random noise
                        noise = torch.randn_like(img) * noise_std
                        perturbed_img = torch.clamp(img + noise, 0, 1)
                        
                        # Check if prediction remains same
                        perturbed_pred = self.model(perturbed_img).argmax(dim=1)
                        if perturbed_pred == orig_pred:
                            consistent_count += 1
                    
                    consistency_rate = consistent_count / num_perturbations
                    results.append({
                        'consistency_rate': consistency_rate,
                        'original_pred': orig_pred.item(),
                        'true_label': labels[i].item()
                    })
        
        return results
    
    def boundary_distance_attack(self, data_loader, max_iterations=50, step_size=0.01):
        """Boundary Distance Attack: Measure distance to decision boundary"""
        results = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(data_loader, desc="Boundary Distance Attack"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                for i in range(imgs.size(0)):
                    img = imgs[i:i+1].clone()
                    original_pred = self.model(img).argmax(dim=1)
                    
                    # Find minimum perturbation to flip label
                    perturbation_magnitude = 0.0
                    current_img = img.clone()
                    
                    for iteration in range(max_iterations):
                        # Add random perturbation
                        noise = torch.randn_like(img) * step_size
                        current_img = torch.clamp(img + noise * (iteration + 1), 0, 1)
                        
                        # Check if label flipped
                        current_pred = self.model(current_img).argmax(dim=1)
                        if current_pred != original_pred:
                            perturbation_magnitude = torch.norm(current_img - img).item()
                            break
                    else:
                        # If no flip found, use maximum perturbation
                        perturbation_magnitude = torch.norm(current_img - img).item()
                    
                    results.append({
                        'boundary_distance': perturbation_magnitude,
                        'original_pred': original_pred.item(),
                        'true_label': labels[i].item()
                    })
        
        return results
    
    def augmentation_attack(self, data_loader, num_augmentations=5):
        """Data Augmentation Attack: Test consistency across augmentations"""
        results = []
        
        def augment_image(img):
            """Simple augmentation: random rotation and noise"""
            # Random rotation (small angle)
            angle = (torch.rand(1) - 0.5) * 20  # ±10 degrees
            
            # Random noise
            noise = torch.randn_like(img) * 0.05
            
            # Apply augmentation (simplified)
            augmented = torch.clamp(img + noise, 0, 1)
            return augmented
        
        with torch.no_grad():
            for imgs, labels in tqdm(data_loader, desc="Augmentation Attack"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                # Get original predictions
                original_preds = self.model(imgs).argmax(dim=1)
                
                for i in range(imgs.size(0)):
                    img = imgs[i:i+1]
                    orig_pred = original_preds[i]
                    consistent_count = 0
                    
                    for _ in range(num_augmentations):
                        # Apply augmentation
                        aug_img = augment_image(img)
                        
                        # Check prediction consistency
                        aug_pred = self.model(aug_img).argmax(dim=1)
                        if aug_pred == orig_pred:
                            consistent_count += 1
                    
                    consistency_rate = consistent_count / num_augmentations
                    results.append({
                        'aug_consistency': consistency_rate,
                        'original_pred': orig_pred.item(),
                        'true_label': labels[i].item()
                    })
        
        return results


def evaluate_mia(model, data_dir, batch_size=32, device="cuda"):
    """Evaluate MIA attacks on member vs non-member data"""
    
    # Member data (forget range): classes 40-44
    _, member_loader, _ = get_dataloaders(data_dir, batch_size, class_range=(40, 44), data_ratio=1.0)
    
    # Non-member data: classes 90-94  
    _, non_member_loader, _ = get_dataloaders(data_dir, batch_size, class_range=(90, 94), data_ratio=1.0)
    
    print("Setting up MIA attacker...")
    attacker = MIAAttacker(model, device)
    
    print("\n=== Gap Attack ===")
    member_gap_results = attacker.gap_attack(member_loader)
    non_member_gap_results = attacker.gap_attack(non_member_loader)
    
    member_gap_scores = [r['consistency_rate'] for r in member_gap_results]
    non_member_gap_scores = [r['consistency_rate'] for r in non_member_gap_results]
    
    print(f"Member consistency: {np.mean(member_gap_scores):.4f} ± {np.std(member_gap_scores):.4f}")
    print(f"Non-member consistency: {np.mean(non_member_gap_scores):.4f} ± {np.std(non_member_gap_scores):.4f}")
    
    print("\n=== Boundary Distance Attack ===")
    member_boundary_results = attacker.boundary_distance_attack(member_loader)
    non_member_boundary_results = attacker.boundary_distance_attack(non_member_loader)
    
    member_boundary_scores = [r['boundary_distance'] for r in member_boundary_results]
    non_member_boundary_scores = [r['boundary_distance'] for r in non_member_boundary_results]
    
    print(f"Member boundary distance: {np.mean(member_boundary_scores):.4f} ± {np.std(member_boundary_scores):.4f}")
    print(f"Non-member boundary distance: {np.mean(non_member_boundary_scores):.4f} ± {np.std(non_member_boundary_scores):.4f}")
    
    print("\n=== Augmentation Attack ===")
    member_aug_results = attacker.augmentation_attack(member_loader)
    non_member_aug_results = attacker.augmentation_attack(non_member_loader)
    
    member_aug_scores = [r['aug_consistency'] for r in member_aug_results]
    non_member_aug_scores = [r['aug_consistency'] for r in non_member_aug_results]
    
    print(f"Member augmentation consistency: {np.mean(member_aug_scores):.4f} ± {np.std(member_aug_scores):.4f}")
    print(f"Non-member augmentation consistency: {np.mean(non_member_aug_scores):.4f} ± {np.std(non_member_aug_scores):.4f}")
    
    # Compute attack accuracy for each method
    def compute_attack_accuracy(member_scores, non_member_scores, higher_is_member=True):
        """Compute MIA attack accuracy using threshold"""
        all_scores = member_scores + non_member_scores
        all_labels = [1] * len(member_scores) + [0] * len(non_member_scores)  # 1=member, 0=non-member
        
        threshold = np.median(all_scores)
        
        if higher_is_member:
            predictions = [1 if score >= threshold else 0 for score in all_scores]
        else:
            predictions = [0 if score >= threshold else 1 for score in all_scores]
        
        accuracy = np.mean([pred == label for pred, label in zip(predictions, all_labels)])
        return accuracy, threshold
    
    gap_acc, gap_thresh = compute_attack_accuracy(member_gap_scores, non_member_gap_scores, higher_is_member=True)
    boundary_acc, boundary_thresh = compute_attack_accuracy(member_boundary_scores, non_member_boundary_scores, higher_is_member=True)
    aug_acc, aug_thresh = compute_attack_accuracy(member_aug_scores, non_member_aug_scores, higher_is_member=True)
    
    print(f"\n=== MIA Attack Accuracies ===")
    print(f"Gap Attack: {gap_acc:.4f} (threshold: {gap_thresh:.4f})")
    print(f"Boundary Attack: {boundary_acc:.4f} (threshold: {boundary_thresh:.4f})")
    print(f"Augmentation Attack: {aug_acc:.4f} (threshold: {aug_thresh:.4f})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', default="/media/jag/volD2/cifer100/cifer")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=== Model Comparison: KL Divergence Analysis ===")
    
    # Load models
    print("Loading Voronoi Unlearned model...")
    voronoi_model = get_model(100, model_path="voronoi_unlearned.pth", device=device)
    
    print("Loading Ideal (reference) model...")
    ideal_model = get_model(100, model_path="checkpoints/ideal.pth", device=device)
    
    # Get test data for KL computation
    _, test_loader, _ = get_dataloaders(args.data_dir, args.batch_size, class_range=(0, 99), data_ratio=0.1)
    
    print("\n=== KL Divergence Computation ===")
    
    # KL(Voronoi || Ideal)
    print("Computing KL divergence: KL(Voronoi || Ideal)...")
    kl_voronoi_to_ideal = compute_kl_divergence(voronoi_model, ideal_model, test_loader, device)
    print(f"KL(Voronoi || Ideal): {kl_voronoi_to_ideal:.6f}")
    
    # KL(Ideal || Voronoi)  
    print("Computing KL divergence: KL(Ideal || Voronoi)...")
    kl_ideal_to_voronoi = compute_kl_divergence(ideal_model, voronoi_model, test_loader, device)
    print(f"KL(Ideal || Voronoi): {kl_ideal_to_voronoi:.6f}")
    
    print(f"\nSymmetric KL divergence: {(kl_voronoi_to_ideal + kl_ideal_to_voronoi) / 2:.6f}")
    
    print("\n=== Membership Inference Attack Evaluation ===")
    print("Testing MIA on Voronoi Unlearned model...")
    evaluate_mia(voronoi_model, args.data_dir, args.batch_size, device)


if __name__ == "__main__":
    main()