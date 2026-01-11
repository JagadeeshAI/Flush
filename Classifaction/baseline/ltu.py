"""
LTU: Learning to Unlearn for Robust Machine Unlearning
Based on: "Learning to Unlearn for Robust Machine Unlearning" (arXiv:2407.10494)

Key components:
- Meta optimization scheme: meta-tune, meta-test, meta-update
- Support/Query set construction for remembering and forgetting feedback
- Gradient Harmonization to resolve gradient conflicts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model
import sys
import os
import time
import argparse
import copy
from tqdm import tqdm
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from codes.data import get_dataloaders


def get_ltu_model(num_classes=100, model_path=None, device="cpu"):
    """
    Load DeiT-tiny with LoRA for LTU unlearning.
    All parameters trainable for meta-learning.
    """
    ckpt_num_classes = 100
    ckpt_lora_r = 16
    print(f"Creating DeiT-tiny model with LoRA (r={ckpt_lora_r}, classes={ckpt_num_classes})")
    
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=True,
        num_classes=ckpt_num_classes
    )
    
    lora_config = LoraConfig(
        r=ckpt_lora_r,
        lora_alpha=ckpt_lora_r * 2,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    if model_path is not None and os.path.exists(model_path):
        print(f"Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    
    # Freeze all except LoRA
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model.to(device)


def compute_h_factor(acc_r, acc_f_before, acc_f_after):
    """H-Mean: harmonic mean of retain accuracy and forget drop"""
    drop = max(acc_f_before - acc_f_after, 0)
    if acc_r + drop == 0:
        return 0.0
    return 2 * acc_r * drop / (acc_r + drop)


@torch.no_grad()
def evaluate(model, data_dir, batch_size, class_range, device, desc="Eval"):
    """Evaluate model accuracy"""
    _, val_loader, _ = get_dataloaders(data_dir, batch_size, class_range=class_range, data_ratio=1.0)
    model.eval()
    correct, total = 0, 0
    for imgs, labels in tqdm(val_loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def gradient_harmonization(g_r, g_f):
    """
    Harmonize forget gradient w.r.t. remember gradient.
    If cos(g_r, g_f) < 0: project g_f onto subspace orthogonal to g_r
    """
    # Flatten to 1D
    g_r_flat = torch.cat([g.view(-1) for g in g_r])
    g_f_flat = torch.cat([g.view(-1) for g in g_f])
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(g_r_flat.unsqueeze(0), g_f_flat.unsqueeze(0))
    
    if cos_sim < 0:
        # Project g_f onto orthogonal subspace of g_r
        # g_f' = g_f - (g_f · g_r / ||g_r||²) * g_r
        proj_coeff = (g_f_flat @ g_r_flat) / (g_r_flat @ g_r_flat + 1e-8)
        g_f_flat_harmonized = g_f_flat - proj_coeff * g_r_flat
    else:
        g_f_flat_harmonized = g_f_flat
    
    # Reshape back
    g_f_harmonized = []
    idx = 0
    for g in g_f:
        numel = g.numel()
        g_f_harmonized.append(g_f_flat_harmonized[idx:idx+numel].view(g.shape))
        idx += numel
    
    return g_f_harmonized, cos_sim.item()


class LTUUnlearning:
    """
    LTU: Learning to Unlearn framework with:
    - Meta optimization (meta-tune, meta-test, meta-update)
    - Gradient harmonization
    """
    
    def __init__(self, model_path, forget_classes, retain_classes, device="cuda",
                 meta_lr=1e-4, rho=0.3, lambda_meta=5.0, forget_weight=0.3, retain_weight=1.0):
        self.device = device
        self.forget_classes = forget_classes
        self.retain_classes = retain_classes
        self.meta_lr = meta_lr
        self.rho = rho  # Fraction of retain set to use
        self.lambda_meta = lambda_meta
        self.forget_weight = forget_weight  # Scale down forget gradient
        self.retain_weight = retain_weight  # Explicit retain loss weight
        
        num_classes = len(forget_classes) + len(retain_classes)
        self.model = get_ltu_model(num_classes, model_path, device)
        
        print(f"\nLTU Config:")
        print(f"  Meta LR (α): {meta_lr}")
        print(f"  Retain subset ratio (ρ): {rho}")
        print(f"  Lambda (meta-test weight): {lambda_meta}")
        print(f"  Forget weight: {forget_weight}")
        print(f"  Retain weight: {retain_weight}")
    
    def create_support_set(self, forget_batch, num_classes=100):
        """
        Create support set: forget samples with random labels
        This simulates the forgetting objective
        """
        imgs, _ = forget_batch
        random_labels = torch.randint(0, num_classes, (imgs.size(0),))
        return imgs, random_labels
    
    def meta_tune(self, model, support_set, lr):
        """
        Meta-tune: temporarily update model on support set
        Returns: temporarily updated model θ^τ
        """
        model_copy = copy.deepcopy(model)
        model_copy.train()
        
        imgs, labels = support_set
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        
        logits = model_copy(imgs)
        loss_tune = F.cross_entropy(logits, labels)
        
        # Compute gradients and apply temporary update
        grads = torch.autograd.grad(loss_tune, 
                                    [p for p in model_copy.parameters() if p.requires_grad],
                                    create_graph=True)
        
        # Apply update to get θ^τ
        for param, grad in zip([p for p in model_copy.parameters() if p.requires_grad], grads):
            param.data = param.data - lr * grad
        
        return model_copy, loss_tune
    
    def meta_test(self, model_tau, query_sets):
        """
        Meta-test: evaluate θ^τ on query sets (retain samples)
        Returns: aggregated test loss
        """
        model_tau.eval()
        total_loss = 0.0
        
        for query_batch in query_sets:
            imgs, labels = query_batch
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = model_tau(imgs)
            total_loss += F.cross_entropy(logits, labels)
        
        return total_loss / len(query_sets) if query_sets else torch.tensor(0.0)
    
    def compute_remember_gradient(self, forget_batch, retain_batches):
        """
        Compute gradient for remembering feedback via meta-optimization
        """
        # Support: forget samples with random labels
        support = self.create_support_set(forget_batch)
        
        # Meta-tune
        model_tau, loss_tune = self.meta_tune(self.model, support, self.meta_lr)
        
        # Meta-test on retain query sets
        loss_test = self.meta_test(model_tau, retain_batches)
        
        # Combined loss for remembering
        loss_remember = loss_tune + self.lambda_meta * loss_test
        
        # Compute gradients w.r.t. original model
        self.model.zero_grad()
        grads = torch.autograd.grad(loss_remember, 
                                    [p for p in self.model.parameters() if p.requires_grad],
                                    allow_unused=True)
        
        # Replace None gradients with zeros
        grads = [g if g is not None else torch.zeros_like(p) 
                 for g, p in zip(grads, [p for p in self.model.parameters() if p.requires_grad])]
        
        return grads, loss_tune.item(), loss_test.item()
    
    def compute_forget_gradient(self, forget_batch):
        """
        Compute gradient for forgetting feedback
        Uses negative cross-entropy to maximize loss on forget samples
        """
        self.model.train()
        imgs, labels = forget_batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        
        logits = self.model(imgs)
        # Maximize CE loss = minimize negative CE
        loss_forget = -F.cross_entropy(logits, labels)
        
        self.model.zero_grad()
        grads = torch.autograd.grad(loss_forget,
                                    [p for p in self.model.parameters() if p.requires_grad])
        
        return list(grads), -loss_forget.item()
    
    def unlearn(self, data_dir, batch_size=32, max_steps=500, lr=5e-4, eval_every=50):
        """Main LTU unlearning loop"""
        
        forget_range = (min(self.forget_classes), max(self.forget_classes))
        retain_range = (min(self.retain_classes), max(self.retain_classes))
        
        # Data loaders
        forget_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=forget_range, data_ratio=1.0)
        retain_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=retain_range, data_ratio=self.rho)
        
        # Initial evaluation
        print("\n=== Initial Evaluation ===")
        init_acc_f = evaluate(self.model, data_dir, batch_size, forget_range, self.device, "Forget")
        init_acc_r = evaluate(self.model, data_dir, batch_size, retain_range, self.device, "Retain")
        print(f"Forget Acc: {init_acc_f:.2f}% | Retain Acc: {init_acc_r:.2f}%")
        
        # Optimizer
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        
        # Iterators
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        
        total_train_time = 0.0
        
        print(f"\n=== Training ({max_steps} steps) ===")
        pbar = tqdm(range(1, max_steps + 1), desc="Training")
        
        for step in pbar:
            train_start = time.time()
            self.model.train()
            
            # Get forget batch
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_batch = next(forget_iter)
            
            # Get multiple retain batches for query sets
            retain_batches = []
            for _ in range(3):  # 3 query sets with diverse distributions
                try:
                    retain_batches.append(next(retain_iter))
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batches.append(next(retain_iter))
            
            # Compute remember gradient (g_r) via meta-optimization
            g_r, l_tune, l_test = self.compute_remember_gradient(forget_batch, retain_batches)
            
            # Compute forget gradient (g_f)
            g_f, l_forget = self.compute_forget_gradient(forget_batch)
            
            # Compute explicit retain gradient to prevent catastrophic forgetting
            self.model.train()
            retain_imgs, retain_labels = retain_batches[0]
            retain_imgs, retain_labels = retain_imgs.to(self.device), retain_labels.to(self.device)
            retain_logits = self.model(retain_imgs)
            loss_retain_explicit = F.cross_entropy(retain_logits, retain_labels)
            
            self.model.zero_grad()
            g_retain = torch.autograd.grad(loss_retain_explicit,
                                           [p for p in self.model.parameters() if p.requires_grad])
            g_retain = list(g_retain)
            l_retain_explicit = loss_retain_explicit.item()
            
            # Gradient Harmonization
            g_f_harmonized, cos_sim = gradient_harmonization(g_r, g_f)
            
            # Apply harmonized gradients: G = retain_weight*g_retain + g_r + forget_weight*g_f'
            optimizer.zero_grad()
            for param, gr, gf, g_ret in zip([p for p in self.model.parameters() if p.requires_grad], 
                                             g_r, g_f_harmonized, g_retain):
                # Prioritize retain, then remember, then forget (scaled down)
                param.grad = self.retain_weight * g_ret + gr + self.forget_weight * gf
            optimizer.step()
            
            total_train_time += time.time() - train_start
            
            # Progress bar
            pbar.set_postfix({
                'L_ret': f"{l_retain_explicit:.3f}",
                'L_test': f"{l_test:.3f}",
                'L_f': f"{l_forget:.3f}",
                'cos': f"{cos_sim:.2f}"
            })
            
            # Periodic evaluation
            if step % eval_every == 0:
                acc_f = evaluate(self.model, data_dir, batch_size, forget_range, self.device, "Forget")
                acc_r = evaluate(self.model, data_dir, batch_size, retain_range, self.device, "Retain")
                h_factor = compute_h_factor(acc_r, init_acc_f, acc_f)
                
                print(f"\n[Step {step}] Acc_f: {acc_f:.2f}% | Acc_r: {acc_r:.2f}% | "
                      f"H-factor: {h_factor:.2f} | Train time: {total_train_time:.1f}s")
        
        pbar.close()
        
        # Final evaluation
        print(f"\n=== Final Evaluation ===")
        final_acc_f = evaluate(self.model, data_dir, batch_size, forget_range, self.device, "Forget")
        final_acc_r = evaluate(self.model, data_dir, batch_size, retain_range, self.device, "Retain")
        final_h = compute_h_factor(final_acc_r, init_acc_f, final_acc_f)
        
        print(f"Initial -> Final:")
        print(f"  Acc_f: {init_acc_f:.2f}% -> {final_acc_f:.2f}% (drop: {init_acc_f - final_acc_f:.2f}%)")
        print(f"  Acc_r: {init_acc_r:.2f}% -> {final_acc_r:.2f}%")
        print(f"  H-factor: {final_h:.2f}")
        print(f"  Total training time: {total_train_time:.2f}s ({total_train_time/60:.2f} min)")
        
        return self.model


def main():
    parser = argparse.ArgumentParser(description="LTU: Learning to Unlearn")
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)  # Reduced from 5e-4
    parser.add_argument('--meta_lr', type=float, default=1e-4, help="Learning rate for meta-tune")
    parser.add_argument('--rho', type=float, default=0.5, help="Retain subset ratio")
    parser.add_argument('--lambda_meta', type=float, default=10.0, help="Weight for meta-test loss")
    parser.add_argument('--forget_weight', type=float, default=0.2, help="Scale for forget gradient")
    parser.add_argument('--retain_weight', type=float, default=2.0, help="Weight for explicit retain loss")
    parser.add_argument('--eval_every', type=int, default=50)
    args = parser.parse_args()
    
    DATA_DIR = "/media/jag/volD2/cifer100/cifer"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../checkpoints/best.pth')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    forget_classes = list(range(0, 45))
    retain_classes = list(range(45, 90))
    
    print("=" * 60)
    print("LTU: Learning to Unlearn for Robust Machine Unlearning")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Forget classes: {min(forget_classes)}-{max(forget_classes)} ({len(forget_classes)} classes)")
    print(f"Retain classes: {min(retain_classes)}-{max(retain_classes)} ({len(retain_classes)} classes)")
    
    unlearner = LTUUnlearning(
        model_path=MODEL_PATH,
        forget_classes=forget_classes,
        retain_classes=retain_classes,
        device=DEVICE,
        meta_lr=args.meta_lr,
        rho=args.rho,
        lambda_meta=args.lambda_meta,
        forget_weight=args.forget_weight,
        retain_weight=args.retain_weight
    )
    
    model = unlearner.unlearn(
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        eval_every=args.eval_every
    )
    
    save_path = os.path.join(os.path.dirname(__file__), 'ltu_unlearned.pth')
    torch.save({
        'model_state': model.state_dict(),
        'forget_classes': forget_classes,
        'retain_classes': retain_classes,
        'config': vars(args)
    }, save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
