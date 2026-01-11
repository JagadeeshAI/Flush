import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model
import sys
import os
import time
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from codes.data import get_dataloaders


def get_frozen_model(num_classes=100, model_path=None, device="cpu"):
    ckpt_num_classes = 100
    ckpt_lora_r = 16
    print(f"Creating frozen DeiT-tiny model with LoRA (r={ckpt_lora_r}, classes={ckpt_num_classes})")
    
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
    
    # Freeze ALL parameters - model stays frozen during forget vector optimization
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters (all frozen): {total:,}")
    
    return model.to(device)


def compute_h_factor(acc_r, acc_f_before, acc_f_after):
    drop = max(acc_f_before - acc_f_after, 0)
    if acc_r + drop == 0:
        return 0.0
    return 2 * acc_r * drop / (acc_r + drop)


def cw_margin_loss(logits, labels, tau=1.0):
    """
    C&W-style margin loss for unlearning.
    L_f = max(f_y(x+δ) - max_{k≠y}(f_k(x+δ)) + τ, 0)
    Minimizing this pushes incorrect class above correct class.
    """
    batch_size = logits.size(0)
    num_classes = logits.size(1)
    
    # Get logit of correct class
    correct_logits = logits[torch.arange(batch_size), labels]
    
    # Get max logit of incorrect classes
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(batch_size), labels] = False
    incorrect_logits = logits.masked_fill(~mask, float('-inf'))
    max_incorrect_logits = incorrect_logits.max(dim=1).values
    
    # Margin loss: want max_incorrect > correct by at least tau margin
    margin = correct_logits - max_incorrect_logits + tau
    loss = F.relu(margin).mean()
    
    return loss


@torch.no_grad()
def evaluate_with_perturbation(model, delta, data_dir, batch_size, class_range, device, desc="Eval"):
    _, val_loader, _ = get_dataloaders(data_dir, batch_size, class_range=class_range, data_ratio=1.0)
    
    model.eval()
    correct, total = 0, 0
    
    for imgs, labels in tqdm(val_loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        # Apply forget vector
        perturbed_imgs = imgs + delta
        perturbed_imgs = torch.clamp(perturbed_imgs, 0, 1)
        
        preds = model(perturbed_imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_original(model, data_dir, batch_size, class_range, device, desc="Eval"):
    _, val_loader, _ = get_dataloaders(data_dir, batch_size, class_range=class_range, data_ratio=1.0)
    
    model.eval()
    correct, total = 0, 0
    
    for imgs, labels in tqdm(val_loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0


class ForgetVectorUnlearning:
    
    def __init__(self, model_path, forget_classes, retain_classes, device="cuda",
                 tau=1.0, lambda1=3.0, lambda2=1.0):
        self.device = device
        self.forget_classes = forget_classes
        self.retain_classes = retain_classes
        self.tau = tau
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        num_classes = len(forget_classes) + len(retain_classes)
        self.model = get_frozen_model(num_classes, model_path, device)
        
        # Initialize forget vector δ (same shape as input: 3×224×224)
        self.delta = torch.zeros(1, 3, 224, 224, device=device, requires_grad=True)
        
        print(f"\nForget Vector Config:")
        print(f"  τ (margin): {tau}")
        print(f"  λ1 (retain weight): {lambda1}")
        print(f"  λ2 (L2 regularization): {lambda2}")
        print(f"  Delta shape: {self.delta.shape}")
    
    def forget_loss(self, imgs, labels):
        perturbed = torch.clamp(imgs + self.delta, 0, 1)
        logits = self.model(perturbed)
        return cw_margin_loss(logits, labels, self.tau)
    
    def retain_loss(self, imgs, labels):
        perturbed = torch.clamp(imgs + self.delta, 0, 1)
        logits = self.model(perturbed)
        return F.cross_entropy(logits, labels)
    
    def l2_regularization(self):
        return torch.norm(self.delta, p=2) ** 2
    
    def unlearn(self, data_dir, batch_size=256, max_steps=40, lr=0.1, momentum=0.9, eval_every=10):
        
        forget_range = (min(self.forget_classes), max(self.forget_classes))
        retain_range = (min(self.retain_classes), max(self.retain_classes))
        
        # Data loaders
        forget_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=forget_range, data_ratio=1.0)
        retain_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=retain_range, data_ratio=0.3)
        
        # Initial evaluation (without perturbation)
        print("\n=== Initial Evaluation (no perturbation) ===")
        init_acc_f = evaluate_original(self.model, data_dir, batch_size//2, forget_range, self.device, "Forget")
        init_acc_r = evaluate_original(self.model, data_dir, batch_size//2, retain_range, self.device, "Retain")
        print(f"Forget Acc: {init_acc_f:.2f}% | Retain Acc: {init_acc_r:.2f}%")
        
        # Optimizer for delta (SGD with momentum as per paper)
        optimizer = torch.optim.SGD([self.delta], lr=lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        
        total_train_time = 0.0
        
        print(f"\n=== Optimizing Forget Vector ({max_steps} steps) ===")
        pbar = tqdm(range(1, max_steps + 1), desc="Training")
        
        for step in pbar:
            train_start = time.time()
            
            # Get batches
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_batch = next(forget_iter)
            
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            
            f_imgs, f_labels = forget_batch
            r_imgs, r_labels = retain_batch
            f_imgs, f_labels = f_imgs.to(self.device), f_labels.to(self.device)
            r_imgs, r_labels = r_imgs.to(self.device), r_labels.to(self.device)
            
            # Compute losses
            l_forget = self.forget_loss(f_imgs, f_labels)
            l_retain = self.retain_loss(r_imgs, r_labels)
            l_reg = self.l2_regularization()
            
            # Total loss: L_f + λ1*L_r + λ2*||δ||²
            total_loss = l_forget + self.lambda1 * l_retain + self.lambda2 * l_reg
            
            # Optimize delta
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_train_time += time.time() - train_start
            
            # Progress bar
            pbar.set_postfix({
                'L_f': f"{l_forget.item():.3f}",
                'L_r': f"{l_retain.item():.3f}",
                '||δ||': f"{torch.norm(self.delta).item():.3f}"
            })
            
            # Periodic evaluation
            if step % eval_every == 0:
                scheduler.step()
                
                acc_f = evaluate_with_perturbation(self.model, self.delta, data_dir, batch_size//2, 
                                                   forget_range, self.device, "Forget")
                acc_r = evaluate_with_perturbation(self.model, self.delta, data_dir, batch_size//2, 
                                                   retain_range, self.device, "Retain")
                h_factor = compute_h_factor(acc_r, init_acc_f, acc_f)
                
                print(f"\n[Step {step}] Acc_f: {acc_f:.2f}% | Acc_r: {acc_r:.2f}% | "
                      f"H-factor: {h_factor:.2f} | ||δ||: {torch.norm(self.delta).item():.3f} | "
                      f"Train time: {total_train_time:.1f}s")
        
        pbar.close()
        
        # Final evaluation
        print(f"\n=== Final Evaluation (with forget vector) ===")
        final_acc_f = evaluate_with_perturbation(self.model, self.delta, data_dir, batch_size//2, 
                                                  forget_range, self.device, "Forget")
        final_acc_r = evaluate_with_perturbation(self.model, self.delta, data_dir, batch_size//2, 
                                                  retain_range, self.device, "Retain")
        final_h = compute_h_factor(final_acc_r, init_acc_f, final_acc_f)
        
        print(f"Initial -> Final:")
        print(f"  Acc_f: {init_acc_f:.2f}% -> {final_acc_f:.2f}% (drop: {init_acc_f - final_acc_f:.2f}%)")
        print(f"  Acc_r: {init_acc_r:.2f}% -> {final_acc_r:.2f}%")
        print(f"  H-factor: {final_h:.2f}")
        print(f"  ||δ||_2: {torch.norm(self.delta).item():.4f}")
        print(f"  Total training time: {total_train_time:.2f}s ({total_train_time/60:.2f} min)")
        
        return self.model, self.delta


def main():
    parser = argparse.ArgumentParser(description="Forget Vector Unlearning")
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=1.0, help="Margin for C&W loss")
    parser.add_argument('--lambda1', type=float, default=0.5, help="Retain loss weight")
    parser.add_argument('--lambda2', type=float, default=1.0, help="L2 regularization weight")
    parser.add_argument('--eval_every', type=int, default=10)
    args = parser.parse_args()
    
    DATA_DIR = "/media/jag/volD2/cifer100/cifer"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../checkpoints/best.pth')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    forget_classes = list(range(0, 45))
    retain_classes = list(range(45, 90))
    
    print("=" * 60)
    print("Forget Vector: Input Perturbation-based Unlearning")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Forget classes: {min(forget_classes)}-{max(forget_classes)} ({len(forget_classes)} classes)")
    print(f"Retain classes: {min(retain_classes)}-{max(retain_classes)} ({len(retain_classes)} classes)")
    
    unlearner = ForgetVectorUnlearning(
        model_path=MODEL_PATH,
        forget_classes=forget_classes,
        retain_classes=retain_classes,
        device=DEVICE,
        tau=args.tau,
        lambda1=args.lambda1,
        lambda2=args.lambda2
    )
    
    model, delta = unlearner.unlearn(
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        momentum=args.momentum,
        eval_every=args.eval_every
    )
    
    save_path = os.path.join(os.path.dirname(__file__), 'forget_vector_unlearned.pth')
    torch.save({
        'model_state': model.state_dict(),
        'forget_vector': delta.detach().cpu(),
        'forget_classes': forget_classes,
        'retain_classes': retain_classes,
        'config': vars(args)
    }, save_path)
    print(f"\nModel and forget vector saved to: {save_path}")


if __name__ == "__main__":
    main()
