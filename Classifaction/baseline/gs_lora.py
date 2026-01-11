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


def get_gs_lora_model(num_classes=100, model_path=None, device="cpu", lora_r=16):
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
    
    for param in model.parameters():
        param.requires_grad = False
    
    lora_params = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params += param.numel()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable LoRA parameters: {lora_params:,} ({100*lora_params/total_params:.2f}%)")
    
    return model.to(device)


def get_lora_groups(model):
    
    groups = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            # Extract block identifier (e.g., "blocks.0", "blocks.1", ...)
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "blocks" and i + 1 < len(parts):
                    block_id = f"block_{parts[i+1]}"
                    break
            else:
                block_id = "other"
            
            if block_id not in groups:
                groups[block_id] = []
            groups[block_id].append(param)
    
    return list(groups.items())


def group_sparsity_loss(lora_groups):
    
    total = 0.0
    for group_name, params in lora_groups:
        group_norm = sum(p.norm(p=2) ** 2 for p in params).sqrt()
        total += group_norm
    return total


def bounded_forget_loss(logits, labels, bnd=5.0):
    
    ce_loss = F.cross_entropy(logits, labels)
    return F.relu(-ce_loss + bnd)


def retain_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def compute_h_factor(acc_r, acc_f_before, acc_f_after):
    
    drop = acc_f_before - acc_f_after
    drop = max(drop, 0)
    
    if acc_r + drop == 0:
        return 0.0
    
    h = 2 * acc_r * drop / (acc_r + drop)
    return h


@torch.no_grad()
def evaluate(model, data_dir, batch_size, class_range, device, desc="Eval"):
    _, val_loader, _ = get_dataloaders(data_dir, batch_size, class_range=class_range, data_ratio=1.0)
    
    model.eval()
    correct = 0
    total = 0
    
    for imgs, labels in tqdm(val_loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0


class GSLoRAUnlearning:
    
    def __init__(self, model_path, forget_classes, retain_classes, device="cuda",
                 bnd=5.0, beta=1.0, alpha=0.01, warmup_steps=100, lora_r=8):
        self.device = device
        self.forget_classes = forget_classes
        self.retain_classes = retain_classes
        self.bnd = bnd
        self.beta = beta
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        
        num_classes = len(forget_classes) + len(retain_classes)
        self.model = get_gs_lora_model(num_classes, model_path, device, lora_r)
        self.lora_groups = get_lora_groups(self.model)
        
        print(f"\nGS-LoRA Config:")
        print(f"  BND (forget bound): {bnd}")
        print(f"  β (retain weight): {beta}")
        print(f"  α (sparsity weight): {alpha}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  LoRA groups: {len(self.lora_groups)}")
    
    def unlearn(self, data_dir, batch_size=32, max_steps=500, lr=1e-4, eval_every=50):
        
        # Data loaders
        forget_range = (min(self.forget_classes), max(self.forget_classes))
        retain_range = (min(self.retain_classes), max(self.retain_classes))
        
        forget_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=forget_range, data_ratio=1.0)
        retain_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=retain_range, data_ratio=0.1)
        
        # Initial evaluation
        print("\n=== Initial Evaluation ===")
        init_acc_f = evaluate(self.model, data_dir, batch_size, forget_range, self.device, "Forget")
        init_acc_r = evaluate(self.model, data_dir, batch_size, retain_range, self.device, "Retain")
        print(f"Forget Acc: {init_acc_f:.2f}% | Retain Acc: {init_acc_r:.2f}%")
        
        # Optimizer (only LoRA params)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params, lr=lr)
        
        # Iterators
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        
        # Tracking
        total_train_time = 0.0
        running_losses = {'forget': 0, 'retain': 0, 'sparsity': 0, 'total': 0, 'count': 0}
        
        print(f"\n=== Training ({max_steps} steps) ===")
        pbar = tqdm(range(1, max_steps + 1), desc="Training")
        
        for step in pbar:
            train_start = time.time()
            self.model.train()
            
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
            
            # Forward pass - Forget
            f_imgs, f_labels = forget_batch
            f_imgs, f_labels = f_imgs.to(self.device), f_labels.to(self.device)
            f_logits = self.model(f_imgs)
            l_forget = bounded_forget_loss(f_logits, f_labels, self.bnd)
            
            # Forward pass - Retain
            r_imgs, r_labels = retain_batch
            r_imgs, r_labels = r_imgs.to(self.device), r_labels.to(self.device)
            r_logits = self.model(r_imgs)
            l_retain = retain_loss(r_logits, r_labels)
            
            # Group sparsity (with warmup)
            current_alpha = 0.0 if step <= self.warmup_steps else self.alpha
            l_sparsity = group_sparsity_loss(self.lora_groups)
            
            # Combined loss
            l_total = l_forget + self.beta * l_retain + current_alpha * l_sparsity
            
            # Backward pass
            optimizer.zero_grad()
            l_total.backward()
            optimizer.step()
            
            # Track training time
            total_train_time += time.time() - train_start
            
            # Update running losses
            running_losses['forget'] += l_forget.item()
            running_losses['retain'] += l_retain.item()
            running_losses['sparsity'] += l_sparsity.item()
            running_losses['total'] += l_total.item()
            running_losses['count'] += 1
            
            # Progress bar update
            avg_losses = {k: running_losses[k] / running_losses['count'] 
                         for k in ['forget', 'retain', 'sparsity'] if running_losses['count'] > 0}
            pbar.set_postfix({
                'L_f': f"{avg_losses.get('forget', 0):.3f}",
                'L_r': f"{avg_losses.get('retain', 0):.3f}",
                'L_s': f"{avg_losses.get('sparsity', 0):.3f}",
                'α': current_alpha
            })
            
            # Periodic evaluation
            if step % eval_every == 0:
                acc_f = evaluate(self.model, data_dir, batch_size, forget_range, self.device, "Forget")
                acc_r = evaluate(self.model, data_dir, batch_size, retain_range, self.device, "Retain")
                h_factor = compute_h_factor(acc_r, init_acc_f, acc_f)
                
                print(f"\n[Step {step}] Acc_f: {acc_f:.2f}% | Acc_r: {acc_r:.2f}% | "
                      f"H-factor: {h_factor:.2f} | Train time: {total_train_time:.1f}s")
                
                # Reset running losses
                running_losses = {'forget': 0, 'retain': 0, 'sparsity': 0, 'total': 0, 'count': 0}
        
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
    parser = argparse.ArgumentParser(description="GS-LoRA Unlearning")
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--bnd', type=float, default=5.0, help="Bound for forget loss (higher = more aggressive)")
    parser.add_argument('--beta', type=float, default=0.5, help="Weight for retain loss")
    parser.add_argument('--alpha', type=float, default=0.01, help="Weight for group sparsity")
    parser.add_argument('--warmup_steps', type=int, default=50, help="Warmup steps before sparsity")
    parser.add_argument('--lora_r', type=int, default=16, help="LoRA rank")
    parser.add_argument('--eval_every', type=int, default=50, help="Evaluate every N steps")
    args = parser.parse_args()
    
    # Paths
    DATA_DIR = "/media/jag/volD2/cifer100/cifer"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../checkpoints/best.pth')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Class splits (from your existing setup)
    forget_classes = list(range(0, 45))      # Classes 0-44 to forget
    retain_classes = list(range(45, 90))     # Classes 45-89 to retain
    
    print("=" * 60)
    print("GS-LoRA: Group Sparse LoRA for Continual Forgetting")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Forget classes: {min(forget_classes)}-{max(forget_classes)} ({len(forget_classes)} classes)")
    print(f"Retain classes: {min(retain_classes)}-{max(retain_classes)} ({len(retain_classes)} classes)")
    
    # Initialize unlearner
    unlearner = GSLoRAUnlearning(
        model_path=MODEL_PATH,
        forget_classes=forget_classes,
        retain_classes=retain_classes,
        device=DEVICE,
        bnd=args.bnd,
        beta=args.beta,
        alpha=args.alpha,
        warmup_steps=args.warmup_steps,
        lora_r=args.lora_r
    )
    
    # Run unlearning
    model = unlearner.unlearn(
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        eval_every=args.eval_every
    )
    
    # Save model
    save_path = os.path.join(os.path.dirname(__file__), 'gs_lora_unlearned.pth')
    torch.save({
        'model_state': model.state_dict(),
        'forget_classes': forget_classes,
        'retain_classes': retain_classes,
        'config': vars(args)
    }, save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
