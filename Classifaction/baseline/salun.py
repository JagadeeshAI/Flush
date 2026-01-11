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
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from codes.data import get_dataloaders


def get_salun_model(num_classes=100, model_path=None, device="cpu"):
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
    
    # Initially freeze all, will selectively unfreeze based on saliency
    for param in model.parameters():
        param.requires_grad = False
    
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    
    return model.to(device)


def compute_h_factor(acc_r, acc_f_before, acc_f_after):
    drop = max(acc_f_before - acc_f_after, 0)
    if acc_r + drop == 0:
        return 0.0
    return 2 * acc_r * drop / (acc_r + drop)


@torch.no_grad()
def evaluate(model, data_dir, batch_size, class_range, device, desc="Eval"):
    _, val_loader, _ = get_dataloaders(data_dir, batch_size, class_range=class_range, data_ratio=1.0)
    
    model.eval()
    correct, total = 0, 0
    
    for imgs, labels in tqdm(val_loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0


def compute_weight_saliency_map(model, forget_loader, device, num_batches=10):
    """
    Compute gradient-based weight saliency map.
    m_S = 1(|∇_θ L_f(θ; D_f)| >= γ)
    where γ = median of absolute gradients
    """
    model.train()
    
    # Temporarily enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Accumulate gradients over forget set
    accumulated_grads = {}
    for name, param in model.named_parameters():
        accumulated_grads[name] = torch.zeros_like(param)
    
    forget_iter = iter(forget_loader)
    for _ in range(min(num_batches, len(forget_loader))):
        try:
            imgs, labels = next(forget_iter)
        except StopIteration:
            break
        
        imgs, labels = imgs.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_grads[name] += param.grad.abs()
    
    # Compute saliency mask using median threshold
    saliency_masks = {}
    all_grads = []
    
    for name, grad in accumulated_grads.items():
        all_grads.append(grad.flatten())
    
    all_grads_flat = torch.cat(all_grads)
    threshold = torch.median(all_grads_flat).item()
    
    salient_count = 0
    total_count = 0
    
    for name, grad in accumulated_grads.items():
        mask = (grad >= threshold).float()
        saliency_masks[name] = mask
        salient_count += mask.sum().item()
        total_count += mask.numel()
    
    print(f"Saliency threshold (median): {threshold:.6f}")
    print(f"Salient parameters: {int(salient_count):,} / {total_count:,} ({100*salient_count/total_count:.2f}%)")
    
    # Reset gradients
    model.zero_grad()
    for param in model.parameters():
        param.requires_grad = False
    
    return saliency_masks


class SalUnUnlearning:
    
    def __init__(self, model_path, forget_classes, retain_classes, device="cuda", alpha=1.0):
        self.device = device
        self.forget_classes = forget_classes
        self.retain_classes = retain_classes
        self.alpha = alpha  # Retain regularization weight
        
        num_classes = len(forget_classes) + len(retain_classes)
        self.model = get_salun_model(num_classes, model_path, device)
        self.original_state = copy.deepcopy(self.model.state_dict())
        
        self.saliency_masks = None
        
        print(f"\nSalUn Config:")
        print(f"  α (retain weight): {alpha}")
    
    def compute_saliency(self, data_dir, batch_size):
        forget_range = (min(self.forget_classes), max(self.forget_classes))
        forget_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=forget_range, data_ratio=1.0)
        
        print("\n=== Computing Weight Saliency Map ===")
        self.saliency_masks = compute_weight_saliency_map(self.model, forget_loader, self.device)
        
        # Enable gradients only for salient parameters
        for name, param in self.model.named_parameters():
            if name in self.saliency_masks:
                # Only train where mask is non-zero
                if self.saliency_masks[name].sum() > 0:
                    param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters after saliency: {trainable:,}")
    
    def apply_saliency_mask(self):
        """Apply saliency mask: only keep updates for salient weights"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.saliency_masks and name in self.original_state:
                    mask = self.saliency_masks[name].to(self.device)
                    original = self.original_state[name].to(self.device)
                    # θ_u = m_S ⊙ θ_current + (1 - m_S) ⊙ θ_original
                    param.data = mask * param.data + (1 - mask) * original
    
    def random_label_loss(self, imgs, labels, num_classes=100):
        """Random labeling loss: assign random wrong labels"""
        # Generate random labels different from true labels
        random_labels = torch.randint(0, num_classes, labels.shape, device=self.device)
        # Ensure random labels are different from true labels
        same_mask = (random_labels == labels)
        random_labels[same_mask] = (random_labels[same_mask] + 1) % num_classes
        
        outputs = self.model(imgs)
        return F.cross_entropy(outputs, random_labels)
    
    def retain_loss(self, imgs, labels):
        outputs = self.model(imgs)
        return F.cross_entropy(outputs, labels)
    
    def unlearn(self, data_dir, batch_size=64, max_steps=200, lr=1e-4, eval_every=50):
        
        forget_range = (min(self.forget_classes), max(self.forget_classes))
        retain_range = (min(self.retain_classes), max(self.retain_classes))
        
        # Compute saliency map first
        self.compute_saliency(data_dir, batch_size)
        
        # Data loaders
        forget_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=forget_range, data_ratio=1.0)
        retain_loader, _, _ = get_dataloaders(data_dir, batch_size, class_range=retain_range, data_ratio=0.3)
        
        # Initial evaluation
        print("\n=== Initial Evaluation ===")
        init_acc_f = evaluate(self.model, data_dir, batch_size, forget_range, self.device, "Forget")
        init_acc_r = evaluate(self.model, data_dir, batch_size, retain_range, self.device, "Retain")
        print(f"Forget Acc: {init_acc_f:.2f}% | Retain Acc: {init_acc_r:.2f}%")
        
        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        
        total_train_time = 0.0
        
        print(f"\n=== Training with SalUn ({max_steps} steps) ===")
        pbar = tqdm(range(1, max_steps + 1), desc="Training")
        
        for step in pbar:
            train_start = time.time()
            self.model.train()
            
            # Get batches
            try:
                f_imgs, f_labels = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                f_imgs, f_labels = next(forget_iter)
            
            try:
                r_imgs, r_labels = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_imgs, r_labels = next(retain_iter)
            
            f_imgs, f_labels = f_imgs.to(self.device), f_labels.to(self.device)
            r_imgs, r_labels = r_imgs.to(self.device), r_labels.to(self.device)
            
            # SalUn loss: Random labeling on forget + CE on retain
            l_forget = self.random_label_loss(f_imgs, f_labels)
            l_retain = self.retain_loss(r_imgs, r_labels)
            
            total_loss = l_forget + self.alpha * l_retain
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Apply saliency mask after each update
            self.apply_saliency_mask()
            
            total_train_time += time.time() - train_start
            
            pbar.set_postfix({
                'L_f': f"{l_forget.item():.3f}",
                'L_r': f"{l_retain.item():.3f}"
            })
            
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
    parser = argparse.ArgumentParser(description="SalUn: Saliency-based Unlearning")
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0, help="Retain regularization weight")
    parser.add_argument('--eval_every', type=int, default=50)
    args = parser.parse_args()
    
    DATA_DIR = "/media/jag/volD2/cifer100/cifer"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../checkpoints/best.pth')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    forget_classes = list(range(0, 45))
    retain_classes = list(range(45, 90))
    
    print("=" * 60)
    print("SalUn: Saliency-based Machine Unlearning (ICLR 2024)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Forget classes: {min(forget_classes)}-{max(forget_classes)} ({len(forget_classes)} classes)")
    print(f"Retain classes: {min(retain_classes)}-{max(retain_classes)} ({len(retain_classes)} classes)")
    
    unlearner = SalUnUnlearning(
        model_path=MODEL_PATH,
        forget_classes=forget_classes,
        retain_classes=retain_classes,
        device=DEVICE,
        alpha=args.alpha
    )
    
    model = unlearner.unlearn(
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        eval_every=args.eval_every
    )
    
    save_path = os.path.join(os.path.dirname(__file__), 'salun_unlearned.pth')
    torch.save({
        'model_state': model.state_dict(),
        'forget_classes': forget_classes,
        'retain_classes': retain_classes,
        'config': vars(args)
    }, save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
