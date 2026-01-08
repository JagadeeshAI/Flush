import torch
from tqdm import tqdm

def _unpack_batch(batch):
    """
    Accepts a batch from a dataloader and returns (imgs, labels).

    Handles batches of form:
      - (imgs, labels)
      - (imgs, labels, indices)
      - (imgs, labels, anything_else...)
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            imgs, labels = batch
            return imgs, labels
        elif len(batch) >= 2:
            imgs, labels = batch[0], batch[1]
            return imgs, labels
    raise ValueError(
        "Unsupported batch format from dataloader. "
        "Expected (imgs, labels) or (imgs, labels, idx, ...)."
    )


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images, labels = _unpack_batch(batch)

            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images, return_embeddings=True)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x, original_output):
        # x: input, original_output: output from frozen layer
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_out


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank=rank,
            alpha=alpha
        )
        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_out = self.original(x)
        return self.lora(x, original_out)


def add_lora_to_model(model, rank=8, alpha=16):
    """Add LoRA to fc layer (embedding projection)"""
    
    # Freeze entire backbone first
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace fc with LoRA version
    original_fc = model.backbone.fc
    model.backbone.fc = LoRALinear(original_fc, rank=rank, alpha=alpha)
    
    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total: {total:,} | Trainable (LoRA): {trainable:,}")
    
    return model


class IrisTrainer:
    """Training pipeline with both softmax pre-training and triplet fine-tuning"""
    
    def __init__(self, num_classes=1000, device='cuda', learning_rate=0.001):
        from model import create_iris_recognition_model, EnhancedTripletLoss
        import torch.optim as optim
        
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lr = learning_rate
        
        self.model = create_iris_recognition_model(num_classes=num_classes)
        self.model.to(self.device)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = EnhancedTripletLoss(margin=0.3)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        print(f"Model initialized on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch_softmax(self, train_loader):
        """Train one epoch with softmax loss"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs, features = self.model(images)
            loss = self.ce_loss(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100.*correct/total:.2f}%'})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train_epoch_triplet(self, train_loader):
        """Train one epoch with triplet loss"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc='Triplet Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            features = self.model.extract_features(images)
            triplet_loss = self._compute_hard_triplet_loss(features, labels)
            
            if triplet_loss is not None:
                self.optimizer.zero_grad()
                triplet_loss.backward()
                self.optimizer.step()
                
                total_loss += triplet_loss.item()
                pbar.set_postfix({'triplet_loss': f'{triplet_loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        return avg_loss
    
    def _compute_hard_triplet_loss(self, embeddings, labels):
        """Compute hard triplet loss with online mining"""
        batch_size = embeddings.size(0)
        if batch_size < 3:
            return None
        
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        losses = []
        for i in range(batch_size):
            anchor_label = labels[i]
            
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size).to(self.device) != i)
            if not pos_mask.any():
                continue
            
            pos_dists = dist_mat[i][pos_mask]
            hardest_pos_dist = pos_dists.max()
            
            neg_mask = labels != anchor_label
            if not neg_mask.any():
                continue
            
            neg_dists = dist_mat[i][neg_mask]
            hardest_neg_dist = neg_dists.min()
            
            loss = torch.log(1 + torch.exp(hardest_pos_dist - hardest_neg_dist))
            losses.append(loss)
        
        if len(losses) == 0:
            return None
        
        return torch.stack(losses).mean()
    
    def train_full_pipeline(self, train_loader, test_loader, pretrain_epochs=200, 
                          finetune_epochs=100, target_acc=95.0):
        """Complete training pipeline: softmax pre-training + triplet fine-tuning"""
        import torch.optim as optim
        import os
        
        print("="*60)
        print("Phase 1: Softmax Pre-training")
        print("="*60)
        
        best_acc = 0
        
        # Phase 1: Softmax pre-training
        for epoch in range(pretrain_epochs):
            print(f"\nEpoch {epoch+1}/{pretrain_epochs}")
            
            train_loss, train_acc = self.train_epoch_softmax(train_loader)
            test_acc = evaluate(self.model, test_loader, self.device)
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Acc: {test_acc:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint('best_model.pth', epoch, test_acc)
            
            if train_acc >= target_acc:
                print(f"\n✓ Reached target accuracy {train_acc:.2f}%")
                break
        
        # Phase 2: E-Triplet fine-tuning
        print("\n" + "="*60)
        print("Phase 2: E-Triplet Fine-tuning")
        print("="*60)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr * 0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.5)
        
        for epoch in range(finetune_epochs):
            print(f"\nEpoch {epoch+1}/{finetune_epochs}")
            
            triplet_loss = self.train_epoch_triplet(train_loader)
            test_acc = evaluate(self.model, test_loader, self.device)
            self.scheduler.step()
            
            print(f"Triplet Loss: {triplet_loss:.4f}")
            print(f"Test Acc: {test_acc:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint('best_model.pth', epoch, test_acc)
        
        return best_acc
    
    def _save_checkpoint(self, filename, epoch, accuracy):
        """Save model checkpoint"""
        import os
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
        }
        path = os.path.join('checkpoints', filename)
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")