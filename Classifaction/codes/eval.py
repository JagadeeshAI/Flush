import torch
import torch.nn as nn
from tqdm import tqdm
from data import get_dataloaders
from utils import get_model
import sys

# ---------------- CONFIG ----------------
DATA_DIR = "/media/jag/volD2/cifer100/cifer"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

@torch.no_grad()
def evaluate_model(model_path, class_range=(0, 99), data_ratio=1.0):
    if not model_path:
        print("Error: Model path required for evaluation")
        sys.exit(1)
    
    _, val_loader, num_classes = get_dataloaders(
        DATA_DIR, BATCH_SIZE, class_range=class_range, data_ratio=data_ratio
    )
    
    model = get_model(num_classes, model_path=model_path, device=DEVICE)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0
    
    for imgs, labels in tqdm(val_loader, desc="Evaluating"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    
    val_loss = total_loss / total
    val_acc = correct / total
    
    print(f"\nValidation Results:")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc:  {val_acc:.4f} ({correct}/{total})")
    
    return val_loss, val_acc

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval.py <model_checkpoint_path> [class_start] [class_end] [data_ratio]")
        print("Example: python eval.py checkpoint_epoch_10.pth 0 99 1.0")
        sys.exit(1)
    
    model_path = sys.argv[1]
    class_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    class_end = int(sys.argv[3]) if len(sys.argv) > 3 else 99
    data_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    
    evaluate_model(model_path, (class_start, class_end), data_ratio)