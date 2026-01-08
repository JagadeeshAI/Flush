import torch
import torch.nn as nn
from tqdm import tqdm
from data import get_dataloaders
from utils import get_model

# ---------------- CONFIG ----------------
DATA_DIR = "/media/jag/volD2/cifer100/cifer"
EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    train_loader, val_loader, num_classes = get_dataloaders(
        DATA_DIR, BATCH_SIZE
    )

    model = get_model(num_classes, model_path=None, device=DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            f"checkpoint_epoch_{epoch}.pth"
        )


if __name__ == "__main__":
    main()
