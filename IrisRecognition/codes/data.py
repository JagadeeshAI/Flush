import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from pathlib import Path

# Hard-coded dataset path
DATASET_ROOT = "/media/jag/volD2/Iris/CASIA-Iris-Thousand/CASIA-Iris-Thousand"


class IrisDataset(Dataset):
    """
    Iris Dataset with class-level unlearning.

    Forget set  = first 40% of classes  (labels: 0 .. forget_count-1)
    Retain set  = last 60% of classes   (labels: forget_count .. num_classes-1)
    """

    def __init__(
        self,
        root_dir=DATASET_ROOT,
        num_classes=50,
        train=True,
        train_ratio=0.8,
        augment=True,
        data_ratio=1.0,
        unlearning=False,
        retain=True,  # True = retained classes, False = forgotten classes
    ):
        self.root_dir = Path(root_dir)
        self.num_classes = num_classes
        self.train = train
        self.augment = augment and train
        self.data_ratio = data_ratio
        self.unlearning = unlearning
        self.retain = retain

        # Compute class split: 40% forget, 60% retain
        forget_count = int(0.1 * num_classes)
        self.forget_classes = set(range(forget_count))
        self.retain_classes = set(range(forget_count, num_classes))

        self.transform = self._build_transforms()
        self.samples = []

        self._load_data(train_ratio)

    # -----------------------------------------------------------
    def _build_transforms(self):
        if self.augment:
            return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomApply(
                        [
                            transforms.RandomChoice(
                                [
                                    transforms.RandomVerticalFlip(p=1.0),
                                    transforms.RandomHorizontalFlip(p=1.0),
                                    transforms.Lambda(
                                        lambda x: transforms.functional.rotate(x, -60)
                                    ),
                                    transforms.Lambda(
                                        lambda x: transforms.functional.rotate(x, -120)
                                    ),
                                    transforms.ColorJitter(brightness=0.3),
                                ]
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    # -----------------------------------------------------------
    def _load_data(self, train_ratio):
        person_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        person_dirs = person_dirs[: self.num_classes]

        classes_loaded = set()

        for label, person_dir in enumerate(person_dirs):

            # SKIP IRRELEVANT CLASSES BASED ON MODE
            if self.unlearning:
                if self.retain:
                    # Retain dataset → only keep retain_classes
                    if label not in self.retain_classes:
                        continue
                    classes_loaded.add(label)
                else:
                    # Forget dataset → only keep forget_classes
                    if label not in self.forget_classes:
                        continue
                    classes_loaded.add(label)

            # Load images from class folder
            imgs = sorted(glob.glob(str(person_dir / "L" / "*.jpg")))
            imgs += sorted(glob.glob(str(person_dir / "R" / "*.jpg")))

            if len(imgs) == 0:
                continue

            # Deterministic shuffle
            np.random.seed(42 + label)
            np.random.shuffle(imgs)

            # Train/test split
            split_idx = int(len(imgs) * train_ratio)
            selected = imgs[:split_idx] if self.train else imgs[split_idx:]

            # Apply data_ratio only to retained train set
            if self.train and self.retain and self.data_ratio < 1.0 and self.unlearning:
                k = max(1, int(len(selected) * self.data_ratio))
                selected = selected[:k]

            # Add samples
            for img_path in selected:
                self.samples.append((img_path, label))

        # Verification
        mode = "TRAIN" if self.train else "TEST"
        flag = "RETAINED" if self.retain else "FORGOTTEN"
        expected_classes = len(self.retain_classes if self.retain else self.forget_classes)
        actual_classes = len(classes_loaded)
        
        print(
            f"{mode} | {flag} | classes={actual_classes}/{expected_classes} | "
            f"ratio={self.data_ratio} → {len(self.samples)} samples"
        )

        # Sanity check
        if self.unlearning and actual_classes != expected_classes:
            print(f"⚠️  WARNING: Expected {expected_classes} classes but loaded {actual_classes}")

    # -----------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------------------
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label, idx


# =====================================================
# GET DATALOADERS
# =====================================================
def get_dataloaders(
    num_classes=50,
    batch_size=4,
    num_workers=4,
    data_ratio=1.0,
    unlearning=False,
):
    
    if not unlearning:
        # Standard classification mode (no unlearning split)
        train_dataset = IrisDataset(
            num_classes=num_classes,
            train=True,
            augment=True,
            data_ratio=data_ratio,
            unlearning=False,
        )

        test_dataset = IrisDataset(
            num_classes=num_classes,
            train=False,
            augment=False,
            data_ratio=1.0,
            unlearning=False,
        )

        return (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        )

    # ---------------------------------------------------------
    # UNLEARNING MODE: RETURN 4 DATASETS
    # ---------------------------------------------------------
    retained_train = IrisDataset(
        num_classes=num_classes,
        train=True,
        augment=True,
        data_ratio=data_ratio,
        unlearning=True,
        retain=True,
    )

    forgotten_train = IrisDataset(
        num_classes=num_classes,
        train=True,
        augment=True,
        data_ratio=1.0,
        unlearning=True,
        retain=False,
    )

    retained_test = IrisDataset(
        num_classes=num_classes,
        train=False,
        augment=False,
        data_ratio=1.0,
        unlearning=True,
        retain=True,
    )

    forgotten_test = IrisDataset(
        num_classes=num_classes,
        train=False,
        augment=False,
        data_ratio=1.0,
        unlearning=True,
        retain=False,
    )

    return (
        DataLoader(
            retained_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        DataLoader(
            forgotten_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(
            retained_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        DataLoader(
            forgotten_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )