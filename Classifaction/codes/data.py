import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataloaders(data_dir="/media/jag/volD2/cifer100/cifer", batch_size=64, num_workers=4, class_range=(0, 99), data_ratio=1.0):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val", transform=val_tfms)
    
    def filter_dataset(dataset, class_range, data_ratio):
        indices = []
        targets = np.array(dataset.targets)
        
        for class_idx in range(class_range[0], min(class_range[1] + 1, len(dataset.classes))):
            class_indices = np.where(targets == class_idx)[0]
            n_samples = int(len(class_indices) * data_ratio)
            if n_samples > 0:
                selected_indices = np.random.choice(class_indices, n_samples, replace=False)
                indices.extend(selected_indices)
        
        return Subset(dataset, indices)
    
    train_ds_filtered = filter_dataset(train_ds, class_range, data_ratio)
    val_ds_filtered = filter_dataset(val_ds, class_range, data_ratio=1.0)

    train_loader = DataLoader(
        train_ds_filtered, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds_filtered, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    num_classes = class_range[1] - class_range[0] + 1
    return train_loader, val_loader, num_classes
