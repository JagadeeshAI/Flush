import torch
import torch.nn as nn


def extract_features(model, imgs):
    """Extract features from penultimate layer for LoRA models"""
    # For LoRA wrapped vision transformer
    if hasattr(model, 'base_model'):
        # Access the underlying vision transformer
        vit = model.base_model.model if hasattr(model.base_model, 'model') else model.base_model
        
        # Forward through vision transformer layers (before head)
        x = vit.patch_embed(imgs)
        cls_token = vit.cls_token.expand(x.shape[0], -1, -1)
        
        # Check if dist_token exists (for DeiT models)
        if hasattr(vit, 'dist_token') and vit.dist_token is not None:
            x = torch.cat((cls_token, vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)
            
        x = vit.pos_drop(x + vit.pos_embed)
        x = vit.blocks(x)
        x = vit.norm(x)
        features = x[:, 0]  # CLS token features
    else:
        # For regular model
        x = model.patch_embed(imgs)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        x = model.blocks(x)
        x = model.norm(x)
        features = x[:, 0]  # CLS token features
    
    return features


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from model's penultimate layer"""
    import gc
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            # Process in smaller chunks to reduce memory
            chunk_size = min(16, imgs.size(0))
            chunk_features = []
            
            for j in range(0, imgs.size(0), chunk_size):
                chunk = imgs[j:j+chunk_size]
                features = extract_features(model, chunk)
                chunk_features.append(features.cpu())
                del features, chunk
                
            # Combine chunks
            batch_features = torch.cat(chunk_features, dim=0)
            embeddings.append(batch_features)
            labels.extend(lbls.tolist())
            
            # Clear memory every few batches
            del imgs, lbls, chunk_features, batch_features
            if i % 5 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
    
    result_embeddings = torch.cat(embeddings, dim=0)
    del embeddings
    gc.collect()
    
    return result_embeddings, torch.tensor(labels)


def compute_class_centroids(embeddings, labels, class_list):
    """Compute centroids for specified classes"""
    centroids = {}
    
    for class_id in class_list:
        class_mask = labels == class_id
        if class_mask.sum() > 0:
            centroids[class_id] = embeddings[class_mask].mean(dim=0)
    
    return centroids


def evaluate_model(model, data_dir, batch_size, forget_classes, retain_classes, device):
    """Evaluate model on forget and retain classes"""
    from codes.data import get_dataloaders
    
    def compute_acc(class_range, ratio):
        _, val_loader, _ = get_dataloaders(data_dir, batch_size, class_range=class_range, data_ratio=ratio)
        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return (correct / total) * 100 if total > 0 else 0.0
    
    forget_acc = compute_acc((0, 49), 1.0) if forget_classes else 0.0
    retain_acc = compute_acc((50, 99), 0.1) if retain_classes else 0.0
    return forget_acc, retain_acc


def get_visualization_data(dataloader):
    """Get limited visualization data"""
    from torch.utils.data import TensorDataset, DataLoader
    vis_data, vis_labels = [], []
    for count, (imgs, labels) in enumerate(dataloader):
        if count >= 10: break
        vis_data.append(imgs); vis_labels.extend(labels.tolist())
    
    if vis_data:
        return DataLoader(TensorDataset(torch.cat(vis_data, 0), torch.tensor(vis_labels)), batch_size=32)
    return dataloader


def create_visualization_step(visualizer, model, step, data_dir, batch_size, device, target_assignment):
    """Create visualization for a specific step"""
    if not visualizer: return
    
    try:
        from codes.data import get_dataloaders
        _, loader, _ = get_dataloaders(data_dir, batch_size//2, class_range=(0, 99), data_ratio=0.05)
        embeddings, labels = [], []
        
        with torch.no_grad():
            for imgs, lbls in loader:
                if len(labels) > 500: break
                imgs = imgs.to(device)
                emb = extract_features(model, imgs)
                embeddings.append(emb.cpu()); labels.extend(lbls.tolist())
        
        if embeddings:
            combined = torch.cat(embeddings, dim=0)
            path = visualizer.create_visualization(step, combined, labels, target_assignment)
            print(f"Visualization: {path}")
    except Exception as e:
        print(f"Visualization error at step {step}: {e}")