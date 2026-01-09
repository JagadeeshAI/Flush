import torch
import torch.nn as nn
import numpy as np
from itertools import combinations
import random
from scipy.spatial import Voronoi
from sklearn.decomposition import PCA

def extract_classifier_weights(model, class_list):
    """Extract classifier head weights for specified classes"""
    weights = {}
    
    # Access the classifier head
    if hasattr(model, 'base_model'):
        # For LoRA wrapped model
        if hasattr(model.base_model, 'head'):
            head_weight = model.base_model.head.weight  # Shape: [num_classes, embed_dim]
        elif hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'head'):
            head_weight = model.base_model.model.head.weight
        else:
            raise ValueError("Could not find classifier head in LoRA model")
    else:
        # For regular model
        if hasattr(model, 'head'):
            head_weight = model.head.weight
        else:
            raise ValueError("Could not find classifier head in model")
    
    # Extract weights for specified classes
    for class_id in class_list:
        if class_id < head_weight.shape[0]:
            weights[class_id] = head_weight[class_id].detach().clone()
    
    return weights


def compute_voronoi_vertices_from_weights(weights_dict, max_vertices=50):
    """Generate targets on perpendicular bisectors between retain class pairs"""
    import gc
    from itertools import combinations
    
    weights = torch.stack(list(weights_dict.values()))
    n_weights = weights.shape[0]
    
    print(f"Generating targets from {max_vertices} weight pairs at multiple distances...")
    
    # Get all weight pairs, prioritize distant pairs
    weight_pairs = list(combinations(range(n_weights), 2))
    
    # Sort pairs by distance (farthest first)
    pair_distances = [(i, j, torch.norm(weights[i] - weights[j]).item()) 
                      for i, j in weight_pairs]
    pair_distances.sort(key=lambda x: x[2], reverse=True)
    
    vertices = []
    for i, j, _ in pair_distances[:max_vertices]:
        # Add intermediate points at multiple distances along the line
        for alpha in [0.25, 0.5, 0.75]:
            interpolated = alpha * weights[i] + (1-alpha) * weights[j]
            vertices.append(interpolated)
        
        # Also add the midpoint (0.5 case, but keeping for clarity)
        midpoint = (weights[i] + weights[j]) / 2
        vertices.append(midpoint)
    
    print(f"Generated {len(vertices)} targets at varying distances along weight boundaries")
    
    vertices_torch = torch.stack(vertices)
    degrees = [2] * len(vertices_torch)
    
    del weights
    gc.collect()
    
    return vertices_torch, degrees

def select_target_vertices(vertices, degrees, n_targets):
    """Select target vertices prioritizing high-degree ones"""
    if vertices is None or len(vertices) == 0:
        return None
    
    # Sort by degree (descending)
    sorted_indices = sorted(range(len(degrees)), key=lambda i: degrees[i], reverse=True)
    
    if len(vertices) >= n_targets:
        selected_indices = sorted_indices[:n_targets]
        return vertices[selected_indices]
    else:
        # Need to generate additional vertices
        selected_vertices = [vertices[i] for i in sorted_indices]
        
        # Generate additional vertices by interpolation
        while len(selected_vertices) < n_targets:
            if len(vertices) == 1:
                # Only one vertex, add noise to create variations
                base_vertex = vertices[0]
                noise = torch.randn_like(base_vertex) * 0.1 * torch.norm(base_vertex)
                new_vertex = base_vertex + noise
            else:
                # Multiple vertices, interpolate between random pairs
                idx1, idx2 = random.sample(range(len(vertices)), 2)
                alpha = random.uniform(0.3, 0.7)
                new_vertex = alpha * vertices[idx1] + (1 - alpha) * vertices[idx2]
            
            selected_vertices.append(new_vertex)
        
        return torch.stack(selected_vertices[:n_targets])


def assign_targets_to_classes(target_vertices, forget_classes, forget_centroids=None, method="simple", 
                             retain_weights_stacked=None, forget_logits=None):
    """Assign target vertices to forget classes with unique targets"""
    if target_vertices is None:
        return {}
    
    assignment = {}
    
    if method == "simple":
        # Random assignment ensuring unique targets per class
        print("Using random target assignment with unique targets...")
        
        if len(target_vertices) < len(forget_classes):
            print(f"Warning: Only {len(target_vertices)} targets available for {len(forget_classes)} forget classes")
        
        shuffled_targets = target_vertices[torch.randperm(len(target_vertices))]
        total_distance = 0.0
        valid_distances = 0
        
        for i, class_id in enumerate(forget_classes):
            if i < len(shuffled_targets):
                # Assign unique target to each forget class
                assignment[class_id] = shuffled_targets[i]
            else:
                # If more forget classes than targets, assign to a random target (but warn)
                target_idx = torch.randint(0, len(shuffled_targets), (1,)).item()
                assignment[class_id] = shuffled_targets[target_idx]
                print(f"Warning: Reusing target for forget class {class_id}")
            
            assigned_target = assignment[class_id]
            
            # Calculate travel distance if forget centroids available
            if forget_centroids is not None and class_id in forget_centroids:
                forget_centroid = forget_centroids[class_id]
                
                # Ensure both tensors are on the same device
                if assigned_target.device != forget_centroid.device:
                    forget_centroid = forget_centroid.to(assigned_target.device)
                
                distance = torch.norm(assigned_target - forget_centroid).item()
                total_distance += distance
                valid_distances += 1
        
        # Report average travel distance
        if forget_centroids is not None and valid_distances > 0:
            avg_distance = total_distance / valid_distances
            print(f"Average travel distance to targets: {avg_distance:.4f}")
        else:
            print("Average travel distance: Not calculated (no forget centroids available)")
    
    elif method == "advance":
        # Nearest target assignment using embedding-space distances to minimize MSE travel distance
        if forget_centroids is None:
            print("Warning: No forget centroids provided, falling back to random assignment")
            return assign_targets_to_classes(target_vertices, forget_classes, None, "simple")
        
        print("Using adaptive assignment with embedding-space distances...")
        total_distance = 0.0
        
        for class_id in forget_classes:
            if class_id in forget_centroids:
                forget_centroid = forget_centroids[class_id]
                
                # Ensure both tensors are on the same device
                if target_vertices.device != forget_centroid.device:
                    forget_centroid = forget_centroid.to(target_vertices.device)
                
                # Compute embedding-space distances to minimize actual MSE travel distance
                distances = torch.norm(target_vertices - forget_centroid.unsqueeze(0), dim=1)
                
                # Assign to nearest vertex in embedding space
                nearest_idx = torch.argmin(distances)
                assignment[class_id] = target_vertices[nearest_idx]
                
                # Track distance for reporting
                total_distance += distances[nearest_idx].item()
        
        # Report average travel distance
        if len(assignment) > 0:
            avg_distance = total_distance / len(assignment)
            print(f"Average embedding-space travel distance to targets: {avg_distance:.4f}")
    
    else:
        raise ValueError(f"Unknown assignment method: {method}")
    
    return assignment


def orthogonalize_embeddings_to_weights(embeddings, weights):
    """Orthogonalize embeddings to given weights using Gram-Schmidt process"""
    # weights: [num_classes, embed_dim] or stacked weight vectors
    # embeddings: [batch_size, embed_dim]
    
    if len(weights.shape) == 1:
        weights = weights.unsqueeze(0)  # Make it [1, embed_dim]
    
    orthogonalized_embeddings = embeddings.clone()
    
    for weight_vector in weights:
        # Normalize the weight vector
        weight_norm = weight_vector / (torch.norm(weight_vector) + 1e-8)
        
        # Project embeddings onto weight vector
        projection = torch.sum(orthogonalized_embeddings * weight_norm.unsqueeze(0), dim=1, keepdim=True) * weight_norm.unsqueeze(0)
        
        # Remove the projection (orthogonalize)
        orthogonalized_embeddings = orthogonalized_embeddings - projection
    
    return orthogonalized_embeddings


def compute_auxiliary_logit_constraint(logits, forget_classes, retain_classes, margin=2.0):
    """Compute auxiliary logit constraint: max(retain_logits) > max(forget_logits) + margin"""
    # Extract forget and retain logits
    forget_logits = logits[:, forget_classes]  # Shape: [batch_size, num_forget_classes]
    retain_logits = logits[:, retain_classes]  # Shape: [batch_size, num_retain_classes]
    
    # Get max logits
    max_forget_logits, _ = forget_logits.max(dim=1)  # Shape: [batch_size]
    max_retain_logits, _ = retain_logits.max(dim=1)  # Shape: [batch_size]
    
    # Compute constraint: we want max_retain > max_forget + margin
    # Loss is 0 if constraint is satisfied, positive otherwise
    constraint_violation = torch.relu(max_forget_logits + margin - max_retain_logits)
    
    return constraint_violation.mean()



def compute_forget_loss(embeddings, labels, target_assignment):
    """Compute MSE loss pulling forget samples toward pre-orthogonalized assigned targets"""
    loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    count = 0
    
    for class_id, target_vertex in target_assignment.items():
        class_mask = labels == class_id
        if class_mask.sum() > 0:
            class_embeddings = embeddings[class_mask]
            target_expanded = target_vertex.expand_as(class_embeddings).to(embeddings.device)
            loss = loss + nn.MSELoss()(class_embeddings, target_expanded)
            count += 1
    
    return loss / max(count, 1)


def compute_retain_loss(embeddings, labels, retain_centroids, use_mse=True):
    """Compute loss to anchor retain samples to their centroids"""
    loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    count = 0
    
    for class_id, centroid in retain_centroids.items():
        class_mask = labels == class_id
        if class_mask.sum() > 0:
            class_embeddings = embeddings[class_mask]
            
            if use_mse:
                centroid_expanded = centroid.expand_as(class_embeddings).to(embeddings.device)
                loss = loss + nn.MSELoss()(class_embeddings, centroid_expanded)
            else:
                # Use cosine similarity loss
                cos_sim = nn.CosineSimilarity(dim=1)
                similarities = cos_sim(class_embeddings, centroid.expand_as(class_embeddings).to(embeddings.device))
                loss = loss + (1 - similarities).mean()
            
            count += 1
    
    return loss / max(count, 1)


def compute_group_sparse_regularization(model, lambda_reg=1e-3):
    """Compute group sparse regularization loss for LoRA parameters"""
    if not hasattr(model, 'base_model'):
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    total_reg_loss = 0.0
    
    # Access LoRA model
    lora_model = model.base_model if hasattr(model, 'base_model') else model
    
    # Iterate through Transformer blocks
    for name, module in lora_model.named_modules():
        if 'blocks' in name and 'attn.qkv' in name:
            # This is an attention module in a transformer block
            group_params = []
            
            # Collect LoRA A and B parameters for this attention block
            for param_name, param in module.named_parameters():
                if 'lora_' in param_name and ('A' in param_name or 'B' in param_name):
                    group_params.append(param.view(-1))  # Flatten parameter
            
            # Compute group L2 norm if we found LoRA parameters
            if group_params:
                # Concatenate all parameters in this group
                group_tensor = torch.cat(group_params)
                # Compute L2 norm
                group_norm = torch.norm(group_tensor, p=2)
                total_reg_loss += group_norm
    
    # Apply lambda regularization weight
    reg_loss = lambda_reg * total_reg_loss
    
    return reg_loss