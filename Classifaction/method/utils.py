import torch
import torch.nn as nn
import numpy as np
from itertools import combinations
import random


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


def compute_true_voronoi_vertex(selected_centroids, max_iterations=100, tolerance=1e-6):
    """Compute true Voronoi vertex equidistant from selected centroids using iterative optimization"""
    k, dim = selected_centroids.shape
    
    if k < 2:
        return None
    
    # Initialize at mean of centroids
    vertex = selected_centroids.mean(dim=0)
    
    for iteration in range(max_iterations):
        # Compute distances to all selected centroids
        distances = torch.norm(selected_centroids - vertex.unsqueeze(0), dim=1)
        
        # Check if already equidistant (within tolerance)
        if torch.std(distances) < tolerance:
            break
        
        # Gradient descent to minimize distance variance
        # Compute gradient of distance variance with respect to vertex
        mean_dist = distances.mean()
        grad = torch.zeros_like(vertex)
        
        for i in range(k):
            diff = vertex - selected_centroids[i]
            dist = distances[i]
            if dist > 0:
                grad += 2 * (dist - mean_dist) * diff / dist
        
        # Update vertex position
        lr = 0.01
        vertex = vertex - lr * grad
    
    # Verify final equidistance quality
    final_distances = torch.norm(selected_centroids - vertex.unsqueeze(0), dim=1)
    variance = torch.var(final_distances)
    
    # Only return if reasonably equidistant
    if variance < tolerance * 10:
        return vertex
    else:
        return None


def compute_circumcenter_nd(three_centroids):
    """Compute circumcenter-like point for 3 centroids in high-dimensional space"""
    if three_centroids.shape[0] != 3:
        return None
    
    # Use circumcenter formula adapted for high dimensions
    a, b, c = three_centroids[0], three_centroids[1], three_centroids[2]
    
    # Vectors
    ab = b - a
    ac = c - a
    
    # Check if points are too close (nearly collinear)
    if torch.norm(ab) < 1e-8 or torch.norm(ac) < 1e-8:
        return (a + b + c) / 3  # Fallback to centroid
    
    # Use generalized circumcenter formula for n-dimensional space
    # Find point equidistant from all three points
    ab_norm_sq = torch.norm(ab) ** 2
    ac_norm_sq = torch.norm(ac) ** 2
    ab_dot_ac = torch.dot(ab, ac)
    
    # Determinant check for non-collinear points
    det = ab_norm_sq * ac_norm_sq - ab_dot_ac ** 2
    if abs(det) < 1e-8:
        return (a + b + c) / 3  # Fallback to centroid
    
    # Solve for coefficients that make point equidistant from a, b, c
    # This is a simplified approach that works in high dimensions
    alpha = (ac_norm_sq - ab_dot_ac) / det
    beta = (ab_norm_sq - ab_dot_ac) / det
    
    # Compute circumcenter
    circumcenter = a + alpha * ab + beta * ac
    
    # Verify equidistance quality
    dist_a = torch.norm(circumcenter - a)
    dist_b = torch.norm(circumcenter - b) 
    dist_c = torch.norm(circumcenter - c)
    
    # If not reasonably equidistant, fall back to centroid
    distances = torch.stack([dist_a, dist_b, dist_c])
    if torch.std(distances) / torch.mean(distances) > 0.1:
        return (a + b + c) / 3
    
    return circumcenter


def compute_voronoi_vertices(centroids_dict, max_vertices=50):
    """Compute true Voronoi vertices equidistant from multiple centroids"""
    import gc
    from tqdm import tqdm
    
    centroids = torch.stack(list(centroids_dict.values()))
    n_centroids = centroids.shape[0]
    vertices = []
    degrees = []
    
    print(f"Computing true Voronoi vertices for {n_centroids} centroids...")
    
    # Start with k=3 and k=2 for stability in high dimensions
    max_k = min(n_centroids, 3)  # Limit to k=3 for high-dimensional stability
    
    pbar = tqdm(desc="Computing Voronoi vertices", total=max_vertices)
    
    for k in [3, 2]:  # Prioritize k=3, then k=2
        if len(vertices) >= max_vertices or k > n_centroids:
            break
            
        # Sample combinations to avoid exponential explosion
        all_combinations = list(combinations(range(n_centroids), k))
        max_combinations = min(len(all_combinations), 100)  # Further reduced
        sampled_combinations = random.sample(all_combinations, max_combinations) if len(all_combinations) > max_combinations else all_combinations
        
        for combo in sampled_combinations:
            if len(vertices) >= max_vertices:
                break
            
            selected_centroids = centroids[list(combo)]
            
            # Compute Voronoi vertex
            if k == 3:
                vertex = compute_circumcenter_nd(selected_centroids)
            elif k == 2:
                # For k=2, midpoint is the true Voronoi vertex
                vertex = selected_centroids.mean(dim=0)
            else:
                vertex = compute_true_voronoi_vertex(selected_centroids)
            
            if vertex is not None:
                # Simple validation: check if vertex is reasonable
                all_distances = torch.norm(centroids - vertex.unsqueeze(0), dim=1)
                
                # Check if the k selected centroids are among the nearest
                sorted_indices = torch.argsort(all_distances)
                k_nearest = sorted_indices[:min(k+1, len(sorted_indices))]  # Allow some tolerance
                
                if len(set(combo).intersection(set(k_nearest.tolist()))) >= k-1:  # At least k-1 should be close
                    vertices.append(vertex)
                    degrees.append(k)
                    pbar.update(1)
    
    pbar.close()
    
    # Always add some high-quality midpoints if we need more
    if len(vertices) < max_vertices:
        print("Adding additional midpoints...")
        remaining = max_vertices - len(vertices)
        pairs = list(combinations(range(n_centroids), 2))
        random.shuffle(pairs)
        
        for i, j in pairs[:remaining]:
            midpoint = (centroids[i] + centroids[j]) / 2
            vertices.append(midpoint)
            degrees.append(2)
    
    del centroids
    gc.collect()
    
    print(f"Generated {len(vertices)} Voronoi vertices")
    return torch.stack(vertices) if vertices else None, degrees


# Removed expensive solve_equidistant_point function
# Now using fast geometric methods instead


def find_safe_border_targets(forget_centroids, retain_centroids, n_targets, min_forget_distance=2.0):
    """Find safe border positions between retain classes with 2-round defense from forget classes
    
    Args:
        forget_centroids: Dict of forget class centroids
        retain_centroids: Dict of retain class centroids
        n_targets: Number of target positions needed
        min_forget_distance: Minimum distance from any forget centroid (2-round defense)
    
    Returns:
        Tensor of safe border target positions
    """
    if not forget_centroids or not retain_centroids:
        print("Warning: Missing centroids for border target generation")
        return None
    
    print(f"Finding {n_targets} safe border targets between retain classes...")
    print(f"2-round defense: ensuring min distance {min_forget_distance:.2f} from forget centroids")
    
    # Convert to tensors
    forget_tensor = torch.stack(list(forget_centroids.values()))
    retain_tensor = torch.stack(list(retain_centroids.values()))
    retain_class_ids = list(retain_centroids.keys())
    
    safe_targets = []
    max_attempts = 1000  # Prevent infinite loops
    attempts = 0
    
    # Generate border positions between pairs of retain classes
    retain_pairs = list(combinations(range(len(retain_tensor)), 2))
    random.shuffle(retain_pairs)  # Randomize order
    
    for i, j in retain_pairs:
        if len(safe_targets) >= n_targets:
            break
            
        attempts += 1
        if attempts > max_attempts:
            print(f"Warning: Reached max attempts ({max_attempts}), using {len(safe_targets)} targets")
            break
        
        # Get pair of retain centroids
        centroid_a = retain_tensor[i]
        centroid_b = retain_tensor[j]
        
        # Generate multiple border points between this pair
        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:  # Various positions along the border
            if len(safe_targets) >= n_targets:
                break
            
            # Border position between two retain centroids
            border_point = alpha * centroid_a + (1 - alpha) * centroid_b
            
            # Check 2-round defense: ensure safe distance from ALL forget centroids
            distances_to_forget = torch.norm(forget_tensor - border_point.unsqueeze(0), dim=1)
            min_distance_to_forget = torch.min(distances_to_forget).item()
            
            # Also check that this point is actually on the border (roughly equidistant from both retain centroids)
            dist_to_a = torch.norm(border_point - centroid_a).item()
            dist_to_b = torch.norm(border_point - centroid_b).item()
            border_balance = abs(dist_to_a - dist_to_b) / (dist_to_a + dist_to_b)
            
            # Accept if: 1) Safe distance from forget classes, 2) Good border balance
            if min_distance_to_forget >= min_forget_distance and border_balance < 0.3:
                safe_targets.append(border_point)
                print(f"Safe border target {len(safe_targets)}: between retain {retain_class_ids[i]}-{retain_class_ids[j]}, "
                      f"min_forget_dist={min_distance_to_forget:.3f}")
    
    # If we need more targets, generate some with perturbations
    while len(safe_targets) < n_targets and attempts < max_attempts:
        attempts += 1
        
        # Pick a random existing safe target and perturb it slightly
        if len(safe_targets) > 0:
            base_target = safe_targets[random.randint(0, len(safe_targets) - 1)]
            
            # Small random perturbation
            perturbation = torch.randn_like(base_target) * 0.1
            perturbed_target = base_target + perturbation
            
            # Check if perturbed version is still safe
            distances_to_forget = torch.norm(forget_tensor - perturbed_target.unsqueeze(0), dim=1)
            min_distance_to_forget = torch.min(distances_to_forget).item()
            
            if min_distance_to_forget >= min_forget_distance:
                safe_targets.append(perturbed_target)
                print(f"Safe perturbed target {len(safe_targets)}: min_forget_dist={min_distance_to_forget:.3f}")
        else:
            # Fallback: use midpoint between most distant retain centroids
            distances_between_retain = torch.cdist(retain_tensor, retain_tensor)
            max_dist_indices = torch.unravel_index(torch.argmax(distances_between_retain), distances_between_retain.shape)
            if max_dist_indices[0] != max_dist_indices[1]:
                midpoint = (retain_tensor[max_dist_indices[0]] + retain_tensor[max_dist_indices[1]]) / 2
                safe_targets.append(midpoint)
                print(f"Fallback target {len(safe_targets)}: midpoint of most distant retain centroids")
    
    if len(safe_targets) == 0:
        print("Warning: No safe border targets found! Using retain centroid midpoints as fallback")
        # Emergency fallback: use midpoints of retain centroids
        for i in range(min(n_targets, len(retain_pairs))):
            idx1, idx2 = retain_pairs[i]
            fallback_target = (retain_tensor[idx1] + retain_tensor[idx2]) / 2
            safe_targets.append(fallback_target)
    
    # Pad with duplicates if needed
    while len(safe_targets) < n_targets:
        safe_targets.append(safe_targets[0] if safe_targets else retain_tensor[0])
    
    result_targets = torch.stack(safe_targets[:n_targets])
    print(f"Generated {len(result_targets)} safe border targets with 2-round forget defense")
    
    return result_targets


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
        # Need to interpolate additional vertices
        selected_vertices = [vertices[i] for i in sorted_indices]
        
        # Generate additional vertices by interpolation
        while len(selected_vertices) < n_targets:
            idx1, idx2 = random.sample(range(len(vertices)), 2)
            alpha = random.uniform(0.3, 0.7)
            new_vertex = alpha * vertices[idx1] + (1 - alpha) * vertices[idx2]
            selected_vertices.append(new_vertex)
        
        return torch.stack(selected_vertices[:n_targets])


def assign_targets_to_classes(target_vertices, forget_classes, forget_centroids=None, method="simple"):
    """Assign target vertices to forget classes
    
    Args:
        target_vertices: Voronoi vertices
        forget_classes: List of forget class IDs
        forget_centroids: Dict of forget class centroids (for adaptive method)
        method: "simple" (random) or "advance" (nearest)
    """
    if target_vertices is None:
        return {}
    
    assignment = {}
    
    if method == "simple":
        # Random assignment (original method)
        print("Using random target assignment...")
        
        shuffled_targets = target_vertices[torch.randperm(len(target_vertices))]
        total_distance = 0.0
        valid_distances = 0
        
        for i, class_id in enumerate(forget_classes):
            target_idx = i % len(shuffled_targets)
            assignment[class_id] = shuffled_targets[target_idx]
            
            # Calculate travel distance if forget centroids available
            if forget_centroids is not None and class_id in forget_centroids:
                forget_centroid = forget_centroids[class_id]
                distance = torch.norm(shuffled_targets[target_idx] - forget_centroid).item()
                total_distance += distance
                valid_distances += 1
        
        # Report average travel distance
        if forget_centroids is not None and valid_distances > 0:
            avg_distance = total_distance / valid_distances
            print(f"Average travel distance to targets: {avg_distance:.4f}")
        else:
            print("Average travel distance: Not calculated (no forget centroids available)")
    
    elif method == "advance":
        # Nearest target assignment (adaptive method)
        if forget_centroids is None:
            print("Warning: No forget centroids provided, falling back to random assignment")
            return assign_targets_to_classes(target_vertices, forget_classes, None, "simple")
        
        print("Using adaptive nearest target assignment...")
        total_distance = 0.0
        
        for class_id in forget_classes:
            if class_id in forget_centroids:
                forget_centroid = forget_centroids[class_id]
                
                # Compute distances to all target vertices
                distances = torch.norm(target_vertices - forget_centroid.unsqueeze(0), dim=1)
                
                # Assign to nearest vertex
                nearest_idx = torch.argmin(distances)
                assignment[class_id] = target_vertices[nearest_idx]
                
                # Track distance for reporting
                total_distance += distances[nearest_idx].item()
        
        # Report average travel distance
        if len(assignment) > 0:
            avg_distance = total_distance / len(assignment)
            print(f"Average travel distance to targets: {avg_distance:.4f}")
    
    else:
        raise ValueError(f"Unknown assignment method: {method}")
    
    return assignment


def compute_forget_loss(embeddings, labels, target_assignment):
    """Compute MSE loss pulling forget samples toward assigned targets"""
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


def compute_classification_loss(model, imgs, labels, retain_classes, temperature=3.0):
    """Compute classification loss for forget samples with uniform target distribution"""
    # Get model predictions
    logits = model(imgs)
    
    # Apply temperature scaling for sharper gradients
    logits = logits / temperature
    
    # Create uniform target distribution over retain classes only
    batch_size = logits.size(0)
    num_retain = len(retain_classes)
    uniform_targets = torch.zeros_like(logits)
    
    # Set equal probability for retain classes
    for class_id in retain_classes:
        uniform_targets[:, class_id] = 1.0 / num_retain
    
    # Use KL divergence loss to push toward uniform distribution
    log_probs = torch.log_softmax(logits, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(log_probs, uniform_targets)
    
    return loss


def compute_group_sparse_regularization(model, lambda_reg=1e-3):
    """Compute group sparse regularization loss for LoRA parameters
    
    For each Transformer block (group g):
    1. Collect all LoRA parameters in that block's attention: {A_q, B_q, A_k, B_k, A_v, B_v}
    2. Compute group L2 norm: ||θ_g||₂ = sqrt(sum of all squared parameters)
    
    Regularization loss: L_group_sparse = λ * Σ_{all groups} ||θ_g||₂
    """
    if not hasattr(model, 'base_model'):
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    total_reg_loss = 0.0
    device = next(model.parameters()).device
    
    # Access LoRA model
    lora_model = model.base_model if hasattr(model, 'base_model') else model
    
    # Iterate through Transformer blocks
    for block_idx, (name, module) in enumerate(lora_model.named_modules()):
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