import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TSNEVoronoiVisualizer:
    """Animated t-SNE Voronoi visualization for Voronoi Unlearning"""
    
    def __init__(self, model, forget_classes, retain_classes, device="cuda", output_dir="visualizations", use_color=True):
        self.model = model
        self.forget_classes = forget_classes
        self.retain_classes = retain_classes
        self.device = device
        self.output_dir = output_dir
        self.use_color = use_color
        
        # Select subset for visualization (3 forget, ALL retain for complete coverage)
        self.viz_forget_classes = forget_classes[:3] if len(forget_classes) >= 3 else forget_classes
        self.viz_retain_classes = retain_classes  # Use ALL retain classes for complete coverage
        
        print(f"Visualization will show {len(self.viz_forget_classes)} forget classes: {self.viz_forget_classes}")
        print(f"Visualization will show {len(self.viz_retain_classes)} retain classes: {self.viz_retain_classes}")
        
        # t-SNE and visualization state
        self.tsne = None
        self.initial_embeddings = None
        self.initial_tsne_projections = None
        self.nn_mapper = None
        self.axis_limits = None
        self.retain_tsne_centroids = None
        self.voronoi_diagram = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup visualization parameters
        # Generate enough colors for all retain classes
        import matplotlib.cm as cm
        import numpy as np
        
        n_retain = len(retain_classes)
        n_colors_needed = max(n_retain, 50)  # Ensure we have enough
        
        # Generate distinct colors using colormap
        retain_colors = [cm.Set3(i / n_colors_needed)[:3] for i in range(n_colors_needed)]
        retain_colors_hex = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) 
                            for r, g, b in retain_colors]
        
        self.colors = {
            'retain': retain_colors_hex,
            'forget': ['#ff4444', '#ff6644', '#ff8844'],
            'targets': ['#FFD700', '#FF6347', '#FF1493'],  # Different colors for each target
            'background': '#f0f0f0'
        }
        
        # Generate gray shades for retain classes
        retain_grays = [f'{(i % 40 + 10)/50:.2f}' for i in range(n_colors_needed)]
        
        self.gray_colors = {
            'retain': retain_grays,
            'forget': ['0.1', '0.15', '0.2'],
            'targets': ['0.0', '0.3', '0.6'],  # Different grays for each target
            'background': '0.9'
        }
        
        # Different markers for each target
        self.target_markers = ['*', 'P', 'X']  # Star, plus, X markers
    
    def setup_visualization(self, dataloader):
        """Setup t-SNE transformation and initial state (call once at start)"""
        print("Setting up t-SNE visualization...")
        
        # Extract initial embeddings for visualization subset only
        from utils.utils import extract_embeddings, compute_class_centroids
        
        embeddings, labels = extract_embeddings(self.model, dataloader, self.device)
        
        # Filter to only include visualization classes
        viz_classes = self.viz_forget_classes + self.viz_retain_classes
        mask = torch.tensor([label in viz_classes for label in labels])
        
        if mask.sum() == 0:
            raise ValueError(f"No samples found for visualization classes {viz_classes}")
        
        viz_embeddings = embeddings[mask]
        viz_labels = labels[mask]
        
        print(f"Using {len(viz_embeddings)} samples from {len(viz_classes)} classes for visualization")
        
        # Store initial embeddings for reference
        self.initial_embeddings = viz_embeddings.cpu().numpy()
        
        # Fit t-SNE on initial embeddings
        print("Fitting t-SNE (this may take a moment)...")
        try:
            # Try newer scikit-learn parameter name
            self.tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(viz_embeddings)//4), max_iter=1000)
        except TypeError:
            # Fallback to older parameter name
            self.tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(viz_embeddings)//4), n_iter=1000)
        
        self.initial_tsne_projections = self.tsne.fit_transform(self.initial_embeddings)
        
        # Setup nearest neighbors mapper for consistent projections
        self.nn_mapper = NearestNeighbors(n_neighbors=min(5, len(viz_embeddings)), algorithm='auto')
        self.nn_mapper.fit(self.initial_embeddings)
        
        # Set fixed axis limits based on initial projection
        x_min, x_max = self.initial_tsne_projections[:, 0].min(), self.initial_tsne_projections[:, 0].max()
        y_min, y_max = self.initial_tsne_projections[:, 1].min(), self.initial_tsne_projections[:, 1].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.2
        y_padding = (y_max - y_min) * 0.2
        
        self.axis_limits = {
            'x': (x_min - x_padding, x_max + x_padding),
            'y': (y_min - y_padding, y_max + y_padding)
        }
        
        # Compute retain class centroids in original space and project to t-SNE (only viz classes)
        retain_centroids = compute_class_centroids(viz_embeddings, viz_labels, self.viz_retain_classes)
        if retain_centroids:
            retain_centroids_np = torch.stack(list(retain_centroids.values())).cpu().numpy()
            
            # Project retain centroids to t-SNE space
            self.retain_tsne_centroids = self._transform_to_tsne(retain_centroids_np)
            
            # Create Voronoi diagram from retain centroids
            if len(self.retain_tsne_centroids) >= 3:  # Need at least 3 points for Voronoi
                # Add boundary points to ensure finite regions that fill the entire space
                x_min, x_max = self.axis_limits['x']
                y_min, y_max = self.axis_limits['y']
                
                # Add corner points far outside the axis limits to create finite regions
                boundary_margin = max(x_max - x_min, y_max - y_min) * 2
                boundary_points = np.array([
                    [x_min - boundary_margin, y_min - boundary_margin],
                    [x_max + boundary_margin, y_min - boundary_margin],
                    [x_max + boundary_margin, y_max + boundary_margin],
                    [x_min - boundary_margin, y_max + boundary_margin]
                ])
                
                # Combine retain centroids with boundary points
                all_points = np.vstack([self.retain_tsne_centroids, boundary_points])
                self.voronoi_diagram = Voronoi(all_points)
                
                # Store number of actual retain centroids (excluding boundary points)
                self.n_retain_centroids = len(self.retain_tsne_centroids)
        else:
            self.retain_tsne_centroids = np.array([]).reshape(0, 2)
        
        print(f"t-SNE visualization setup complete. Output directory: {self.output_dir}")
        print(f"Axis limits: x{self.axis_limits['x']}, y{self.axis_limits['y']}")
        print(f"Using {len(self.viz_retain_classes)} retain classes for Voronoi diagram")
    
    def _transform_to_tsne(self, new_embeddings):
        """Transform new embeddings to t-SNE space using nearest neighbor interpolation"""
        if self.nn_mapper is None:
            raise ValueError("Visualization not setup. Call setup_visualization() first.")
        
        # Find nearest neighbors in original embedding space
        distances, indices = self.nn_mapper.kneighbors(new_embeddings)
        
        # Interpolate t-SNE positions using inverse distance weighting
        tsne_projections = np.zeros((len(new_embeddings), 2))
        
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            # Avoid division by zero
            weights = 1.0 / (dists + 1e-8)
            weights = weights / weights.sum()
            
            # Weighted average of nearest neighbors' t-SNE positions
            tsne_projections[i] = np.average(self.initial_tsne_projections[idxs], weights=weights, axis=0)
        
        return tsne_projections
    
    def create_visualization(self, step, embeddings, labels, target_assignment=None):
        """Create t-SNE Voronoi visualization for current step"""
        if self.tsne is None:
            raise ValueError("Visualization not setup. Call setup_visualization() first.")
        
        # Convert labels to tensor if needed
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        
        # Filter to only include visualization classes
        viz_classes = self.viz_forget_classes + self.viz_retain_classes
        mask = torch.tensor([label in viz_classes for label in labels])
        
        if mask.sum() == 0:
            print(f"Warning: No samples found for visualization classes {viz_classes} at step {step}")
            return None
        
        viz_embeddings = embeddings[mask]
        viz_labels = labels[mask]
        
        # Transform current embeddings to t-SNE space
        current_embeddings_np = viz_embeddings.cpu().numpy()
        current_tsne_projections = self._transform_to_tsne(current_embeddings_np)
        
        # Compute current centroids (only for viz classes)
        from method.utils import compute_class_centroids
        
        forget_centroids = compute_class_centroids(viz_embeddings, viz_labels, self.viz_forget_classes)
        retain_centroids = compute_class_centroids(viz_embeddings, viz_labels, self.viz_retain_classes)
        
        # Project centroids to t-SNE space
        if forget_centroids:
            forget_centroids_np = torch.stack(list(forget_centroids.values())).cpu().numpy()
            forget_tsne_centroids = self._transform_to_tsne(forget_centroids_np)
        else:
            forget_tsne_centroids = np.array([]).reshape(0, 2)
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Set background color
        bg_color = self.gray_colors['background'] if not self.use_color else self.colors['background']
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Draw Voronoi diagram showing ALL class regions (retain + forget)
        if len(forget_tsne_centroids) > 0 and len(self.retain_tsne_centroids) > 0:
            # Create combined Voronoi diagram with both retain and forget centroids
            all_centroids = np.vstack([self.retain_tsne_centroids, forget_tsne_centroids])
            
            # Add boundary points to ensure finite regions
            x_min, x_max = self.axis_limits['x']
            y_min, y_max = self.axis_limits['y']
            boundary_margin = max(x_max - x_min, y_max - y_min) * 1.5
            boundary_points = np.array([
                [x_min - boundary_margin, y_min - boundary_margin],
                [x_max + boundary_margin, y_min - boundary_margin], 
                [x_max + boundary_margin, y_max + boundary_margin],
                [x_min - boundary_margin, y_max + boundary_margin]
            ])
            
            combined_points = np.vstack([all_centroids, boundary_points])
            current_voronoi = Voronoi(combined_points)
            
            self._draw_combined_voronoi_regions(ax, current_voronoi, len(self.retain_tsne_centroids), len(forget_tsne_centroids))
        elif self.voronoi_diagram is not None:
            # Fallback to retain-only regions
            self._draw_voronoi_regions(ax)
        
        # Plot retain centroids (fixed positions)
        if len(self.retain_tsne_centroids) > 0:
            colors = self.colors['retain'] if self.use_color else self.gray_colors['retain']
            for i, (class_id, centroid) in enumerate(zip(self.viz_retain_classes, self.retain_tsne_centroids)):
                color = colors[i % len(colors)]
                ax.scatter(centroid[0], centroid[1], 
                          c=color, s=200, marker='o', 
                          edgecolors='black', linewidths=2,
                          label=f'R{class_id}' if i < 3 else "")
                ax.annotate(f'R{class_id}', (centroid[0], centroid[1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Plot forget centroids (moving)
        if len(forget_tsne_centroids) > 0:
            colors = self.colors['forget'] if self.use_color else self.gray_colors['forget']
            for i, (class_id, centroid) in enumerate(zip(self.viz_forget_classes, forget_tsne_centroids)):
                color = colors[i % len(colors)]
                ax.scatter(centroid[0], centroid[1], 
                          c=color, s=200, marker='s', 
                          edgecolors='black', linewidths=2,
                          label=f'F{class_id}' if i < 3 else "")
                ax.annotate(f'F{class_id}', (centroid[0], centroid[1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Plot target vertices and arrows (only for visualization classes)
        if target_assignment is not None and len(forget_tsne_centroids) > 0:
            target_colors = self.colors['targets'] if self.use_color else self.gray_colors['targets']
            
            target_idx = 0
            for class_id, target_vertex in target_assignment.items():
                if class_id in self.viz_forget_classes:
                    # Get unique color and marker for this target
                    color = target_colors[target_idx % len(target_colors)]
                    marker = self.target_markers[target_idx % len(self.target_markers)]
                    
                    # Transform target vertex to t-SNE space
                    target_np = target_vertex.cpu().numpy().reshape(1, -1)
                    target_tsne = self._transform_to_tsne(target_np)[0]
                    
                    # Plot target vertex with unique marker
                    ax.scatter(target_tsne[0], target_tsne[1], 
                              c=color, s=300, marker=marker, 
                              edgecolors='black', linewidths=2,
                              label=f'Target{target_idx+1}' if target_idx < 3 else "")
                    ax.annotate(f'T{class_id}', (target_tsne[0], target_tsne[1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
                    
                    # Draw arrow from forget centroid to target
                    forget_idx = self.viz_forget_classes.index(class_id)
                    if forget_idx < len(forget_tsne_centroids):
                        forget_pos = forget_tsne_centroids[forget_idx]
                        # Use matching arrow color
                        arrow_color = color if self.use_color else 'black'
                        ax.annotate('', xy=target_tsne, xytext=forget_pos,
                                   arrowprops=dict(arrowstyle='->', lw=2, color=arrow_color))
                    
                    target_idx += 1
        
        # Formatting
        ax.set_xlim(self.axis_limits['x'])
        ax.set_ylim(self.axis_limits['y'])
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f'Voronoi Unlearning Visualization - Step {step}', fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        filename = f"step_{step:06d}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close to free memory
        
        return filepath
    
    def _draw_combined_voronoi_regions(self, ax, voronoi_diagram, n_retain, n_forget):
        """Draw Voronoi regions for both retain and forget classes"""
        # Draw ridge lines
        for ridge_points, ridge_vertices in zip(voronoi_diagram.ridge_points, voronoi_diagram.ridge_vertices):
            ridge_vertices = np.asarray(ridge_vertices)
            if np.all(ridge_vertices >= 0):  # Finite ridge
                vertices = voronoi_diagram.vertices[ridge_vertices]
                ax.plot(vertices[:, 0], vertices[:, 1], 'k-', alpha=0.3, linewidth=0.8)
        
        # Fill regions for all classes
        x_min, x_max = self.axis_limits['x'] 
        y_min, y_max = self.axis_limits['y']
        
        total_centroids = n_retain + n_forget
        
        for point_idx in range(total_centroids):
            # Determine if this is a retain or forget class
            if point_idx < n_retain:
                # Retain class
                colors = self.colors['retain'] if self.use_color else self.gray_colors['retain']
                color = colors[point_idx % len(colors)]
                alpha = 0.3
            else:
                # Forget class
                forget_idx = point_idx - n_retain
                colors = self.colors['forget'] if self.use_color else self.gray_colors['forget'] 
                color = colors[forget_idx % len(colors)]
                alpha = 0.4  # Slightly more opaque for forget regions
            
            # Find region for this point
            region = voronoi_diagram.regions[voronoi_diagram.point_region[point_idx]]
            
            if len(region) > 0 and -1 not in region:  # Valid finite region
                vertices = voronoi_diagram.vertices[region]
                
                # Properly clip polygon to visualization bounds
                from matplotlib.path import Path
                import matplotlib.patches as patches
                
                # Create clipping rectangle
                clip_rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                            linewidth=0, edgecolor=None, facecolor=None)
                
                # Create polygon from vertices
                if len(vertices) >= 3:
                    # Sort vertices by angle for proper polygon
                    centroid = vertices.mean(axis=0)
                    angles = np.arctan2(vertices[:, 1] - centroid[1], 
                                      vertices[:, 0] - centroid[0])
                    sorted_indices = np.argsort(angles)
                    sorted_vertices = vertices[sorted_indices]
                    
                    # Create polygon and clip it to bounds
                    polygon = patches.Polygon(sorted_vertices, closed=True, 
                                            facecolor=color, alpha=alpha, 
                                            edgecolor='black', linewidth=0.3)
                    
                    # Apply clipping
                    polygon.set_clip_box(ax.bbox)
                    ax.add_patch(polygon)

    def _draw_voronoi_regions(self, ax):
        """Draw Voronoi regions on the plot (retain classes only)"""
        if self.voronoi_diagram is None:
            return
        
        # Draw Voronoi regions
        for ridge_points, ridge_vertices in zip(self.voronoi_diagram.ridge_points, self.voronoi_diagram.ridge_vertices):
            ridge_vertices = np.asarray(ridge_vertices)
            if np.all(ridge_vertices >= 0):  # Finite ridge
                vertices = self.voronoi_diagram.vertices[ridge_vertices]
                ax.plot(vertices[:, 0], vertices[:, 1], 'k-', alpha=0.5, linewidth=1)
        
        # Fill Voronoi regions with colors (only for retain classes)
        colors = self.colors['retain'] if self.use_color else self.gray_colors['retain']
        
        for i in range(min(len(self.retain_tsne_centroids), getattr(self, 'n_retain_centroids', len(self.retain_tsne_centroids)))):
            color = colors[i % len(colors)]
            
            # Find region for this retain class
            region = self.voronoi_diagram.regions[self.voronoi_diagram.point_region[i]]
            
            if len(region) > 0 and -1 not in region:  # Valid finite region
                vertices = self.voronoi_diagram.vertices[region]
                
                # Clip to axis limits
                x_min, x_max = self.axis_limits['x']
                y_min, y_max = self.axis_limits['y']
                
                # Clip vertices to bounds
                clipped_vertices = []
                for vertex in vertices:
                    x_clipped = np.clip(vertex[0], x_min, x_max)
                    y_clipped = np.clip(vertex[1], y_min, y_max)
                    clipped_vertices.append([x_clipped, y_clipped])
                
                if len(clipped_vertices) >= 3:
                    clipped_vertices = np.array(clipped_vertices)
                    
                    # Sort vertices by angle
                    centroid = clipped_vertices.mean(axis=0)
                    angles = np.arctan2(clipped_vertices[:, 1] - centroid[1], 
                                      clipped_vertices[:, 0] - centroid[0])
                    sorted_indices = np.argsort(angles)
                    sorted_vertices = clipped_vertices[sorted_indices]
                    
                    # Fill the region
                    alpha = 0.3 if self.use_color else 0.5
                    ax.fill(sorted_vertices[:, 0], sorted_vertices[:, 1], 
                           color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
    
    def create_animation_script(self, fps=2):
        """Create a script to generate animation from saved frames"""
        script_path = os.path.join(self.output_dir, "create_animation.sh")
        
        script_content = f"""#!/bin/bash
# Create animation from t-SNE visualization frames

echo "Creating animation from frames..."

# Using ffmpeg to create MP4 animation
ffmpeg -y -framerate {fps} -pattern_type glob -i "{self.output_dir}/step_*.png" \\
       -c:v libx264 -pix_fmt yuv420p -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \\
       "{self.output_dir}/voronoi_unlearning_animation.mp4"

# Create GIF version (optional)
ffmpeg -y -framerate {fps} -pattern_type glob -i "{self.output_dir}/step_*.png" \\
       -vf "scale=800:600:force_original_aspect_ratio=decrease,pad=800:600:(ow-iw)/2:(oh-ih)/2,palettegen" \\
       "{self.output_dir}/palette.png"

ffmpeg -y -framerate {fps} -pattern_type glob -i "{self.output_dir}/step_*.png" \\
       -i "{self.output_dir}/palette.png" \\
       -filter_complex "scale=800:600:force_original_aspect_ratio=decrease,pad=800:600:(ow-iw)/2:(oh-ih)/2[x];[x][1:v]paletteuse" \\
       "{self.output_dir}/voronoi_unlearning_animation.gif"

echo "Animation created: {self.output_dir}/voronoi_unlearning_animation.mp4"
echo "GIF created: {self.output_dir}/voronoi_unlearning_animation.gif"
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"Animation script created: {script_path}")
        print(f"Run with: bash {script_path}")
        
        return script_path