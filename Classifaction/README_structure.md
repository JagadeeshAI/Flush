# Project Structure Overview

## New Organized Structure

```
├── method/
│   ├── VU.py              # Main Voronoi Unlearning class (199 lines)
│   └── utils.py           # VU-specific utilities (orthogonalization, targets, losses)
│
├── utils/
│   ├── __init__.py
│   └── utils.py           # Common utilities (extract_features, embeddings, centroids)
│
├── visual/
│   ├── __init__.py
│   └── tsne_visualizer.py # Complete t-SNE Voronoi visualization system
│
└── codes/
    ├── data.py            # Data loading utilities
    └── utils.py           # Model utilities
```

## Key Files and Their Purpose

### method/VU.py (199 lines)
**Main class for Voronoi Unlearning**
- `VoronoiUnlearning` class with all core functionality
- Setup, training loop, evaluation
- Integrated visualization support
- Clean, focused implementation

### method/utils.py
**VU-specific utilities**
- `extract_classifier_weights()` - Get classifier head weights
- `compute_voronoi_vertices_from_weights()` - Generate distributed targets  
- `orthogonalize_embeddings_to_weights()` - Gram-Schmidt orthogonalization
- `compute_auxiliary_logit_constraint()` - Logit constraint loss
- `compute_forget_loss()` / `compute_retain_loss()` - Core losses
- `assign_targets_to_classes()` - Target assignment logic
- `compute_group_sparse_regularization()` - LoRA regularization

### utils/utils.py
**Common utilities (reusable across methods)**
- `extract_features()` - Extract embeddings from model
- `extract_embeddings()` - Batch embedding extraction
- `compute_class_centroids()` - Compute class centroids

### visual/tsne_visualizer.py
**Complete visualization system**
- `TSNEVoronoiVisualizer` class
- t-SNE projection with consistent transformations
- Voronoi diagram rendering for all classes
- Animation generation
- Supports 3 forget + 50 retain classes
- Color and grayscale modes

## Usage

### Basic Usage
```bash
python method/VU.py --method simple --epochs 5
```

### With Visualization
```bash
python method/VU.py --method simple --epochs 5 --enable-viz --viz-dir my_visualizations
```

### With Grayscale Visualization
```bash
python method/VU.py --method advance --epochs 10 --enable-viz --grayscale
```

## Key Features

1. **Modular Design**: Each component has a specific purpose
2. **Clean Separation**: Method logic separate from utilities and visualization
3. **Lightweight Core**: Main VU.py is under 200 lines
4. **Comprehensive Visualization**: Rich t-SNE Voronoi diagrams
5. **Easy Extension**: New methods can reuse common utilities

## Dependencies Between Files

- `method/VU.py` imports from `utils/utils.py`, `method/utils.py`, and `visual/`
- `method/utils.py` is self-contained (only standard imports)
- `utils/utils.py` is self-contained
- `visual/tsne_visualizer.py` is self-contained

This structure makes the codebase much more maintainable and allows for easy extension with new unlearning methods.