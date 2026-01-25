#!/usr/bin/env python3
"""
Visualize embeddings using t-SNE
Simple script to visualize CPC learned representations

Sample run:
python visualize_tsne.py \
    --checkpoint /root/results/ckpts_small/best.ckpt \
    --config configs/cpc_config.yaml \
    --dataset_folders /root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/MEDITERRANEAN_FIN_WHALE \
    --output /root/vis/tsne_mediterranean.png 

python visualize_tsne.py \
    --checkpoint /root/results/ckpts_small/best-v42.ckpt \
    --config configs/cpc_config.yaml \
    --dataset_folders /root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/CARABBEAN_HUMPBACK_WHALE \
    --output /root/vis/tsne_caribbean.png 

python visualize_tsne.py \
    --checkpoint /root/results/ckpts_small/best-v42.ckpt \
    --config configs/cpc_config.yaml \
    --dataset_folders /root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/MEDITERRANEAN_FIN_WHALE /root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/CARABBEAN_HUMPBACK_WHALE \
    --output /root/vis/tsne_both.png 
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

# Import from eval.py - reuse tested functions!
from eval import load_model_from_checkpoint, extract_embeddings
from dataloader import create_annotation_dataloaders


def compute_tsne(embeddings, perplexity=30, learning_rate=200, n_iter=1000):
    """Compute t-SNE projection"""
    print("\n" + "="*80)
    print("Computing t-SNE")
    print("="*80)
    print(f"Parameters:")
    print(f"  Perplexity: {perplexity}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iterations: {n_iter}")
    
    # Handle sklearn version differences: n_iter (old) vs max_iter (new)
    # New sklearn (>=1.0) uses max_iter instead of n_iter
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,  # Use max_iter (works with sklearn >= 1.0)
        random_state=42,
        verbose=1
    )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print(f"✓ t-SNE completed")
    return embeddings_2d


def plot_tsne(embeddings_2d, labels, save_path, title=None):
    """Plot t-SNE visualization"""
    print("\n" + "="*80)
    print("Creating Visualization")
    print("="*80)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors and labels
    colors = ['#FF6B6B', '#4ECDC4']  # Red for negative, Teal for positive
    label_names = ['No Whale Call', 'Whale Call']
    
    # Plot each class
    for label_idx in [0, 1]:
        mask = labels == label_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[label_idx],
            label=label_names[label_idx],
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Styling
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    if title is None:
        title = f't-SNE Visualization of CPC Embeddings (n={len(labels)})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✓ Visualization saved to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize CPC embeddings using t-SNE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Basic usage (uses config's val_data_folder)
        python visualize_tsne.py --checkpoint results/cpc_finwhale_smalldata/best.ckpt
        
        # Single dataset
        python visualize_tsne.py --checkpoint best.ckpt --dataset_folders RESOURCES/SEGLVIK
        
        # Multiple datasets (combined)
        python visualize_tsne.py --checkpoint best.ckpt --dataset_folders RESOURCES/SEGLVIK RESOURCES/MEDITERRANEAN_FIN_WHALE
        
        # With custom output path
        python visualize_tsne.py --checkpoint best.ckpt --output my_tsne.png
        
        # More samples and custom t-SNE parameters
        python visualize_tsne.py --checkpoint best.ckpt --max_samples 3000 --perplexity 50
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/cpc_config.yaml',
        help='Path to config file (default: configs/cpc_config.yaml)'
    )
    parser.add_argument(
        '--dataset_folders',
        type=str,
        nargs='+',
        default=None,
        help='Path(s) to dataset folder(s) (default: use val_data_path from config). '
             'Can specify multiple: --dataset_folders RESOURCES/SEGLVIK RESOURCES/MEDITERRANEAN_FIN_WHALE'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tsne_visualization.png',
        help='Output path for visualization (default: tsne_visualization.png)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=2000,
        help='Maximum number of samples to visualize (default: 2000)'
    )
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter (default: 30)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=200,
        help='t-SNE learning rate (default: 200)'
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1000,
        help='t-SNE max iterations (default: 1000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("t-SNE VISUALIZATION OF CPC EMBEDDINGS")
    print("="*80)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model using eval.py function
    print("\n[1/3] Loading model...")
    module, cfg = load_model_from_checkpoint(args.checkpoint, args.config, device)
    print(f"✓ Model loaded on {device}")
    
    # Determine dataset folders
    if args.dataset_folders:
        dataset_folders = args.dataset_folders
    else:
        # Use config value (could be single or list)
        dataset_folders = cfg.data.val_data_folder
    
    # Create dataloader using annotation-based dataset
    print(f"\n[2/3] Creating annotation-based dataloader...")
    if isinstance(dataset_folders, list):
        print(f"Dataset folders: {len(dataset_folders)} datasets")
        for folder in dataset_folders:
            print(f"  - {folder}")
    else:
        print(f"Dataset folder: {dataset_folders}")
    
    dataloader = create_annotation_dataloaders(
        dataset_folders=dataset_folders,
        split='val',
        window_duration_sec=cfg.data.window_duration_sec,
        sample_rate=cfg.sample_rate,
        batch_size=64,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
        shuffle=False,
        drop_last=False,
    )
    
    # Extract embeddings using eval.py function
    print(f"\n[3/3] Extracting embeddings...")
    X, y, groups = extract_embeddings(module, dataloader, device, verbose=True)
    
    # Limit to max_samples
    if len(X) > args.max_samples:
        print(f"Limiting to {args.max_samples} samples (from {len(X)})")
        indices = np.random.RandomState(42).choice(len(X), args.max_samples, replace=False)
        embeddings = X[indices]
        labels = y[indices]
    else:
        embeddings = X
        labels = y
    
    print(f"✓ Using {len(embeddings)} embeddings for visualization")
    
    # Compute t-SNE
    embeddings_2d = compute_tsne(
        embeddings,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter
    )
    
    # Plot and save
    fig = plot_tsne(embeddings_2d, labels, args.output)
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\nVisualization saved to: {args.output}")
    print(f"Total samples visualized: {len(labels)}")
    print(f"  - Whale calls: {np.sum(labels == 1)}")
    print(f"  - No whale calls: {np.sum(labels == 0)}")
    
    # Optional: show plot
    try:
        plt.show()
    except:
        print("\n(Unable to display plot, but saved to file)")


if __name__ == '__main__':
    main()
