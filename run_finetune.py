"""
Quick fine-tuning experiments
Easily compare different strategies
"""
import torch
from finetune import fine_tune

# =============================================================================
# Configuration
# =============================================================================

CHECKPOINT_PATH = "/root/results/current/short-best.ckpt"
CONFIG_PATH = "configs/cpc_config.yaml"

# Dataset selection
MEDITERRANEAN = "/root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/MEDITERRANEAN_FIN_WHALE"
CARIBBEAN = "/root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/CARABBEAN_HUMPBACK_WHALE"

# Choose dataset
DATASET = MEDITERRANEAN  # or CARIBBEAN

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Experiment configurations
# =============================================================================

EXPERIMENTS = {
    # Fast baseline: Frozen backbone + Linear head
    "frozen_linear": {
        "freeze_backbone": True,
        "head_type": "linear",
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "output_dir": "results/finetune/frozen_linear",
    },
    
    # Frozen backbone + MLP head
    "frozen_mlp": {
        "freeze_backbone": True,
        "head_type": "mlp",
        "hidden_dim": 256,
        "dropout": 0.3,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "output_dir": "results/finetune/frozen_mlp",
    },
    
    # Full fine-tuning with Linear head
    "full_linear": {
        "freeze_backbone": False,
        "head_type": "linear",
        "num_epochs": 100,
        "learning_rate": 1e-4,  # Lower LR for full fine-tuning
        "output_dir": "results/finetune/full_linear",
    },
    
    # Full fine-tuning with MLP head
    "full_mlp": {
        "freeze_backbone": False,
        "head_type": "mlp",
        "hidden_dim": 256,
        "dropout": 0.3,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "output_dir": "results/finetune/full_mlp",
    },
}

# =============================================================================
# Run experiments
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fine-tuning experiments")
    parser.add_argument(
        '--exp',
        type=str,
        default='frozen_linear',
        choices=list(EXPERIMENTS.keys()),
        help='Experiment to run'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=MEDITERRANEAN,
        help='Dataset path'
    )
    
    args = parser.parse_args()
    
    # Get experiment config
    exp_config = EXPERIMENTS[args.exp]
    
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {args.exp.upper()}")
    print("=" * 80)
    print(f"Dataset: {args.dataset.split('/')[-1]}")
    print(f"Strategy: {'Frozen' if exp_config['freeze_backbone'] else 'Full fine-tuning'}")
    print(f"Head: {exp_config['head_type'].upper()}")
    print("=" * 80 + "\n")
    
    # Run fine-tuning
    history, best_val_f1, test_metrics = fine_tune(
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
        dataset_folders=args.dataset,
        batch_size=64,
        weight_decay=1e-4,
        seed=42,
        device=DEVICE,
        **exp_config
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"EXPERIMENT {args.exp.upper()} COMPLETED")
    print("=" * 80)
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Results saved to: {exp_config['output_dir']}")
    print("=" * 80 + "\n")
