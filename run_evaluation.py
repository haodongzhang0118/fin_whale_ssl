"""
Simple evaluation script for trained CPC model
All evaluation logic is in eval.py

Examples:

# Single dataset (Mediterranean, all splits)
python run_evaluation.py

# Change to Caribbean
DATASET_FOLDERS = "/root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/CARABBEAN_HUMPBACK_WHALE"

# Multiple datasets (combine Mediterranean + Caribbean)
DATASET_FOLDERS = [
    "/root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/MEDITERRANEAN_FIN_WHALE",
    "/root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/CARABBEAN_HUMPBACK_WHALE"
]
"""
import torch
from eval import evaluate_checkpoint

# =============================================================================
# Configuration
# =============================================================================

# Paths
CHECKPOINT_PATH = "/root/results/current/short-best.ckpt"
DATASET_FOLDERS = "/root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/CARABBEAN_HUMPBACK_WHALE"  # Single dataset
CONFIG_PATH = "configs/cpc_config.yaml"

# Evaluation settings
SPLIT = "all"  # Options: 'train', 'val', 'test', 'both' (train+val), 'all' (train+val+test)
BATCH_SIZE = 512
NUM_WORKERS = 4
SAMPLE_RATE = 16000
WINDOW_DURATION = 8.0

# Sklearn settings
N_SPLITS = 5  # CV folds
N_COMPONENTS = 50  # PCA components
SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Run Evaluation
# =============================================================================

if __name__ == "__main__":
    metrics = evaluate_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
        dataset_folders=DATASET_FOLDERS,  # ✅ Changed from data_folder
        split=SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sample_rate=SAMPLE_RATE,
        window_duration=WINDOW_DURATION,
        n_splits=N_SPLITS,
        n_components=N_COMPONENTS,
        seed=SEED,
        device=DEVICE,
    )
