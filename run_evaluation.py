"""
Simple evaluation script for trained CPC model
All evaluation logic is in eval.py
"""
import torch
from eval import evaluate_checkpoint

# =============================================================================
# Configuration
# =============================================================================

# Paths
CHECKPOINT_PATH = "/root/results/ckpts/best-v2.ckpt"
DATA_FOLDER = "/root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/SEGLVIK"
CONFIG_PATH = "configs/cpc_config.yaml"

# Evaluation settings
SPLIT = "val"  # or "test"
BATCH_SIZE = 64
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
        data_folder=DATA_FOLDER,
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
