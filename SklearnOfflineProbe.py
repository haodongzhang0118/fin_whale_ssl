import numpy as np
from lightning.pytorch import Callback
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from stable_pretraining.utils import get_data_from_batch_or_outputs
from loguru import logger as logging


class SklearnOfflineProbe(Callback):
    """
    Sklearn-based probe that evaluates on validation data using cross-validation.
    Uses the same evaluation method as eval.py for consistency.
    """
    
    def __init__(self, name, input, target, n_components=50, n_splits=5):
        super().__init__()
        self.name = name
        self.input = input
        self.target = target
        self.n_components = n_components
        self.n_splits = n_splits
        
        # Create sklearn pipeline
        self.pipeline = make_pipeline(
            Normalizer(),
            PCA(n_components=n_components, whiten=True, random_state=42),
            LogisticRegression(max_iter=2000)
        )
        
        # Buffers for collecting data
        self.X_val = []
        self.y_val = []
        self.groups_val = []
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect embeddings, labels, and groups from validation"""
        x = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        y = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.name
        )
        
        if x is None or y is None:
            return
        
        # Convert to numpy and store
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().squeeze()
        
        self.X_val.append(x_np)
        self.y_val.append(y_np)
        
        # Collect groups (file names) for stratified cross-validation
        if "file_name" in batch:
            self.groups_val += list(batch["file_name"])
        else:
            # Fallback: use batch indices as groups
            self.groups_val += [f"batch_{batch_idx}_sample_{i}" for i in range(len(x_np))]
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Evaluate using cross-validation (same as eval.py)"""
        # Skip evaluation during sanity check
        if trainer.sanity_checking:
            logging.info(f"{self.name}: Skipping evaluation during sanity check")
            self.X_val = []
            self.y_val = []
            self.groups_val = []
            return
        
        if len(self.X_val) == 0:
            logging.warning(f"{self.name}: No validation data collected")
            return
        
        # Concatenate all batches
        X = np.vstack(self.X_val)
        y = np.concatenate(self.y_val)
        groups = np.array(self.groups_val)
        
        logging.info(f"{self.name}: Evaluating with {self.n_splits}-fold CV on {X.shape[0]} samples")
        
        # Use StratifiedGroupKFold for proper cross-validation
        cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # Compute metrics with cross-validation (same as eval.py)
        try:
            acc = cross_val_score(self.pipeline, X, y, groups=groups, cv=cv, scoring="accuracy").mean()
            precision = cross_val_score(self.pipeline, X, y, groups=groups, cv=cv, scoring="precision").mean()
            recall = cross_val_score(self.pipeline, X, y, groups=groups, cv=cv, scoring="recall").mean()
            f1 = cross_val_score(self.pipeline, X, y, groups=groups, cv=cv, scoring="f1").mean()
            auroc = cross_val_score(self.pipeline, X, y, groups=groups, cv=cv, scoring="roc_auc").mean()
            
            metrics = {
                f"eval/{self.name}_acc": acc,
                f"eval/{self.name}_precision": precision,
                f"eval/{self.name}_recall": recall,
                f"eval/{self.name}_f1": f1,
                f"eval/{self.name}_auroc": auroc,
            }
            
            logging.info(f"{self.name} Results: Acc={acc:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}")
            
        except Exception as e:
            logging.error(f"{self.name}: Cross-validation failed: {e}")
            metrics = {}
        
        # Log metrics
        if metrics:
            pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        # Clear buffers
        self.X_val = []
        self.y_val = []
        self.groups_val = []