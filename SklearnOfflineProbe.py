import numpy as np
from lightning.pytorch import Callback
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from stable_pretraining.utils import get_data_from_batch_or_outputs
from loguru import logger as logging


class SklearnOfflineProbe(Callback):
    """
    Sklearn-based probe that fits on validation data.
    Collects embeddings during validation, fits sklearn model, evaluates.
    """
    
    def __init__(self, name, input, target, n_components=50):
        super().__init__()
        self.name = name
        self.input = input
        self.target = target
        self.n_components = n_components
        
        # Create sklearn pipeline
        self.pipeline = make_pipeline(
            Normalizer(),
            PCA(n_components=n_components, whiten=True, random_state=42),
            LogisticRegression(max_iter=2000)
        )
        
        # Buffers for collecting data
        self.X_val = []
        self.y_val = []
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect embeddings and labels from validation"""
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
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Fit sklearn model and evaluate"""
        if len(self.X_val) == 0:
            logging.warning(f"{self.name}: No validation data collected")
            return
        
        # Concatenate all batches
        X = np.vstack(self.X_val)
        y = np.concatenate(self.y_val)
        
        logging.info(f"{self.name}: Fitting sklearn probe on {X.shape[0]} samples")
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        
        # Evaluate
        y_pred = self.pipeline.predict(X)
        y_proba = self.pipeline.predict_proba(X)[:, 1]
        
        # Compute metrics
        metrics = {
            f"eval/{self.name}_acc": accuracy_score(y, y_pred),
            f"eval/{self.name}_precision": precision_score(y, y_pred, zero_division=0),
            f"eval/{self.name}_recall": recall_score(y, y_pred, zero_division=0),
            f"eval/{self.name}_f1": f1_score(y, y_pred, zero_division=0),
            f"eval/{self.name}_auroc": roc_auc_score(y, y_proba),
        }
        
        # Log metrics
        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        # Clear buffers
        self.X_val = []
        self.y_val = []