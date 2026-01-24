"""
Offline probe based on stable-pretraining's OnlineProbe design
Collects embeddings during validation, then trains a linear classifier
"""
from typing import Optional, Union

import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch import Callback
from loguru import logger as logging
from stable_pretraining.utils import get_data_from_batch_or_outputs, detach_tensors


class OfflineProbe(Callback):
    """Offline probe that collects validation embeddings and trains a classifier.
    
    Based on stable-pretraining's OnlineProbe but modified for offline evaluation:
    - Collects embeddings during validation batches
    - At validation end, trains a linear classifier on collected data
    - Reports metrics on held-out validation split
    
    This approach mirrors OnlineProbe but without PCA or cross-validation,
    keeping the architecture simple and aligned with stable-pretraining conventions.
    
    Args:
        name: Unique identifier for logging
        input: Key in batch/outputs dict for input features (e.g., 'embedding')
        target: Key in batch/outputs dict for target labels (e.g., 'label')
        input_dim: Dimension of input features
        num_classes: Number of output classes
        train_epochs: Number of epochs to train probe
        lr: Learning rate for probe optimizer
        weight_decay: Weight decay for regularization
        optimizer: Optimizer type ('adam', 'adamw', 'sgd', 'lars')
        metrics: Dict of torchmetrics to track
    """
    
    def __init__(
        self,
        name: str,
        input: str,
        target: str,
        input_dim: int,
        num_classes: int = 2,
        train_epochs: int = 100,
        lr: float = 0.1,
        weight_decay: float = 0.0,
        optimizer: str = 'lars',
        metrics: Optional[Union[dict, torchmetrics.Metric]] = None,
    ):
        super().__init__()
        self.name = name
        self.input = input
        self.target = target
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.train_epochs = train_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        
        # Buffers for collecting data
        self.X_buffer = []
        self.y_buffer = []
        
        # Probe model (created fresh each validation)
        self.probe = None
        
        # Setup metrics
        self.metrics = self._format_metrics(metrics)
        
        logging.info(f"Initialized {self.name} (OfflineProbe)")
        logging.info(f"  - Input: {input}")
        logging.info(f"  - Target: {target}")
        logging.info(f"  - Input dim: {input_dim}")
        logging.info(f"  - Num classes: {num_classes}")
        logging.info(f"  - Train epochs: {train_epochs}")
        logging.info(f"  - LR: {lr}")
        logging.info(f"  - Optimizer: {optimizer}")
    
    def _format_metrics(self, metrics):
        """Format metrics into standard dict structure"""
        if metrics is None:
            return {}
        elif isinstance(metrics, torchmetrics.Metric):
            return {metrics.__class__.__name__: metrics}
        elif isinstance(metrics, dict):
            return metrics
        else:
            raise ValueError("metrics must be None, Metric, or dict of Metrics")
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect embeddings and labels during validation"""
        # Get data from batch or outputs
        x = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        y = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.name
        )
        
        if x is None or y is None:
            return
        
        # Detach and move to CPU to save GPU memory
        x = detach_tensors(x).cpu()
        y = detach_tensors(y).cpu()
        
        self.X_buffer.append(x)
        self.y_buffer.append(y)
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear buffers at start of validation"""
        self.X_buffer = []
        self.y_buffer = []
        logging.info(f"{self.name}: Clearing buffer at validation start")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Train probe on collected data and evaluate"""
        # Skip during sanity check
        if trainer.sanity_checking:
            logging.info(f"{self.name}: Skipping during sanity check")
            return
        
        if len(self.X_buffer) == 0:
            logging.warning(f"{self.name}: No validation data collected")
            return
        
        # Concatenate all batches
        X = torch.cat(self.X_buffer, dim=0)  # [N, D]
        y = torch.cat(self.y_buffer, dim=0)  # [N] or [N, 1]
        
        if y.dim() > 1:
            y = y.squeeze(-1)  # [N]
        
        n_samples = X.shape[0]
        logging.info(f"{self.name}: Collected {n_samples} samples")
        
        # Move to device
        device = pl_module.device
        X = X.to(device)
        y = y.to(device).long()
        
        # Split into train/val (80/20 split)
        n_train = int(0.8 * n_samples)
        
        # Shuffle indices
        perm = torch.randperm(n_samples, device=device)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        X_train = X_shuffled[:n_train]
        y_train = y_shuffled[:n_train]
        X_val = X_shuffled[n_train:]
        y_val = y_shuffled[n_train:]
        
        logging.info(f"{self.name}: Train: {n_train}, Val: {n_samples - n_train}")
        
        # Train probe
        probe_metrics = self._train_probe(
            X_train, y_train, X_val, y_val, device
        )
        
        # Log metrics
        pl_module.log_dict(probe_metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log summary
        metric_str = ", ".join([f"{k.split('/')[-1]}={v:.4f}" 
                                for k, v in probe_metrics.items()])
        logging.info(f"{self.name} Results: {metric_str}")
        
        # Clear buffers
        self.X_buffer = []
        self.y_buffer = []
    
    def _train_probe(self, X_train, y_train, X_val, y_val, device):
        """Train linear probe using specified optimizer"""
        # Create fresh probe
        probe = nn.Linear(self.input_dim, self.num_classes, bias=True).to(device)
        
        # Initialize with Xavier
        nn.init.xavier_uniform_(probe.weight)
        nn.init.zeros_(probe.bias)
        
        # Create optimizer
        if self.optimizer_type == 'lars':
            from stable_pretraining.optim import LARS
            optimizer = LARS(
                probe.parameters(),
                lr=self.lr,
                clip_lr=True,
                eta=0.02,
                exclude_bias_n_norm=True,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                probe.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                probe.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                probe.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        # IMPORTANT: Enable gradients since we're in validation context (no_grad)
        probe.train()
        best_val_acc = 0.0
        best_state = None
        
        with torch.enable_grad():  # Re-enable gradients for probe training
            for epoch in range(self.train_epochs):
                # Forward pass
                logits = probe(X_train)
                loss = criterion(logits, y_train)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Validate every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == self.train_epochs - 1:
                    probe.eval()
                    with torch.no_grad():
                        val_logits = probe(X_val)
                        val_acc = (val_logits.argmax(dim=1) == y_val).float().mean()
                        
                        # Track best model
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                    
                    probe.train()
        
        # Load best model
        if best_state is not None:
            probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
        # Final evaluation
        probe.eval()
        with torch.no_grad():
            logits = probe(X_val)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # Compute metrics
            metrics_dict = self._compute_metrics(preds, probs, y_val)
        
        # Store probe
        self.probe = probe.cpu()
        
        return metrics_dict
    
    def _compute_metrics(self, preds, probs, targets):
        """Compute classification metrics"""
        # Basic metrics
        acc = (preds == targets).float().mean()
        
        # Confusion matrix elements
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        tn = ((preds == 0) & (targets == 0)).sum().float()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # AUROC
        if self.num_classes == 2:
            try:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(
                    targets.cpu().numpy(),
                    probs[:, 1].cpu().numpy()
                )
                auroc = torch.tensor(auroc)
            except Exception as e:
                logging.warning(f"{self.name}: Could not compute AUROC: {e}")
                auroc = torch.tensor(0.5)
        else:
            auroc = acc  # Fallback for multiclass
        
        # Build metrics dict
        metrics_dict = {
            f"eval/{self.name}_acc": acc,
            f"eval/{self.name}_precision": precision,
            f"eval/{self.name}_recall": recall,
            f"eval/{self.name}_f1": f1,
            f"eval/{self.name}_auroc": auroc,
        }
        
        # Add custom metrics if provided
        for metric_name, metric in self.metrics.items():
            metric = metric.to(preds.device)
            try:
                metric_value = metric(preds, targets)
                metrics_dict[f"eval/{self.name}_{metric_name}"] = metric_value
            except Exception as e:
                logging.warning(f"{self.name}: Could not compute {metric_name}: {e}")
        
        return metrics_dict