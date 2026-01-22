"""
Custom KNN callback that only collects queue data during validation, not training.

This allows using unsupervised training data without labels while still
building a KNN index from labeled validation data.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from torch import Tensor

from stable_pretraining.utils import OrderedQueue, get_data_from_batch_or_outputs
from stable_pretraining.utils.distance_metrics import compute_pairwise_distances_chunked
from stable_pretraining.callbacks.utils import format_metrics_as_dict


class OfflineQueue(Callback):
    """
    Queue callback that ONLY collects data during validation, not training.
    
    This is useful when you want to build a queue from labeled validation data
    without trying to collect from unlabeled training data.
    """

    # Class-level registry to track shared queues by key
    _shared_queues: Dict[str, "OrderedQueue"] = {}
    _queue_info: Dict[str, dict] = {}

    def __init__(
        self,
        key: str,
        queue_length: int,
        dim: Optional[Union[int, tuple]] = None,
        dtype: Optional[torch.dtype] = None,
        gather_distributed: bool = False,
    ) -> None:
        super().__init__()

        self.key = key
        self.requested_length = queue_length
        self.dim = dim
        self.dtype = dtype
        self.gather_distributed = gather_distributed
        self._snapshot = None

        logging.info(f"OfflineQueue initialized for key '{key}' (validation only)")
        logging.info(f"\t- requested_length: {queue_length}")
        logging.info(f"\t- dim: {dim}")
        logging.info(f"\t- dtype: {dtype}")

    @property
    def actual_queue_length(self) -> int:
        """Get the actual size of the underlying shared queue."""
        if self.key in self._queue_info:
            return self._queue_info[self.key]["max_length"]
        return self.requested_length

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize or connect to shared queue during setup phase."""
        if self.key not in self._queue_info:
            self._queue_info[self.key] = {
                "max_length": self.requested_length,
                "dim": self.dim,
                "dtype": self.dtype,
                "callbacks": [self],
            }
            logging.info(
                f"OfflineQueue: New key '{self.key}' with initial size {self.requested_length}"
            )
        else:
            old_max = self._queue_info[self.key]["max_length"]
            if self.requested_length > old_max:
                self._queue_info[self.key]["max_length"] = self.requested_length
                logging.info(
                    f"OfflineQueue: Increased max size for key '{self.key}' "
                    f"from {old_max} to {self.requested_length}"
                )

            if self not in self._queue_info[self.key]["callbacks"]:
                self._queue_info[self.key]["callbacks"].append(self)

        max_length = self._queue_info[self.key]["max_length"]

        if self.key not in self._shared_queues:
            self._shared_queues[self.key] = OrderedQueue(
                max_length, self.dim, self.dtype
            )
            queue_key = f"offline_queue_{self.key}"
            pl_module.callbacks_modules[queue_key] = self._shared_queues[self.key]
            logging.info(
                f"OfflineQueue: Created shared queue for '{self.key}' with size {max_length}"
            )
        elif self._shared_queues[self.key].max_length < max_length:
            old_queue = self._shared_queues[self.key]
            old_data = (
                old_queue.get() if old_queue.pointer > 0 or old_queue.filled else None
            )

            new_queue = OrderedQueue(max_length, self.dim, self.dtype)

            if old_data is not None and len(old_data) > 0:
                new_queue.append(old_data)
                logging.info(
                    f"OfflineQueue: Resized queue for '{self.key}' from "
                    f"{old_queue.max_length} to {max_length}, preserved {len(old_data)} items"
                )

            self._shared_queues[self.key] = new_queue
            queue_key = f"offline_queue_{self.key}"
            pl_module.callbacks_modules[queue_key] = new_queue

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Collect data during validation instead of training.
        This is the key difference from OnlineQueue.
        """
        # Only the first callback for each key should append data
        if self._queue_info[self.key]["callbacks"][0] is not self:
            return

        with torch.no_grad():
            data = get_data_from_batch_or_outputs(
                self.key, batch, outputs, caller_name="OfflineQueue"
            )
            if data is None:
                return

            if isinstance(self.dim, int) and data.dim() == 1:
                data = data.unsqueeze(1)

            # Append to the shared queue
            self._shared_queues[self.key].append(data)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Clear the queue at the start of validation to collect fresh data."""
        logging.info(
            f"OfflineQueue: Clearing queue for key '{self.key}' at validation start"
        )
        # Reset the queue to collect fresh validation data
        self._shared_queues[self.key] = OrderedQueue(
            self.actual_queue_length, self.dim, self.dtype
        )
        queue_key = f"offline_queue_{self.key}"
        if hasattr(pl_module, "callbacks_modules"):
            pl_module.callbacks_modules[queue_key] = self._shared_queues[self.key]

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Create snapshot after validation is complete."""
        logging.info(
            f"OfflineQueue: Creating snapshot for key '{self.key}' "
            f"(requesting {self.requested_length} from queue of size {self.actual_queue_length})"
        )

        full_queue_data = self._shared_queues[self.key].get()

        if len(full_queue_data) > self.requested_length:
            tensor = full_queue_data[-self.requested_length :]
            logging.info(
                f"\t- Extracted last {self.requested_length} items from {len(full_queue_data)} available"
            )
        else:
            tensor = full_queue_data
            if len(tensor) < self.requested_length:
                logging.info(
                    f"\t- Queue not full yet: {len(tensor)}/{self.requested_length} items"
                )

        if self.gather_distributed and trainer.world_size > 1:
            gathered = pl_module.all_gather(tensor).flatten(0, 1)
            self._snapshot = gathered
            logging.info(
                f"\t- {self.key}: {tensor.shape} -> {gathered.shape} (gathered)"
            )
        else:
            self._snapshot = tensor
            logging.info(f"\t- {self.key}: {tensor.shape}")

    @property
    def data(self) -> Optional[torch.Tensor]:
        """Get snapshot data."""
        if self._snapshot is None:
            # During validation batch processing, return current queue state
            queue_data = self._shared_queues[self.key].get()
            if len(queue_data) == 0:
                return None
            return queue_data
        return self._snapshot


class OfflineKNN(Callback):
    """
    K-Nearest Neighbors evaluator that ONLY runs during validation.
    
    Unlike OnlineKNN which collects data during training, this callback
    collects embeddings and labels only during validation. This allows
    using unlabeled training data while still performing KNN evaluation
    on labeled validation data.
    
    The KNN is computed using validation embeddings to predict validation labels,
    effectively measuring how well the learned representations cluster by class.
    
    Usage:
        >>> from core.model.OfflineKNN import OfflineKNN
        >>> 
        >>> offline_knn = OfflineKNN(
        ...     name="knn_probe",
        ...     input="embedding",
        ...     target="label",
        ...     queue_length=20000,
        ...     k=10,
        ...     metrics={
        ...         "acc": torchmetrics.classification.BinaryAccuracy(),
        ...     },
        ... )
    """

    def __init__(
        self,
        name: str,
        input: str,
        target: str,
        queue_length: int,
        metrics: Dict,
        input_dim: Optional[Union[Tuple[int, ...], List[int], int]] = None,
        target_dim: Optional[int] = None,
        k: int = 5,
        temperature: float = 0.07,
        chunk_size: int = -1,
        distance_metric: Literal[
            "euclidean", "squared_euclidean", "cosine", "manhattan"
        ] = "euclidean",
    ) -> None:
        super().__init__()

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if chunk_size == 0 or chunk_size < -1:
            raise ValueError(f"chunk_size must be positive or -1, got {chunk_size}")

        if input_dim is not None and isinstance(input_dim, (list, tuple)):
            input_dim = int(np.prod(input_dim))

        self.name = name
        self.input = input
        self.target = target
        self.queue_length = queue_length
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.k = k
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.distance_metric = distance_metric
        self.metrics = metrics

        self._input_queue = None
        self._target_queue = None

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"OfflineKNN[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Create OfflineQueue callbacks for input and target."""
        logging.info(f"Setting up {self.state_key} callback!")
        
        if self._input_queue is None or self._target_queue is None:
            # Create OfflineQueue for input (embeddings)
            self._input_queue = OfflineQueue(
                key=self.input,
                queue_length=self.queue_length,
                dim=self.input_dim,
                dtype=torch.float32 if self.input_dim is not None else None,
                gather_distributed=True,
            )
            logging.info(f"{self.name}: Created OfflineQueue for input '{self.input}'")
            
            # Create OfflineQueue for target (labels)
            self._target_queue = OfflineQueue(
                key=self.target,
                queue_length=self.queue_length,
                dim=self.target_dim,
                dtype=torch.long if self.target_dim is not None else None,
                gather_distributed=True,
            )
            logging.info(f"{self.name}: Created OfflineQueue for target '{self.target}'")
            
            # Add queues to trainer callbacks and set them up
            trainer.callbacks.append(self._input_queue)
            trainer.callbacks.append(self._target_queue)
            
            self._input_queue.setup(trainer, pl_module, stage)
            self._target_queue.setup(trainer, pl_module, stage)
            
            # Setup metrics
            logging.info(f"{self.name}: Setting up metrics")
            pl_module.callbacks_metrics[self.name] = format_metrics_as_dict(
                self.metrics
            )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """
        Compute KNN predictions after all validation data is collected.
        This is called after validation completes and queues are full.
        """
        cached_features = self._input_queue.data
        cached_labels = self._target_queue.data

        if cached_features is None or cached_labels is None:
            logging.warning(
                f"{self.name}: Queue data not available for KNN computation"
            )
            return

        if cached_features.numel() == 0 or cached_labels.numel() == 0:
            logging.warning(
                f"{self.name}: Queue data is empty, skipping KNN computation"
            )
            return

        logging.info(
            f"{self.name}: Computing KNN with {cached_features.size(0)} samples"
        )

        # Compute KNN predictions using all validation samples
        predictions = self._compute_knn_predictions(
            cached_features, cached_features, cached_labels
        )

        if predictions is not None:
            self._log_metrics(pl_module, predictions, cached_labels)

    @torch.no_grad()
    def _compute_knn_predictions(
        self,
        features: Tensor,
        cached_features: Tensor,
        cached_labels: Tensor,
    ) -> Optional[Tensor]:
        """Compute KNN predictions."""
        batch_size = features.size(0)
        num_classes = int(cached_labels.max().item()) + 1

        predictions = torch.zeros(
            batch_size, num_classes, device=features.device, dtype=torch.float32
        )

        if cached_features.device != features.device:
            cached_features = cached_features.to(features.device)
            cached_labels = cached_labels.to(features.device)

        # Don't use k=1 to avoid self-matching
        k_actual = min(self.k + 1, cached_features.size(0))

        if cached_features.dtype != features.dtype:
            cached_features = cached_features.float()
            features = features.float()

        chunk_size = batch_size if self.chunk_size == -1 else self.chunk_size
        dist_matrix = compute_pairwise_distances_chunked(
            cached_features,
            features,
            metric=self.distance_metric,
            chunk_size=chunk_size,
        )

        # Get k+1 nearest neighbors and skip the first one (self)
        dist_weight, sim_indices = dist_matrix.topk(k=k_actual, dim=0, largest=False)
        
        # Skip the closest neighbor (which might be itself)
        if k_actual > 1:
            dist_weight = dist_weight[1:, :]
            sim_indices = sim_indices[1:, :]

        dist_weight = 1 / dist_weight.add_(self.temperature)

        labels_1d = (
            cached_labels.squeeze(-1) if cached_labels.dim() > 1 else cached_labels
        )
        selected_labels = labels_1d[sim_indices].long()
        one_hot_labels = F.one_hot(selected_labels, num_classes=num_classes)

        predictions = (dist_weight.unsqueeze(-1) * one_hot_labels).sum(0)
        
        return predictions

    def _log_metrics(
        self, pl_module: LightningModule, predictions: Tensor, targets: Tensor
    ) -> None:
        """Compute and log validation metrics."""
        # Squeeze targets from [batch_size, 1] to [batch_size] for BinaryAccuracy
        # BinaryAccuracy expects integer class indices, not one-hot vectors
        if targets.dim() > 1 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        
        # Convert weighted votes [batch_size, num_classes] to predicted classes [batch_size]
        # This converts KNN's weighted voting results to class predictions
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            predictions = predictions.argmax(dim=1)
        
        logs = {}
        for metric_name, metric in pl_module.callbacks_metrics[self.name][
            "_val"
        ].items():
            metric(predictions, targets)
            logs[f"eval/{self.name}_{metric_name}"] = metric

        pl_module.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)

