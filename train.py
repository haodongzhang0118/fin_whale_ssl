"""
Training script for CPC on fin-whale audio
Using stable-pretraining framework
"""
import torch.nn as nn
import torchmetrics
import lightning as pl
import stable_pretraining as spt
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from backbone import cpc_backbone, cpc_backbone_transformer
from loss import CPCLoss
from forward import cpc_forward
from dataloader import create_dataloaders, create_supervised_dataloaders
from SklearnOfflineProbe import SklearnOfflineProbe
from OfflineKNN import OfflineKNN


def create_cpc_module(cfg):
    """
    Create CPC module for stable-pretraining
    
    Args:
        cfg: OmegaConf configuration object
        
    Returns:
        spt.Module instance configured for CPC
    """
    # Create backbone based on backbone_type
    backbone_type = cfg.get('backbone_type', 'gru')
    print(f"Creating CPC module with {backbone_type.upper()} backbone...")
    
    if backbone_type == 'transformer':
        backbone = cpc_backbone_transformer(
            sample_rate=cfg.model.sample_rate,
            enc_hidden=cfg.model.enc_hidden,
            transformer_hidden=cfg.model.transformer_hidden,
            sinc_channels=cfg.model.sinc_channels,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            drop_path=cfg.model.drop_path,
            attn_drop=cfg.model.attn_drop,
            proj_drop=cfg.model.proj_drop,
            pos_encoding=cfg.model.pos_encoding,
        )
        context_dim = cfg.model.transformer_hidden
    else:  # gru
        backbone = cpc_backbone(
            sample_rate=cfg.model.sample_rate,
            enc_hidden=cfg.model.enc_hidden,
            gru_hidden=cfg.model.gru_hidden,
            sinc_channels=cfg.model.sinc_channels
        )
        context_dim = cfg.model.gru_hidden
    
    # Create prediction heads (one per future time step)
    Wk = nn.ModuleList([
        nn.Linear(context_dim, cfg.model.enc_hidden) 
        for _ in range(cfg.model.timestep)
    ])
    
    # Create CPC loss
    cpc_loss = CPCLoss(
        tau=cfg.model.tau,
        normalize=cfg.model.normalize
    )
    
    # Create stable-pretraining Module
    module = spt.Module(
        backbone=backbone,
        forward=cpc_forward,
        Wk=Wk,
        cpc_loss=cpc_loss,
        timestep=cfg.model.timestep,
        optim=cfg.optim,
        hparams={"model": cfg.model},
    )
    
    return module


def create_callbacks(cfg):
    """
    Create training callbacks
    
    Args:
        cfg: Configuration object
        
    Returns:
        List of callback instances
    """
    callbacks = []
    
    # Model checkpoint
    # Saves two checkpoints:
    # 1. best.ckpt - Best model based on monitored metric
    # 2. last.ckpt - Last epoch checkpoint
    if cfg.trainer.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            dirpath=cfg.checkpoint.dirpath,
            filename=cfg.checkpoint.filename,
            verbose=True,
            auto_insert_metric_name=False,  # Don't auto-insert metric in filename
        )
        callbacks.append(checkpoint_callback)
        print(f"Checkpointing enabled:")
        print(f"  - Monitoring: {cfg.checkpoint.monitor} ({cfg.checkpoint.mode})")
        print(f"  - Save directory: {cfg.checkpoint.dirpath}")
        print(f"  - Best checkpoint: {cfg.checkpoint.filename}.ckpt")
        print(f"  - Last checkpoint: last.ckpt")
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Add evaluation probes if configured
    if cfg.get("use_probes", True):  # Default to True
        print("Adding evaluation callbacks...")
        
        # 1. Sklearn-based Linear Probe (offline with cross-validation)
        # Collects validation embeddings, evaluates with cross-validation
        sklearn_probe = SklearnOfflineProbe(
            name="sklearn_probe",
            input="embedding",           # Get embeddings from model output
            target="label",               # Get labels from batch
            n_components=cfg.get("probe_pca_components", 50),  # PCA components
            n_splits=5,                  # Cross-validation folds
        )
        callbacks.append(sklearn_probe)
        print(f"  - Added SklearnOfflineProbe (PCA: {cfg.get('probe_pca_components', 50)}, CV folds: 5)")
        
        # 2. KNN Probe (offline)
        # Collects validation embeddings during validation, computes KNN
        knn_probe = OfflineKNN(
            name="knn_probe",
            input="embedding",           # Get embeddings from model output
            target="label",               # Get labels from batch
            queue_length=cfg.get("knn_queue_length", 20000),  # Max samples to cache
            k=cfg.get("knn_k", 10),      # Number of neighbors
            temperature=cfg.get("knn_temperature", 0.07),
            distance_metric=cfg.get("knn_distance_metric", "euclidean"),
            input_dim=cfg.model.gru_hidden,  # Dimension of embeddings
            target_dim=1,                 # Binary classification
            metrics={
                "acc": torchmetrics.classification.BinaryAccuracy(),
            },
        )
        callbacks.append(knn_probe)
        print(f"  - Added OfflineKNN (k={cfg.get('knn_k', 10)}, queue={cfg.get('knn_queue_length', 20000)})")
    
    return callbacks


def create_datamodule(cfg):
    """
    Create data module for training
    
    Uses the simplified SEGLVIK dataloader for CPC training:
    - Training: Unsupervised (no labels needed)
    - Validation: Supervised (with labels for probes)
    
    Args:
        cfg: Configuration object
        
    Returns:
        spt.data.DataModule instance
    """
    # NOTE: Import your actual dataloader creation functions here
    # For now, this is a placeholder structure
    
    print(f"\n{'='*80}")
    print("Creating DataModule")
    print(f"{'='*80}")
    print(f"Train data folder: {cfg.data.train_data_folder}")
    print(f"Val data folder:   {cfg.data.val_data_folder}")
    print(f"Window duration: {cfg.data.window_duration_sec} s")
    print(f"Hop duration: {cfg.data.hop_duration_sec} s")
    print(f"Sample rate: {cfg.data.sample_rate} Hz")
    print(f"Batch size: {cfg.data.batch_size}")
    print(f"{'='*80}\n")
    
    # Create training dataloader (unsupervised)
    print("Creating training dataloader...")
    train_loader = create_dataloaders(
        data_folder=cfg.data.train_data_folder,  # Use train_data_folder
        split="train",
        window_duration_sec=cfg.data.window_duration_sec,
        hop_duration_sec=cfg.data.hop_duration_sec,
        sample_rate=cfg.data.sample_rate,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=cfg.data.get("shuffle_train", True),
        drop_last=True,
        pin_memory=True,
        seed=cfg.seed,
        data_fraction=cfg.data.get("train_data_fraction", 1.0),
    )
    
    # Create validation dataloader (supervised, with labels)
    print("\nCreating validation dataloader...")
    val_loader = create_supervised_dataloaders(
        data_folder=cfg.data.val_data_folder,  # Use val_data_folder
        split="val",
        window_duration_sec=cfg.data.window_duration_sec,
        sample_rate=cfg.data.sample_rate,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=cfg.data.get("shuffle_val", False),
        drop_last=False,
        pin_memory=True,
        seed=cfg.seed,
    )
    
    # Create DataModule
    datamodule = spt.data.DataModule(
        train=train_loader,
        val=val_loader,
        test=val_loader  # Use val as test for now
    )
    
    print(f"\n✅ DataModule created successfully!")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Val batches: {len(val_loader)}")
    print(f"{'='*80}\n")
    
    return datamodule


def create_trainer(cfg, callbacks):
    """
    Create PyTorch Lightning trainer
    
    Args:
        cfg: Configuration object
        callbacks: List of callbacks
        
    Returns:
        pl.Trainer instance
    """
    # Setup logger
    if cfg.trainer.logger == "wandb" and not cfg.get("disable_wandb", True):
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            save_dir=cfg.wandb.save_dir,
        )
    else:
        logger = CSVLogger(
            save_dir=cfg.trainer.log_dir,
            name=cfg.exp_name
        )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", 2),
        enable_checkpointing=cfg.trainer.enable_checkpointing,
    )
    
    return trainer


def main(config_path="configs/cpc_config.yaml"):
    """
    Main training function
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    cfg = OmegaConf.load(config_path)
    
    print("=" * 80)
    print("CPC Training with Stable-Pretraining")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set random seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    
    # Create components
    print("Creating module...")
    module = create_cpc_module(cfg)
    
    print("Creating datamodule...")
    datamodule = create_datamodule(cfg)
    
    print("Creating callbacks...")
    callbacks = create_callbacks(cfg)
    
    print("Creating trainer...")
    trainer = create_trainer(cfg, callbacks)
    
    # Create manager and start training
    print("Creating manager...")
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=datamodule,
        seed=cfg.get("seed"),
        ckpt_path=cfg.get("ckpt_path"),
    )
    
    print("Starting training...")
    manager()
    
    print("Training completed!")


if __name__ == "__main__":
    import sys
    
    # Allow config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/cpc_config.yaml"
    main(config_path)
