"""
Evaluation utilities for CPC models
Includes embedding extraction and sklearn-based metrics
"""
import torch
import torch.nn as nn
import numpy as np
import stable_pretraining as spt
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from backbone import cpc_backbone, cpc_backbone_transformer
from loss import CPCLoss
from forward import cpc_forward
from tqdm import tqdm


def extract_embeddings(module, loader, device, verbose=True):
    """
    Extract embeddings from CPC model
    
    Args:
        module: spt.Module with backbone attribute
        loader: DataLoader with supervised data (raw_audio, label, file_name)
        device: torch.device
        verbose: Whether to print progress
        
    Returns:
        X: Embeddings [N, D]
        y: Labels [N]
        groups: File names [N]
    """
    module.eval()
    backbone = module.backbone
    
    Xs, ys, groups = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(loader), desc="Extracting embeddings", total=len(loader)):
            # Get audio and move to device
            wav = batch["raw_audio"].to(device)
            if wav.ndim == 2:
                wav = wav.unsqueeze(1)  # [B, 1, T]
            
            # Extract embeddings using backbone's method
            emb = backbone.extract_embedding(wav)  # [B, D]
            
            # Collect
            Xs.append(emb.cpu().numpy())
            ys.append(batch["label"].cpu().numpy())
            groups += list(batch["file_name"])
    
    # Concatenate
    X = np.concatenate(Xs, axis=0)  # [N, D]
    y = np.concatenate(ys, axis=0).astype(int)  # [N]
    groups = np.array(groups)
    
    return X, y, groups


def eval_separation_metrics(X, y, groups, n_splits=5, n_components=50, seed=42):
    """
    Evaluate separation metrics using sklearn
    
    Args:
        X: Embeddings [N, D]
        y: Binary labels [N]
        groups: Group identifiers for stratification [N]
        n_splits: Number of CV folds
        n_components: PCA components
        seed: Random seed
        
    Returns:
        Dictionary of metrics
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Linear Probe (Normalizer + PCA + LR)
    pipe_lr = make_pipeline(
        Normalizer(),
        PCA(n_components=min(n_components, X.shape[1]), whiten=True, random_state=seed),
        LogisticRegression(max_iter=2000)
    )
    acc = cross_val_score(pipe_lr, X, y, groups=groups, cv=cv, scoring="accuracy").mean()
    roc = cross_val_score(pipe_lr, X, y, groups=groups, cv=cv, scoring="roc_auc").mean()
    f1  = cross_val_score(pipe_lr, X, y, groups=groups, cv=cv, scoring="f1").mean()
    precision = cross_val_score(pipe_lr, X, y, groups=groups, cv=cv, scoring="precision").mean()
    recall = cross_val_score(pipe_lr, X, y, groups=groups, cv=cv, scoring="recall").mean()

    # KNN Probe
    pipe_knn = make_pipeline(
        Normalizer(),
        PCA(n_components=min(n_components, X.shape[1]), whiten=True, random_state=seed),
        KNeighborsClassifier(n_neighbors=10, metric="cosine")
    )
    knn_acc = cross_val_score(pipe_knn, X, y, groups=groups, cv=cv, scoring="accuracy").mean()

    return {
        "Acc": acc,
        "ROC": roc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "KNN": knn_acc,
    }


def load_model_from_checkpoint(checkpoint_path, config_path, device):
    """
    Load CPC model from checkpoint
    
    Args:
        checkpoint_path: Path to .ckpt file
        config_path: Path to config.yaml
        device: torch.device
        
    Returns:
        Loaded spt.Module
    """   
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Create backbone
    backbone_type = cfg.get('backbone_type', 'gru')
    
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
    
    # Create prediction heads
    k_start = cfg.model.get("k_start", 1)  # Default: start from t+1
    num_prediction_steps = cfg.model.timestep - k_start + 1
    
    Wk = nn.ModuleList([
        nn.Linear(context_dim, cfg.model.enc_hidden) 
        for _ in range(num_prediction_steps)
    ])
    
    # Create CPC loss
    cpc_loss = CPCLoss(tau=cfg.model.tau)
    
    # Create module
    module = spt.Module(
        backbone=backbone,
        forward=cpc_forward,
        Wk=Wk,
        cpc_loss=cpc_loss,
        timestep=cfg.model.timestep,
        k_start=k_start,  # Pass k_start for evaluation
        optim=cfg.optim,
        hparams={"model": cfg.model},
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    module.load_state_dict(checkpoint['state_dict'], strict=False)
    module.eval()
    module.to(device)
    
    return module, cfg


def evaluate_checkpoint(
    checkpoint_path,
    config_path,
    dataset_folders,
    split="all",
    batch_size=128,
    num_workers=4,
    sample_rate=16000,
    window_duration=8.0,
    n_splits=5,
    n_components=50,
    seed=42,
    device=None,
):
    """
    Complete evaluation pipeline (using annotation-based dataloader)
    
    Args:
        checkpoint_path: Path to .ckpt file
        config_path: Path to config.yaml
        dataset_folders: Path(s) to dataset folder(s) (single path or list of paths)
        split: 'train', 'val', 'test', 'both', or 'all' (default: 'all' = train+val+test)
        batch_size: Batch size for data loading
        num_workers: Number of data loader workers
        sample_rate: Audio sample rate
        window_duration: Window duration in seconds
        n_splits: CV folds
        n_components: PCA components
        seed: Random seed
        device: torch.device (auto-detect if None)
        
    Returns:
        Dictionary of evaluation metrics
    """
    from dataloader import create_annotation_dataloaders
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("CPC Model Evaluation (Annotation-Based Dataloader)")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    if isinstance(dataset_folders, list):
        print(f"Dataset folders: {len(dataset_folders)} datasets")
        for folder in dataset_folders:
            print(f"  - {folder}")
    else:
        print(f"Dataset folder: {dataset_folders}")
    print(f"Split: {split}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading model from checkpoint...")
    module, cfg = load_model_from_checkpoint(checkpoint_path, config_path, device)
    backbone_type = cfg.get('backbone_type', 'gru')
    print(f"✅ Model loaded (backbone: {backbone_type.upper()})")
    
    # Create dataloader (annotation-based)
    print(f"\n[2/3] Creating annotation-based dataloader (split={split})...")
    val_loader = create_annotation_dataloaders(
        dataset_folders=dataset_folders,
        split=split,
        window_duration_sec=window_duration,
        sample_rate=sample_rate,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        seed=seed,
    )
    print(f"✅ Dataloader created ({len(val_loader)} batches)")
    
    # Extract embeddings
    print(f"\n[3/3] Extracting embeddings...")
    X, y, groups = extract_embeddings(module, val_loader, device, verbose=True)
    print(f"✅ Embeddings extracted")
    print(f"   Shape: {X.shape}")
    print(f"   Labels: Positive={y.sum()}, Negative={(1-y).sum()}")
    print(f"   Groups: {len(set(groups))} unique files")
    
    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluating Metrics")
    print("=" * 80)
    print(f"Cross-validation: {n_splits} folds (Stratified Group)")
    print(f"PCA components: {n_components}")
    
    metrics = eval_separation_metrics(X, y, groups, n_splits, n_components, seed)
    
    print("\n📊 Results:")
    print("-" * 80)
    print(f"  Linear Probe Accuracy:  {metrics['Acc']:.4f}")
    print(f"  ROC-AUC:                {metrics['ROC']:.4f}")
    print(f"  F1-Score:               {metrics['F1']:.4f}")
    print(f"  Precision:              {metrics['Precision']:.4f}")
    print(f"  Recall:                 {metrics['Recall']:.4f}")
    print(f"  KNN Accuracy:           {metrics['KNN']:.4f}")
    print("-" * 80)
    print("\n✅ Evaluation completed!")
    
    return metrics