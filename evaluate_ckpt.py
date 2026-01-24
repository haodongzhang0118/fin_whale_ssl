#!/usr/bin/env python3
"""
Simple script to evaluate a checkpoint on validation/test set
Uses OfflineProb.py method (PyTorch-based linear probe)

Usage:
    python evaluate_ckpt.py --ckpt path/to/checkpoint.ckpt --data /path/to/data
"""
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from eval import load_model_from_checkpoint
from dataloader import create_supervised_dataloaders


def extract_embeddings(module, loader, device):
    """Extract embeddings from CPC model (same as OfflineProb collection)"""
    module.eval()
    backbone = module.backbone
    
    X_list, y_list = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            # Get audio and move to device
            wav = batch["raw_audio"].to(device)
            if wav.ndim == 2:
                wav = wav.unsqueeze(1)  # [B, 1, T]
            
            # Extract embeddings
            emb = backbone.extract_embedding(wav)  # [B, D]
            
            # Collect (keep as tensors like OfflineProb)
            X_list.append(emb.cpu())
            y_list.append(batch["label"].cpu())
    
    # Concatenate
    X = torch.cat(X_list, dim=0)  # [N, D]
    y = torch.cat(y_list, dim=0)  # [N] or [N, 1]
    
    if y.dim() > 1:
        y = y.squeeze(-1)
    
    return X, y


def train_linear_probe(X_train, y_train, X_val, y_val, input_dim, num_classes, 
                       train_epochs, lr, optimizer_type, weight_decay, device):
    """Train linear probe (same as OfflineProb._train_probe)"""
    # Create probe
    probe = nn.Linear(input_dim, num_classes, bias=True).to(device)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)
    
    # Create optimizer
    if optimizer_type == 'lars':
        from stable_pretraining.optim import LARS
        optimizer = LARS(
            probe.parameters(),
            lr=lr,
            clip_lr=True,
            eta=0.02,
            exclude_bias_n_norm=True,
            weight_decay=weight_decay,
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device).long()
    X_val = X_val.to(device)
    y_val = y_val.to(device).long()
    
    # Training loop
    probe.train()
    best_val_acc = 0.0
    best_state = None
    
    print(f"\nTraining linear probe ({train_epochs} epochs)...")
    with torch.enable_grad():
        for epoch in range(train_epochs):
            # Forward pass
            logits = probe(X_train)
            loss = criterion(logits, y_train)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validate every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == train_epochs - 1:
                probe.eval()
                with torch.no_grad():
                    val_logits = probe(X_val)
                    val_acc = (val_logits.argmax(dim=1) == y_val).float().mean()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                    
                    if (epoch + 1) % 20 == 0 or epoch == train_epochs - 1:
                        print(f"  Epoch {epoch+1}/{train_epochs}: val_acc={val_acc:.4f}")
                
                probe.train()
    
    # Load best model
    if best_state is not None:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    return probe


def compute_metrics(probe, X_val, y_val, device):
    """Compute metrics (same as OfflineProb._compute_metrics)"""
    probe.eval()
    X_val = X_val.to(device)
    y_val = y_val.to(device).long()
    
    with torch.no_grad():
        logits = probe(X_val)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        # Basic metrics
        acc = (preds == y_val).float().mean()
        
        # Confusion matrix
        tp = ((preds == 1) & (y_val == 1)).sum().float()
        fp = ((preds == 1) & (y_val == 0)).sum().float()
        fn = ((preds == 0) & (y_val == 1)).sum().float()
        tn = ((preds == 0) & (y_val == 0)).sum().float()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # AUROC
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(y_val.cpu().numpy(), probs[:, 1].cpu().numpy())
        except Exception:
            auroc = 0.5
    
    return {
        'accuracy': acc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'auroc': auroc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CPC checkpoint using OfflineProb method")
    
    # Required arguments
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file (.ckpt)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data folder")
    parser.add_argument("--config", type=str, default="configs/cpc_config.yaml",
                        help="Path to config file (default: configs/cpc_config.yaml)")
    
    # Optional arguments
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Data split to evaluate (default: val)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers (default: 4)")
    parser.add_argument("--window", type=float, default=8.0,
                        help="Window duration in seconds (default: 8.0)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    parser.add_argument("--train_epochs", type=int, default=100,
                        help="Probe training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Probe learning rate (default: 0.1)")
    parser.add_argument("--optimizer", type=str, default="lars",
                        choices=["lars", "adam", "adamw", "sgd"],
                        help="Probe optimizer (default: lars)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay (default: 0.0)")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Train/val split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Set seed
    torch.manual_seed(args.seed)
    
    print("=" * 80)
    print("CPC Checkpoint Evaluation (OfflineProb Method)")
    print("=" * 80)
    print(f"Checkpoint:     {args.ckpt}")
    print(f"Config:         {args.config}")
    print(f"Data folder:    {args.data}")
    print(f"Split:          {args.split}")
    print(f"Window:         {args.window}s")
    print(f"Device:         {device}")
    print(f"Train epochs:   {args.train_epochs}")
    print(f"Learning rate:  {args.lr}")
    print(f"Optimizer:      {args.optimizer}")
    print(f"Train/val:      {args.train_split:.0%}/{1-args.train_split:.0%}")
    print("=" * 80)
    
    # Step 1: Load model
    print("\n[1/4] Loading model from checkpoint...")
    module, cfg = load_model_from_checkpoint(args.ckpt, args.config, device)
    backbone_type = cfg.get('backbone_type', 'gru')
    
    # Get embedding dimension
    if backbone_type == 'transformer':
        input_dim = cfg.model.transformer_hidden
    else:
        input_dim = cfg.model.gru_hidden
    
    print(f"✅ Model loaded (backbone: {backbone_type.upper()}, dim: {input_dim})")
    
    # Step 2: Create dataloader
    print(f"\n[2/4] Creating {args.split} dataloader...")
    loader = create_supervised_dataloaders(
        data_folder=args.data,
        split=args.split,
        window_duration_sec=args.window,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        seed=args.seed,
    )
    print(f"✅ Dataloader created ({len(loader)} batches)")
    
    # Step 3: Extract embeddings
    print(f"\n[3/4] Extracting embeddings...")
    X, y = extract_embeddings(module, loader, device)
    n_samples = X.shape[0]
    print(f"✅ Embeddings extracted")
    print(f"   Shape:      {X.shape}")
    print(f"   Positive:   {y.sum()} ({y.sum()/n_samples*100:.1f}%)")
    print(f"   Negative:   {(y==0).sum()} ({(y==0).sum()/n_samples*100:.1f}%)")
    
    # Step 4: Train linear probe
    print(f"\n[4/4] Training linear probe...")
    
    # Split train/val (same as OfflineProb)
    n_train = int(args.train_split * n_samples)
    perm = torch.randperm(n_samples)
    
    X_train = X[perm[:n_train]]
    y_train = y[perm[:n_train]]
    X_val = X[perm[n_train:]]
    y_val = y[perm[n_train:]]
    
    print(f"   Train:      {n_train} samples")
    print(f"   Val:        {n_samples - n_train} samples")
    
    # Train probe
    probe = train_linear_probe(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        num_classes=2,
        train_epochs=args.train_epochs,
        lr=args.lr,
        optimizer_type=args.optimizer,
        weight_decay=args.weight_decay,
        device=device
    )
    
    # Compute final metrics
    metrics = compute_metrics(probe, X_val, y_val, device)
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 Final Results (on validation split)")
    print("=" * 80)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1']:.4f}")
    print(f"  ROC-AUC:     {metrics['auroc']:.4f}")
    print("=" * 80)
    print("\n✅ Evaluation completed!\n")
    
    return metrics


if __name__ == "__main__":
    main()
