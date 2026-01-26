"""
Fine-tuning script for CPC model with classification head
Evaluate downstream task performance with different strategies:
1. Frozen backbone + train head only
2. Full fine-tuning (backbone + head)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from omegaconf import OmegaConf
import argparse
from pathlib import Path

from eval import load_model_from_checkpoint
from dataloader import create_annotation_dataloaders


class ClassificationHead(nn.Module):
    """
    Classification head for fine-tuning
    Supports both linear and MLP architectures
    """
    def __init__(self, input_dim, num_classes=2, hidden_dim=None, dropout=0.3):
        super().__init__()
        
        if hidden_dim is None:
            # Simple linear head
            self.head = nn.Linear(input_dim, num_classes)
        else:
            # MLP head with dropout
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
    
    def forward(self, x):
        return self.head(x)


class FineTuneModel(nn.Module):
    """
    CPC backbone + Classification head
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        # Extract embedding from backbone
        with torch.set_grad_enabled(self.backbone.training):
            embedding = self.backbone.extract_embedding(x)
        
        # Classification
        logits = self.head(embedding)
        return logits


def train_epoch(model, dataloader, criterion, optimizer, device, freeze_backbone=True):
    """Train for one epoch"""
    model.train()
    if freeze_backbone:
        model.backbone.eval()  # Keep backbone in eval mode if frozen
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        audio = batch["raw_audio"].to(device)
        labels = batch["label"].to(device)
        
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(audio)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Collect predictions
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        audio = batch["raw_audio"].to(device)
        labels = batch["label"].to(device)
        
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]
        
        # Forward pass
        logits = model(audio)
        loss = criterion(logits, labels)
        
        # Collect predictions
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc
    }


def fine_tune(
    checkpoint_path,
    config_path,
    dataset_folders,
    output_dir="results/finetune",
    freeze_backbone=True,
    head_type="linear",  # 'linear' or 'mlp'
    hidden_dim=256,
    dropout=0.3,
    batch_size=64,
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    seed=42,
    device=None,
):
    """
    Fine-tune CPC model with classification head
    
    Uses dataset's original train/val/test splits:
    - Train on trainset
    - Validate on validset (for early stopping)
    - Final evaluation on testset
    
    Args:
        checkpoint_path: Path to pretrained CPC checkpoint
        config_path: Path to config file
        dataset_folders: Dataset folder(s) for fine-tuning
        output_dir: Directory to save results
        freeze_backbone: If True, only train head; if False, train everything
        head_type: 'linear' or 'mlp'
        hidden_dim: Hidden dim for MLP head (ignored if head_type='linear')
        dropout: Dropout rate for MLP head
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        seed: Random seed
        device: torch device
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FINE-TUNING CPC MODEL FOR DOWNSTREAM CLASSIFICATION")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_folders}")
    print(f"Strategy: {'Frozen backbone' if freeze_backbone else 'Full fine-tuning'}")
    print(f"Head type: {head_type.upper()}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load pretrained model
    print("\n[1/5] Loading pretrained CPC model...")
    module, cfg = load_model_from_checkpoint(checkpoint_path, config_path, device)
    backbone = module.backbone
    
    # Get embedding dimension
    if cfg.backbone_type == "transformer":
        embedding_dim = cfg.model.transformer_hidden
    else:
        embedding_dim = cfg.model.gru_hidden
    
    print(f"✓ Loaded backbone (embedding_dim: {embedding_dim})")
    
    # Create classification head
    print(f"\n[2/5] Creating {head_type} classification head...")
    if head_type == "linear":
        head = ClassificationHead(embedding_dim, num_classes=2, hidden_dim=None)
        print(f"✓ Linear head: {embedding_dim} → 2")
    else:  # mlp
        head = ClassificationHead(embedding_dim, num_classes=2, hidden_dim=hidden_dim, dropout=dropout)
        print(f"✓ MLP head: {embedding_dim} → {hidden_dim} → 2 (dropout={dropout})")
    
    # Create full model
    model = FineTuneModel(backbone, head).to(device)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in head.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Backbone frozen (trainable: {trainable_params:,} / {total_params:,})")
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Full fine-tuning (trainable: {trainable_params:,})")
    
    # Load data - use original train/val/test splits
    print(f"\n[3/5] Loading dataset...")
    
    # Training set
    train_loader = create_annotation_dataloaders(
        dataset_folders=dataset_folders,
        split="train",
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        seed=seed,
    )
    
    # Validation set
    val_loader = create_annotation_dataloaders(
        dataset_folders=dataset_folders,
        split="val",
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        seed=seed,
    )
    
    # Test set (for final evaluation)
    test_loader = create_annotation_dataloaders(
        dataset_folders=dataset_folders,
        split="test",
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        seed=seed,
    )
    
    print(f"✓ Train: {len(train_loader.dataset)} samples")
    print(f"✓ Val:   {len(val_loader.dataset)} samples")
    print(f"✓ Test:  {len(test_loader.dataset)} samples")
    
    # Setup training
    print(f"\n[4/5] Setting up training...")
    criterion = nn.CrossEntropyLoss()
    
    if freeze_backbone:
        # Only optimize head parameters
        optimizer = optim.Adam(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        # Optimize all parameters (differential learning rate)
        optimizer = optim.Adam([
            {'params': backbone.parameters(), 'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': head.parameters(), 'lr': learning_rate}
        ], weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"✓ Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"✓ Scheduler: CosineAnnealingLR")
    print(f"✓ Epochs: {num_epochs}")
    
    # Training loop
    print(f"\n[5/5] Training...")
    best_val_f1 = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_f1': [], 'val_f1': [], 'val_auroc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, freeze_backbone
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, "
              f"AUROC: {val_metrics['auroc']:.4f}, Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_metrics['f1'])
        history['val_auroc'].append(val_metrics['auroc'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'backbone_state_dict': backbone.state_dict(),
                'head_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, output_dir / "best_finetuned.pth")
            print(f"✓ Saved best model (F1: {best_val_f1:.4f})")
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    
    # Load best model
    best_checkpoint = torch.load(output_dir / "best_finetuned.pth", map_location=device)
    backbone.load_state_dict(best_checkpoint['backbone_state_dict'])
    head.load_state_dict(best_checkpoint['head_state_dict'])
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Results (Best model from epoch {best_epoch}):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    
    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETED")
    print("=" * 80)
    print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Model saved to: {output_dir / 'best_finetuned.pth'}")
    print("=" * 80)
    
    return history, best_val_f1, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CPC model with classification head",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

# Frozen backbone (fast, recommended for first try)
python finetune.py \\
    --checkpoint /root/results/ckpts_small/best.ckpt \\
    --dataset /root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/MEDITERRANEAN_FIN_WHALE \\
    --freeze_backbone \\
    --head_type linear

# Full fine-tuning with MLP head
python finetune.py \\
    --checkpoint /root/results/ckpts_small/best.ckpt \\
    --dataset /root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/MEDITERRANEAN_FIN_WHALE \\
    --head_type mlp \\
    --hidden_dim 256 \\
    --epochs 100

# Fine-tune on Caribbean dataset
python finetune.py \\
    --checkpoint /root/results/ckpts_small/best.ckpt \\
    --dataset /root/ICML_2026_FIN_HUMPBACK_WHALE/RESOURCES/CARABBEAN_HUMPBACK_WHALE \\
    --freeze_backbone \\
    --epochs 50
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pretrained CPC checkpoint')
    parser.add_argument('--config', type=str, default='configs/cpc_config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset folder')
    parser.add_argument('--output_dir', type=str, default='results/finetune',
                       help='Output directory for results')
    
    # Model architecture
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone and only train head')
    parser.add_argument('--head_type', type=str, default='linear', choices=['linear', 'mlp'],
                       help='Type of classification head')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for MLP head')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for MLP head')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Run fine-tuning
    history, best_val_f1, test_metrics = fine_tune(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        dataset_folders=args.dataset,
        output_dir=args.output_dir,
        freeze_backbone=args.freeze_backbone,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
