import os
import sys
from pathlib import Path

# Configure utf-8 stdout/stderr for Windows compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from training.dataset import PTBXLDataModule
from training.train_module import LitCardioSignal
from models.architectures import MODEL_REGISTRY, list_available_models

H5_PATH = os.path.join("data", "processed", "ptbxl_preprocessed.h5")
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

def get_checkpoint_filename(model_key: str):
    slug = model_key.lower().replace("-", "_").replace(" ", "_")
    return f"{slug}_best"

def train_single_model(model_key: str, max_epochs: int = 5, batch_size: int = 64, 
                       learning_rate: float = 1e-3, num_workers: int = 0):
    """Trains a single model architecture on PTB-XL folds 1-8, validating on fold 9 and testing on fold 10."""
    pl.seed_everything(42, workers=True)

    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(f"Missing {H5_PATH}. Run preprocessing/build_dataset.py first.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[*] Training Model: {model_key}")
    print(f"   Architecture: {MODEL_REGISTRY[model_key]['description']}")
    print(f"   Max Epochs: {max_epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f"{'='*60}")

    dm = PTBXLDataModule(h5_path=H5_PATH, batch_size=batch_size, num_workers=num_workers)
    lit_model = LitCardioSignal(model_key=model_key, num_classes=5, lr=learning_rate)

    safe_name = get_checkpoint_filename(model_key)
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=f"{safe_name}",
        save_top_k=1,
        monitor="val_macro_auroc",
        mode="max",
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=4,
        mode="min",
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()],
        logger=CSVLogger(LOG_DIR, name=f"CardioSignal_{safe_name}"),
        gradient_clip_val=1.0,
        deterministic=True
    )

    trainer.fit(lit_model, datamodule=dm)

    best_ckpt = checkpoint_callback.best_model_path
    if not best_ckpt or not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(CHECKPOINT_DIR, f"{safe_name}.ckpt")
        trainer.save_checkpoint(best_ckpt)

    print(f"\n[*] Evaluating Best Checkpoint on Test Set (Fold 10): {best_ckpt}")
    test_results = trainer.test(lit_model, datamodule=dm, ckpt_path=best_ckpt)
    print(f"[✓] Completed training & testing for {model_key}")
    return test_results


def run_training():
    parser = argparse.ArgumentParser(description="Train CardioSignal-XAI Models on PTB-XL")
    parser.add_argument("--model", type=str, default="all", 
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        help="Specific model to train, or 'all' to train all registered models")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs per model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--force", action="store_true", help="Force retrain even if checkpoint exists")
    args = parser.parse_args()

    models_to_train = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    
    print(f"[*] CardioSignal-XAI Multi-Model Training Pipeline")
    print(f"[*] Models queued ({len(models_to_train)}): {models_to_train}")

    for idx, m_key in enumerate(models_to_train, 1):
        safe_name = get_checkpoint_filename(m_key)
        ckpt_candidate = os.path.join(CHECKPOINT_DIR, f"{safe_name}.ckpt")
        if os.path.exists(ckpt_candidate) and not args.force:
            print(f"\n[i] [{idx}/{len(models_to_train)}] Checkpoint already exists for {m_key}: {ckpt_candidate}. Skipping (use --force to overwrite).")
            continue

        print(f"\n>>> [{idx}/{len(models_to_train)}] Starting {m_key}...")
        train_single_model(
            model_key=m_key,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_workers=args.num_workers
        )

    print("\n" + "="*60)
    print("[✓] All queued models trained and saved to checkpoints/!")
    print("="*60)


if __name__ == "__main__":
    run_training()