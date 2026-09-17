import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from training.dataset import PTBXLDataModule
from training.train_module import LitCardioSignal

H5_PATH = os.path.join("data", "processed", "ptbxl_preprocessed.h5")
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

def run_training(max_epochs=20, batch_size=64, learning_rate=1e-3, num_workers=2):
    pl.seed_everything(42, workers=True)

    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(f"Missing {H5_PATH}. Run preprocessing/build_dataset.py first.")

    dm = PTBXLDataModule(h5_path=H5_PATH, batch_size=batch_size, num_workers=num_workers)
    lit_model = LitCardioSignal(num_classes=5, lr=learning_rate)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="cardiosignal_best-{epoch:02d}-{val_macro_auroc:.4f}",
        save_top_k=1,
        monitor="val_macro_auroc",
        mode="max",
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()],
        logger=CSVLogger(LOG_DIR, name="CardioSignal_Experiment"),
        gradient_clip_val=1.0,
        deterministic=True
    )

    print("[*] Starting Multimodal Model Training...")
    trainer.fit(lit_model, datamodule=dm)

    print("\n[*] Evaluating Best Model on Test Set (Fold 10)...")
    trainer.test(lit_model, datamodule=dm, ckpt_path=checkpoint_callback.best_model_path)

if __name__ == "__main__":
    run_training()