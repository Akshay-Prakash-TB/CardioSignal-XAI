import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
from models.architectures import build_model, MODEL_REGISTRY

class LitCardioSignal(pl.LightningModule):
    """
    PyTorch Lightning module supporting all registered multimodal ECG architectures.
    Provides standard multi-label training, validation, and testing metrics.
    """
    def __init__(self, model_key="CardioSignal-CrossAttn", num_classes=5, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model_key = model_key
        self.model = build_model(model_key, num_classes=num_classes)
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.val_f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.test_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.test_f1 = MultilabelF1Score(num_labels=num_classes, average="macro")

    def forward(self, ecg, metadata):
        return self.model(ecg, metadata)

    def training_step(self, batch, batch_idx):
        logits = self(batch["ecg"], batch["metadata"])
        loss = self.criterion(logits, batch["label"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["ecg"], batch["metadata"])
        loss = self.criterion(logits, batch["label"])
        preds = torch.sigmoid(logits)
        
        self.val_auroc.update(preds, batch["label"].long())
        self.val_f1.update(preds, batch["label"].long())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_auroc_val = self.val_auroc.compute()
        val_f1_val = self.val_f1.compute()
        self.log("val_macro_auroc", val_auroc_val, prog_bar=True)
        self.log("val_macro_f1", val_f1_val, prog_bar=True)
        self.val_auroc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        logits = self(batch["ecg"], batch["metadata"])
        preds = torch.sigmoid(logits)
        self.test_auroc.update(preds, batch["label"].long())
        self.test_f1.update(preds, batch["label"].long())

    def on_test_epoch_end(self):
        self.log("test_macro_auroc", self.test_auroc.compute())
        self.log("test_macro_f1", self.test_f1.compute())
        self.test_auroc.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}