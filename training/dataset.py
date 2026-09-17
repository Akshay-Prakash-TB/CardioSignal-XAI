import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class PTBXLDataset(Dataset):
    def __init__(self, h5_path, fold_indices):
        self.h5_path = h5_path
        self.fold_indices = fold_indices
        self.h5_file = None
        
        with h5py.File(self.h5_path, "r") as f:
            folds = f["folds"][:]
            self.indices = np.where(np.isin(folds, self.fold_indices))[0]
            
    def _open_h5(self):
        self.h5_file = h5py.File(self.h5_path, "r")
        self.ecg_signals = self.h5_file["ecg_signals"]
        self.metadata = self.h5_file["metadata"]
        self.labels = self.h5_file["labels"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self._open_h5()
            
        real_idx = self.indices[idx]
        
        # Shape: (Channels=12, Length=1000) for 1D-CNN
        ecg = self.ecg_signals[real_idx].T.astype(np.float32)
        meta = self.metadata[real_idx].astype(np.float32)
        label = self.labels[real_idx].astype(np.float32)

        return {
            "ecg": torch.tensor(ecg, dtype=torch.float32),
            "metadata": torch.tensor(meta, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }

class PTBXLDataModule(pl.LightningDataModule):
    def __init__(self, h5_path="data/processed/ptbxl_preprocessed.h5", batch_size=64, num_workers=2):
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Folds 1-8: Train, Fold 9: Validation, Fold 10: Held-out Test
        self.train_dataset = PTBXLDataset(self.h5_path, fold_indices=list(range(1, 9)))
        self.val_dataset = PTBXLDataset(self.h5_path, fold_indices=[9])
        self.test_dataset = PTBXLDataset(self.h5_path, fold_indices=[10])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)