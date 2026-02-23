import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pytorch_lightning as L


class ArrayDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class WBosonDataModule(L.LightningDataModule):
    """
    Supports BOTH:
        1) Random split via val_frac / test_frac
        2) Explicit KFold splits via train_idx / val_idx
    """

    def __init__(
        self,
        X,
        Y,
        seed=114,
        batch_size=512,
        val_frac=0.1,
        test_frac=0.1,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    ):
        super().__init__()

        self.X = X
        self.Y = Y
        self.batch_size = batch_size

        self.val_frac = val_frac
        self.test_frac = test_frac

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.seed = seed

        if num_workers is None:
            self.num_workers = max(1, int(os.cpu_count() * 0.8))
            print(f"Automatically setting num_workers to {self.num_workers}")
        else:
            self.num_workers = num_workers
            print(f"Using {self.num_workers} num of workers in data loading.")

    def setup(self, stage=None):
        ds = ArrayDataset(self.X, self.Y)
        self.std_ds = ds  # keep for (inverse) transform.

        # -------- KFold / explicit split --------
        if self.train_idx is not None and self.val_idx is not None:
            print("Using explicit (KFold) dataset split")

            self.train_ds = Subset(ds, self.train_idx)
            self.val_ds = Subset(ds, self.val_idx)

            if self.test_idx is not None:
                self.test_ds = Subset(ds, self.test_idx)
            else:
                self.test_ds = None

        # -------- Random split (original behavior) --------
        else:
            print("Using random split (val_frac / test_frac)")

            n = len(ds)
            n_val = int(self.val_frac * n)
            n_test = int(self.test_frac * n)
            n_train = n - n_val - n_test

            self.train_ds, self.val_ds, self.test_ds = random_split(
                ds,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(self.seed),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
