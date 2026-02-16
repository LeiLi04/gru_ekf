"""PyTorch Lightning DataModule for the range-only GRU-augmented EKF dataset."""
from __future__ import annotations

from pathlib import Path
from functools import partial
from typing import Optional

import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, Subset

from src.data.components import (
    MeasurementDataset,
    collate_padded_observations,
    create_splits_file_name,
    load_splits_file,
    obtain_tr_val_test_idx,
    obtain_tr_val_test_warm_idx,
    save_splits_file,
    windowed_dataset,
)


class RangeDataModule(pl.LightningDataModule):
    """LightningDataModule handling NPZ range datasets with optional windowing and warm subset."""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        window_length: int = 500,
        window_stride: Optional[int] = None,
        tr_to_test_split: float = 0.9,
        tr_to_val_split: float = 0.8333,
        warm_fraction: float = 0.1,
        warmup_split: str = "val",
        splits_name: str = "",
        split_path: str = "",
        obs_dim: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.shuffle = bool(shuffle)
        self.window_length = int(window_length)
        self.window_stride = int(window_stride) if window_stride is not None else None
        self.tr_to_test_split = float(tr_to_test_split)
        self.tr_to_val_split = float(tr_to_val_split)
        self.warm_fraction = float(warm_fraction)
        self.warmup_split = (warmup_split or "val").lower()
        self.splits_name = splits_name
        self.split_path = split_path
        self.obs_dim = int(obs_dim)
        self.seed = int(seed)

        self.dataset: Optional[MeasurementDataset] = None
        self.train_subset: Optional[Subset] = None
        self.val_subset: Optional[Subset] = None
        self.test_subset: Optional[Subset] = None
        self.warm_subset: Optional[Subset] = None
        self.train_windows = None
        self.anchors: Optional[torch.Tensor] = None
        self.dt: Optional[float] = None

    def prepare_data(self) -> None:  # noqa: D401
        """Nothing to download; ensure dataset file exists."""
        path = Path(to_absolute_path(self.dataset_path))
        if not path.exists():
            raise FileNotFoundError(f"Dataset NPZ not found at {path}")

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_path = Path(to_absolute_path(self.dataset_path))
        self.dataset = MeasurementDataset(dataset_path)
        self.anchors = self.dataset.anchors
        self.dt = self.dataset.dt

        splits_path = (
            Path(to_absolute_path(self.split_path))
            if self.split_path
            else create_splits_file_name(dataset_path, self.splits_name)
        )
        if splits_path.exists():
            splits = load_splits_file(splits_path)
            train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
            warm_idx = splits.get("warm")
        else:
            train_idx, val_idx, test_idx, warm_idx = obtain_tr_val_test_warm_idx(
                self.dataset,
                self.tr_to_test_split,
                self.tr_to_val_split,
                self.warm_fraction,
                seed=self.seed,
            )
            save_payload = {"train": train_idx, "val": val_idx, "test": test_idx, "warm": warm_idx}
            save_splits_file(splits_path, save_payload)

        self.train_subset = Subset(self.dataset, train_idx)
        self.val_subset = Subset(self.dataset, val_idx)
        self.test_subset = Subset(self.dataset, test_idx)
        self.warm_subset = Subset(self.dataset, warm_idx) if warm_idx is not None else None

        warm_split_target = self.warmup_split if self.warmup_split in {"train", "val", "test"} else "val"
        if self.warm_subset is None:
            if warm_split_target == "train":
                self.warm_subset = self.train_subset
            elif warm_split_target == "test":
                self.warm_subset = self.test_subset
            else:
                self.warm_subset = self.val_subset

        self.train_windows = windowed_dataset(
            self.train_subset, window_length=self.window_length, stride=self.window_stride
        )

    def _make_loader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_padded_observations, obs_dim=self.obs_dim),
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_windows is None:
            raise RuntimeError("DataModule.setup must be called before requesting dataloaders.")
        return self._make_loader(self.train_windows, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        if self.val_subset is None:
            raise RuntimeError("DataModule.setup must be called before requesting dataloaders.")
        return self._make_loader(self.val_subset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_subset is None:
            raise RuntimeError("DataModule.setup must be called before requesting dataloaders.")
        return self._make_loader(self.test_subset, shuffle=False)

    def warm_dataloader(self) -> Optional[DataLoader]:
        if self.warm_subset is None:
            return None
        return self._make_loader(self.warm_subset, shuffle=False)


__all__ = ["RangeDataModule"]
