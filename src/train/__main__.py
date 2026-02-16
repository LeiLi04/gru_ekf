"""Hydra + Lightning training entrypoint.

Usage:
    python -m src.train
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@hydra.main(version_base="1.3", config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    # Tee stdout/stderr to both console and log file in the Hydra run directory.
    log_dir = Path(cfg.hydra.run.dir) if hasattr(cfg, "hydra") else Path.cwd()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "__main__.log"
    log_file = open(log_path, "a", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        @property
        def encoding(self):
            for s in self._streams:
                enc = getattr(s, "encoding", None)
                if enc is not None:
                    return enc
            return None

        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self._streams:
                s.flush()

        def isatty(self):
            return any(getattr(s, "isatty", lambda: False)() for s in self._streams)

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    datamodule = instantiate(cfg.data)
    # Keep nested Hydra objects (e.g., optimizer/scheduler configs) as DictConfig
    # so the LightningModule can instantiate them with access to model params.
    model = instantiate(cfg.model, _recursive_=False)
    callbacks: List = [instantiate(cb) for cb in cfg.callbacks.values() if cb is not None]

    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    # Log dataset Q (Wiener/CV) once for traceability.
    try:
        dataset_path = Path(to_absolute_path(str(cfg.data.dataset_path)))
        if dataset_path.exists():
            payload = np.load(dataset_path, allow_pickle=True)
            q_base = payload.get("Q", None)
            r_base = payload.get("R", None)
            sigma_r = payload.get("sigma_r", None)
            qc_val = payload.get("qc", None)
            dt_val = payload.get("dt", None)
            if q_base is not None:
                print(f"[dataset] Q shape={np.asarray(q_base).shape}, qc={qc_val}, dt={dt_val}")
                print(f"[dataset] Q=\n{np.asarray(q_base)}")
            else:
                print("[dataset] Q not found in NPZ.")
            if r_base is not None:
                print(f"[dataset] R shape={np.asarray(r_base).shape}")
                print(f"[dataset] R=\n{np.asarray(r_base)}")
            elif sigma_r is not None:
                sigma_val = float(np.asarray(sigma_r).item())
                print(f"[dataset] sigma_r={sigma_val} (R = sigma_r^2 * I)")
            else:
                print("[dataset] R/sigma_r not found in NPZ.")
        else:
            print(f"[dataset] path not found: {dataset_path}")
    except Exception as exc:  # pragma: no cover - logging only
        print(f"[dataset] failed to read Q from NPZ: {exc}")

    ckpt_path: Optional[str] = cfg.get("ckpt_path", None)
    if ckpt_path:
        ckpt_path = str(to_absolute_path(str(ckpt_path)))
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    test_ckpt = ckpt_path if ckpt_path else None
    trainer.test(model=model, datamodule=datamodule, ckpt_path=test_ckpt)


if __name__ == "__main__":
    main()
