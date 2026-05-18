# GRU-augmented EKF (Range-only Tracking)

This project implements a `GRU + EKF` framework for 2D multi-anchor range-only tracking. The main goals are:

- Improve tracking accuracy under motion-model mismatch (turn maneuvers).
- Use innovation statistics (NIS / NLL) for unsupervised training and consistency evaluation.
- Support online `beta` noise-scale adaptation in `varyQ` scenarios.

## 1. What Has Been Implemented

- The training pipeline is unified under `Hydra + PyTorch Lightning`, with entry point:
  - `python -m src.train`
- Two dataset generators are provided:
  - `src/data/components/data_generator.py`: fixed-`Q` datasets (matched / mismatch).
  - `src/data/components/data_generator_varyQ.py`: piecewise time-varying `Q` datasets (three `qc` segments).
- Notebook roles:
  - `E1_Annotation.ipynb`: main generalization experiment under mismatch.
  - `E2_Annotation.ipynb`: online beta-adaptation analysis.

## 2. Dependencies and Environment

- Python: recommended `3.10+`
- Install dependencies (we use [`uv`](https://docs.astral.sh/uv/) for fast resolution; plain `pip` also works):

```bash
# With uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

- Main dependencies in `requirements.txt`:
  - `torch==2.2.2+cu121`, `torchvision==0.17.2+cu121`, `torchaudio==2.2.2+cu121`
  - `pytorch-lightning==2.4.0`
  - `hydra-core==1.3.2`, `omegaconf==2.3.0`
  - `numpy`, `scipy`, `matplotlib`, `pyyaml`, `statsmodels`, `tensorboard`

## 3. Training and Running

- Default training command:

```bash
python -m src.train
```

## 4. Configuration

- Main config entry:
  - `configs/config.yaml`
- Common config files:
  - `configs/model/gru_augmented_ekf.yaml`
  - `configs/data/range_npz.yaml`
  - `configs/trainer/default.yaml`
  - `configs/callbacks/default.yaml`
- Key parameters:
  - `model.q_init`: nominal process-noise scalar (default `0.05`)
  - `model.sigma0_scale`: initial covariance scale (default `1.0`)
  - `data.dataset_path`: dataset path for training/testing

## 5. Data Generation

### 5.1 Fixed-Q data (commonly used in E1)

Script: `src/data/components/data_generator.py`

### 5.2 Time-varying-Q data (commonly used in E2)

Script: `src/data/components/data_generator_varyQ.py`
