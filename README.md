**C-VAE MNIST**

- Generate MNIST digits with a Conditional Variational Autoencoder (CVAE). This project covers data preparation, training, sample generation, and experiment tracking with MLflow (local by default, optional Docker server).

**Overview**

- Task: learn the conditional distribution p(x|y) and generate 28x28 images conditioned on the digit y ∈ {0,…,9}.
- Main code in `src/` with a simple pipeline: data → train → generate.
- Supporting notebook in `notebooks/quick_start.ipynb`.

**Requirements**

- Python 3.10
- Poetry (dependency management)
- Git
- Docker and Docker Compose (optional, for MLflow server)
- CUDA GPU (optional; speeds up training)

**Installation**

- Clone and install dependencies:
  - `git clone <this-repo> && cd cvae-mnist`
  - `poetry env use 3.10`
  - `poetry install`
  - (optional) `poetry shell`

**Quick Start**

- Prepare MNIST (download and save tensors):
  - `make data` or `python src/dataset.py`
- Train the CVAE:
  - `python src/modeling/train.py`
- Generate an image for a digit (e.g., 6):
  - `python src/modeling/generate.py --digit 6`
- Notebook:
  - `poetry run jupyter lab` and open `notebooks/quick_start.ipynb`

**Expected Outputs**

- `data/processed/`: `.pt` files with train/test images and labels.
- `models/`: saved model weights (`model.pth`) and decoder (`decoder.pth`).
- `reports/figures/`: generated images, e.g., `cvae_digit6.png`.
- `mlruns/`: MLflow local directory (default tracking).

**Hyperparameters & Config**

- `params.yaml:1`: sets `dataset.batch_size`, `dataset.image_size`, `model.latent_dim`, `train.epochs`, etc.
- `src/config.py:1`: paths, loads `params.yaml`, and sets `mlflow.set_tracking_uri` to `mlruns/`.

**MLflow**

- Local (default): tracking in `mlruns/`.
  - Local UI: `poetry run mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000`
- Server with Docker (Postgres + MinIO):
  - File: `mlflow-server/compose.yaml:1`
  - Start services: `cd mlflow-server && docker compose up -d`
  - Access `http://localhost:5000` and point the project in `src/config.py:1`:
    - `mlflow.set_tracking_uri("http://localhost:5000")`

**Training (CLI)**

- Script: `src/modeling/train.py:1` (Typer app entry).
- Useful args (default from `params.yaml`):
  - `--latent_dim`, `--batch_size`, `--num_epochs`, `--lr`, `--momentum`, `--num_classes`.
- Example: `python src/modeling/train.py --latent_dim 100 --num_epochs 10`

**Generation (CLI)**

- Script: `src/modeling/generate.py:1`
- Example: `python src/modeling/generate.py --digit 7` → `reports/figures/cvae_digit7.png`

**Project Structure**

- `data/` data at different stages (created on demand)
- `mlflow-server/` docker-compose for MLflow + Postgres + MinIO
- `models/` training artifacts (`.pth`)
- `notebooks/` Jupyter notebooks
- `references/` reference materials (VAE, CVAE)
- `reports/` outputs and figures
- `src/` source code (config, dataset, modeling)
- `params.yaml` hyperparameters
- `Makefile` automations (`requirements`, `data`, `lint`, `format`, `test`)

**Useful Commands (Makefile)**

- `make requirements` install dependencies
- `make data` prepare MNIST dataset
- `make lint` style checks (flake8, isort, black --check)
- `make format` code formatting (isort, black)
- `make test` run tests (if present)

**License**

- MIT (see `LICENSE`)

**Credits**

- CVAE implementation in PyTorch with one-hot conditioning. Main code: `src/modeling/models/cvae.py:1`, `src/modeling/models/conditional_encoder.py:1`, `src/modeling/models/conditional_decoder.py:1`, utilities in `src/modeling/utils.py:1`.
