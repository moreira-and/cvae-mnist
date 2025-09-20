# -----------------------------------------
from pathlib import Path
from types import SimpleNamespace

# -----------------------------------------
from dotenv import load_dotenv

# -----------------------------------------
from loguru import logger

# -----------------------------------------
import mlflow

# -----------------------------------------
import torch
import yaml

# -----------------------------------------
load_dotenv()  # Load environment variables from .env file if it exists
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -----------------------------------------

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CONFIG_FILE = PROJ_ROOT / "params.yaml"

MLFLOW_TRACKING_URI = PROJ_ROOT / "mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# -----------------------------------------
# Load configuration from params.yaml
def dict_to_namespace(d: dict) -> SimpleNamespace:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


with open(CONFIG_FILE, "r") as f:
    params_dict = yaml.safe_load(f)

params = dict_to_namespace(params_dict)
# -----------------------------------------

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
