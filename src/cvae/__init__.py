"""API pública do pacote CVAE."""

from cvae.service.dataset import main as prepare_data
from cvae.service.experiment import main as experiment_run
from cvae.service.gen import main as generate_digit
from cvae.service.plots import main as plot_results
from cvae.service.train import main as train_model

__all__ = ["prepare_data", "train_model", "plot_results", "generate_digit", "experiment_run"]
