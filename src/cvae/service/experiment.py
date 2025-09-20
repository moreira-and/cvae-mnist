import time

import typer

import cvae
from cvae.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, logger, mlflow, params

app = typer.Typer()


def save_experiment(elapsed_time: float):
    mlflow.set_experiment("CVAE_MNST")

    with mlflow.start_run():
        # arquivos individuais -> log_artifact
        mlflow.log_artifact(str(MODELS_DIR / "model.pth"), artifact_path="models")
        mlflow.log_artifact(str(FIGURES_DIR / "cvae_comparison.png"), artifact_path="figures")

        # parâmetros -> log_param
        mlflow.log_param("elapsed_time", elapsed_time)
        mlflow.log_param("batch_size", params.dataset.batch_size)
        mlflow.log_param("test_batch_size", params.dataset.test_batch_size)
        mlflow.log_param("image_size", params.dataset.image_size)
        mlflow.log_param("num_classes", params.dataset.num_classes)
        mlflow.log_param("latent_dim", params.model.latent_dim)
        mlflow.log_param("seed", params.model.seed)
        mlflow.log_param("lr", params.train.lr)
        mlflow.log_param("momentum", params.train.momentum)
        mlflow.log_param("epochs", params.train.epochs)

        mlflow.log_artifact(str(REPORTS_DIR / "eval_metrics.csv"), artifact_path="models")


@app.command(name="experiment")
def main(
    do_data: bool = typer.Option(True, help="Executa etapa de dados"),
    do_train: bool = typer.Option(True, help="Executa treino"),
    do_plots: bool = typer.Option(True, help="Gera plots"),
    do_eval: bool = typer.Option(False, help="Avalia modelo"),
):

    t0 = time.time()
    logger.info("Starting experiment...")
    if do_data:
        cvae.prepare_data()
    if do_train:
        cvae.train_model()
    if do_plots:
        cvae.plot_results()
    if do_eval:
        cvae.evaluate_model()
    elapsed = time.time() - t0
    save_experiment(elapsed_time=elapsed)
    logger.success(f"Experiment complete in {elapsed:.2f}s")


if __name__ == "__main__":
    app()
