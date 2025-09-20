import typer

import cvae

app = typer.Typer(help="C-VAE MNIST CLI")


# ------------- Wrappers [project.scripts] -------------
def experiment_main():
    typer.run(cvae.experiment_run)


def dataset_main():
    typer.run(cvae.prepare_data)


def train_main():
    typer.run(cvae.train_model)


def plot_main():
    typer.run(cvae.plot_results)


def gen_main():
    typer.run(cvae.generate_digit)
