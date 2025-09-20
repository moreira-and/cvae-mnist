import time

import matplotlib.pyplot as plt
import typer

from cvae.config import FIGURES_DIR, logger, params
from cvae.service.gen import generate_samples
from cvae.service.utils import get_data_loader

app = typer.Typer()


@app.command()
def main():
    # -----------------------------------------
    start_time = time.time()
    logger.info("Generating plot from data...")
    # -----------------------------------------

    train_loader = get_data_loader(params.dataset.batch_size, train=True)

    data, targets = train_loader.dataset.tensors
    real_samples = []
    for digit in range(10):
        idx = (targets == digit).nonzero(as_tuple=True)[0][0]
        real_samples.append(data[idx])

    fig, axs = plt.subplots(4, 10, figsize=(18, 4))

    # Linha 1 - reais
    for i, img in enumerate(real_samples):
        axs[0, i].imshow(img.squeeze(), cmap="gray")
        axs[0, i].set_title(f"Real: {i}")
        axs[0, i].axis("off")

    # Linha 2 - gerados
    generated_samples = generate_samples()
    for i, img in enumerate(generated_samples):
        axs[1, i].imshow(img.squeeze(), cmap="gray")
        axs[1, i].set_title(f"Generated: {i}")
        axs[1, i].axis("off")

    # Linha 3 - gerados
    generated_samples = generate_samples()
    for i, img in enumerate(generated_samples):
        axs[2, i].imshow(img.squeeze(), cmap="gray")
        axs[2, i].set_title(f"Generated: {i}")
        axs[2, i].axis("off")

    # Linha 4 - gerados
    generated_samples = generate_samples()
    for i, img in enumerate(generated_samples):
        axs[3, i].imshow(img.squeeze(), cmap="gray")
        axs[3, i].set_title(f"Generated: {i}")
        axs[3, i].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cvae_comparison.png")
    plt.show()
    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Plot generation complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()
