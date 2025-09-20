import time

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import typer

from cvae.config import DATA_DIR, PROCESSED_DATA_DIR, logger

app = typer.Typer()


def get_mnist_data(train: bool) -> None:
    dataset = MNIST(root=DATA_DIR, train=train, download=True, transform=transforms.ToTensor())

    images = torch.stack([img for img, _ in dataset])  # [N, 1, 28, 28]
    labels = torch.tensor([label for _, label in dataset], dtype=torch.long)  # [N]

    file_name = "train" if train else "test"
    path = PROCESSED_DATA_DIR / f"mnist_{file_name}.pt"

    torch.save({"images": images, "labels": labels}, path)
    print(f"Saved {path} with {images.shape[0]} samples.")


@app.command()
def main():
    # -----------------------------------------
    start_time = time.time()
    logger.info("Starting dataset processing...")
    # -----------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    get_mnist_data(train=True)
    get_mnist_data(train=False)

    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Processing dataset complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()
