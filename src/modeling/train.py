import argparse
import time

from loguru import logger
import torch
import torch.optim as optim
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, device, params
from src.modeling.models.cvae import CVAE
from src.modeling.utils import loss_function, one_hot

app = typer.Typer()


def _get_train_data_loader(batch_size: int, train: bool, **kwargs):
    file_name = "train" if train else "test"
    path = PROCESSED_DATA_DIR / f"mnist_{file_name}.pt"

    payload = torch.load(path, map_location="cpu")
    images, labels = payload["images"], payload["labels"]

    dataset = torch.utils.data.TensorDataset(images, labels)

    # Shuffle at each epoch
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)


def _save_model(model):
    logger.info("Saving the model.")
    # Recommended approach: http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), MODELS_DIR / "model.pth")
    torch.save(model.decoder.state_dict(), MODELS_DIR / "decoder.pth")


def get_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters and settings (defaults loaded from params.yaml)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=params.model.latent_dim,
        metavar="N",
        help="Latent dimension for CVAE (default from params.yaml)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=params.dataset.batch_size,
        metavar="N",
        help="Training batch size (default from params.yaml)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=params.dataset.test_batch_size,
        metavar="N",
        help="Test batch size (default from params.yaml)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=params.train.epochs,
        metavar="N",
        help="Number of training epochs (default from params.yaml)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=params.dataset.num_classes,
        metavar="N",
        help="Number of classes (default from params.yaml)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=params.model.lr,
        metavar="LR",
        help="Learning rate (default from params.yaml)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=params.model.momentum,
        metavar="M",
        help="SGD momentum (not used by Adam; kept for compatibility)",
    )
    parser.add_argument(
        "--seed", type=int, default=params.model.seed, metavar="S", help="Random seed"
    )

    return parser.parse_known_args()


@app.command()
def main():
    # -----------------------------------------
    start_time = time.time()
    logger.info("Training CVAE model...")
    # -----------------------------------------

    args, _ = get_args()
    kwargs = {}
    train_loader = _get_train_data_loader(args.batch_size, train=True, **kwargs)

    logger.info("Dataset loaded.")

    # Initialize model and optimizer
    model = CVAE(input_channels=1, latent_dim=args.latent_dim, num_classes=args.num_classes).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info("Model initialized.")
    model.train()

    for epoch in tqdm(range(args.num_epochs)):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = one_hot(labels, args.num_classes, device)

            # Zero gradients
            optimizer.zero_grad()
            x_reconstructed, mu, logvar = model(data, labels)

            # Sanity check: shapes must match
            assert (
                x_reconstructed.shape == data.shape
            ), f"Shape mismatch: {x_reconstructed.shape} vs {data.shape}"

            # Compute loss and update
            loss = loss_function(x_reconstructed, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        logger.info(
            f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {train_loss/len(train_loader.dataset):.4f}"
        )

    _save_model(model)

    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Training complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()
