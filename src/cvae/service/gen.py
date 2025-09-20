import io
import json
import time

from matplotlib import pyplot as plt
import numpy as np
import torch
import typer

from cvae.config import FIGURES_DIR, MODELS_DIR, device, logger, params
from cvae.service.models.nn_cvae import nn_CVAE
from cvae.service.utils import one_hot

app = typer.Typer()


def model_fn():
    """Load the trained CVAE model from `MODELS_DIR`."""
    logger.info("Starting model_fn to load the model.")

    # Load configuration parameters
    latent_dim = params.model.latent_dim
    num_classes = params.dataset.num_classes
    input_channels = 1  # MNIST has 1 channel (grayscale)

    logger.info(
        f"Model parameters: latent_dim={latent_dim}, num_classes={num_classes}, input_channels={input_channels}"
    )

    # Initialize model
    model = nn_CVAE(input_channels=input_channels, latent_dim=latent_dim, num_classes=num_classes)
    logger.info("CVAE model initialized.")

    # Load model state
    model_path = MODELS_DIR / "model.pth"
    logger.info(f"Loading model state from: {model_path}")

    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        logger.info("Model state loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model state: {e}")
        raise

    model.eval()
    logger.info("Model set to evaluation mode.")

    return model


def input_fn(request_body, request_content_type):
    """Parse input payload into a label tensor (0–10)."""
    logger.info(f"Received payload with content type: {request_content_type}")

    if request_content_type == "application/json":
        # Load JSON body
        if isinstance(request_body, str):
            data = json.loads(request_body)
        else:
            data = json.loads(request_body.decode("utf-8"))
        logger.info(f"Received JSON: {data}")

        # Validate presence and range of 'label'
        if "label" not in data:
            logger.error("Missing 'label' key in JSON payload.")
            raise ValueError(
                "JSON must contain a 'label' key with an integer value between 0 and 10."
            )

        if not isinstance(data["label"], int) or not (0 <= data["label"] <= 10):
            logger.error(f"Invalid 'label' value (expected int in [0,10]): {data['label']}")
            raise ValueError(
                "JSON must contain a 'label' key with an integer value between 0 and 10."
            )

        # Convert to tensor
        label = torch.tensor([data["label"]], dtype=torch.int)
        logger.info(f"Converted label to tensor: {label}")
        return label
    elif request_content_type == "application/x-npy":
        logger.info("Processing .npy input payload.")
        with io.BytesIO(request_body) as f:
            np_data = np.load(f)
        label = torch.tensor(np_data, dtype=torch.int)
        logger.info(f"Converted .npy payload to tensor: {label}")
        return label
    else:
        logger.error(f"Unsupported content type: {request_content_type}")
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    logger.info(f"Starting prediction with input: {input_data}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = params.model.latent_dim
    num_classes = params.dataset.num_classes

    logger.info(f"latent_dim={latent_dim}, num_classes={num_classes}")

    with torch.no_grad():
        # Sample random latent vector
        z = torch.randn(1, latent_dim, device=device)  # [1, Z]
        logger.info(f"Sampled latent vector with shape: {z.shape}")

        y = torch.tensor([input_data], dtype=torch.long, device=device)
        y = one_hot(y, num_classes, device).float()  # [1, C]
        logger.info(f"One-hot label shape: {y.shape}")

        img = model.decoder(z, y)
        logger.info("Generated decoder output.")

    return img.squeeze().cpu().numpy()


def generate_samples() -> np.ndarray:

    model = model_fn()

    latent_dim = params.model.latent_dim
    num_classes = params.dataset.num_classes

    samples = []
    with torch.no_grad():
        for digit in range(num_classes):
            z = torch.randn(1, latent_dim, device=device)  # [1, Z]
            y = torch.tensor([digit], dtype=torch.long, device=device)
            y = one_hot(y, num_classes, device).float()  # [1, C]
            img = model.decoder(z, y)
            samples.append(img.squeeze().cpu().numpy())
    return np.array(samples)


@app.command(name="gen")
def main(digit: int = typer.Option(..., "--digit", "-d", help="Dígito 0-9")):
    """
    Generate a digit image using the trained CVAE decoder and save it to
    `reports/figures/cvae_digit{digit}.png`.
    """
    # -----------------------------------------
    start_time = time.time()
    logger.info("Running inference...")
    # -----------------------------------------

    model = model_fn()
    img = predict_fn(digit, model)

    plt.imshow(img, cmap="gray")
    plt.savefig(FIGURES_DIR / f"cvae_digit{digit}.png")

    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Inference complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()
