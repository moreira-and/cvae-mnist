import csv
from pathlib import Path
from typing import Optional

import torch
from torch import nn
import typer

from cvae.config import REPORTS_DIR, device, logger, params
from cvae.service.models.nn_cvae import nn_CVAE
from cvae.service.utils import get_data_loader, one_hot

app = typer.Typer(help="Evaluate CVAE reconstructions and generations")


def _load_model() -> nn.Module:
    model = nn_CVAE(
        input_channels=1,
        latent_dim=params.model.latent_dim,
        num_classes=params.dataset.num_classes,
    ).to(device)

    model_path = (Path(REPORTS_DIR).parent / "models" / "model.pth").resolve()
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch < 2.4 fallback (no weights_only arg)
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _bce(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")


def _mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(x_hat, x, reduction="sum")


def _try_ssim_psnr(
    x_hat: torch.Tensor, x: torch.Tensor
) -> tuple[Optional[float], Optional[float]]:
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except Exception:
        return None, None

    # tensors to numpy (N, H, W)
    x_np = x.detach().cpu().numpy().squeeze()
    xh_np = x_hat.detach().cpu().numpy().squeeze()

    # ensure 3D arrays (N, H, W)
    if x_np.ndim == 2:
        x_np = x_np[None, ...]
        xh_np = xh_np[None, ...]

    ssim_vals = []
    psnr_vals = []
    for a, b in zip(x_np, xh_np):
        ssim_vals.append(structural_similarity(a, b, data_range=1.0))
        psnr_vals.append(peak_signal_noise_ratio(a, b, data_range=1.0))

    return float(sum(ssim_vals) / len(ssim_vals)), float(sum(psnr_vals) / len(psnr_vals))


@app.command()
def main(
    max_batches: Optional[int] = None,
    sample_ssim: int = 128,
    write_csv: bool = True,
):
    """Evaluate reconstruction quality on the test split.

    Computes BCE and MSE over the test set. If scikit-image is installed, also
    computes SSIM and PSNR on up to `sample_ssim` samples.
    """

    logger.info("Loading test data and model for evaluation...")
    test_loader = get_data_loader(params.dataset.test_batch_size, train=False)
    model = _load_model()

    total_bce = 0.0
    total_mse = 0.0
    n_images = 0

    # For SSIM/PSNR sub-sample
    ssim_accum = []
    psnr_accum = []
    ssim_budget = int(sample_ssim) if sample_ssim else 0

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_loader):
            x = x.to(device)
            y = one_hot(labels, params.dataset.num_classes, device).float()

            x_hat, mu, logvar = model(x, y)

            total_bce += float(_bce(x_hat, x).item())
            total_mse += float(_mse(x_hat, x).item())
            n_images += x.shape[0]

            # SSIM/PSNR on a sub-sample
            if ssim_budget > 0:
                k = min(ssim_budget, x.shape[0])
                ssim, psnr = _try_ssim_psnr(x_hat[:k], x[:k])
                if ssim is not None:
                    ssim_accum.append(ssim)
                if psnr is not None:
                    psnr_accum.append(psnr)
                ssim_budget -= k

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    # Normalize per-image metrics
    bce_per_image = total_bce / max(n_images, 1)
    mse_per_image = total_mse / max(n_images, 1)
    ssim_mean = float(sum(ssim_accum) / len(ssim_accum)) if ssim_accum else None
    psnr_mean = float(sum(psnr_accum) / len(psnr_accum)) if psnr_accum else None

    logger.info(
        f"Eval metrics — images: {n_images}, BCE/img: {bce_per_image:.4f}, MSE/img: {mse_per_image:.6f}, "
        + (
            f"SSIM: {ssim_mean:.4f}, PSNR: {psnr_mean:.2f}dB"
            if ssim_mean is not None
            else "SSIM/PSNR: (skipped)"
        )
    )

    # Save CSV
    if write_csv:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = REPORTS_DIR / "eval_metrics.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["n_images", "bce_per_image", "mse_per_image", "ssim", "psnr"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "n_images": n_images,
                    "bce_per_image": round(bce_per_image, 6),
                    "mse_per_image": round(mse_per_image, 8),
                    "ssim": None if ssim_mean is None else round(ssim_mean, 6),
                    "psnr": None if psnr_mean is None else round(psnr_mean, 6),
                }
            )
        logger.success(f"Saved metrics to {out_csv}")


if __name__ == "__main__":
    app()
