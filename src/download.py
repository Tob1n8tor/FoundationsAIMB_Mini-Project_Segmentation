import urllib.request
from pathlib import Path

from tqdm import tqdm

from src.config import (
    CHECKPOINTS_DIR,
    DATASET_JSON_PATH,
    DATASET_JSON_URL,
    MODEL_CONFIGS,
)


class _DownloadProgressBar(tqdm):
    """Helper hook to display a tqdm progress bar during urllib downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: Path, force: bool = False) -> Path:
    """Download a file from `url` to `dest_path` if it does not already exist."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not force:
        print(f"[SKIP] Already exists: {dest_path.name}")
        return dest_path

    print(f"[DOWNLOADING] {dest_path.name} from {url} ...")
    with _DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=dest_path.name
    ) as t:
        urllib.request.urlretrieve(url, filename=str(dest_path), reporthook=t.update_to)
    print(f"[DONE] Saved to {dest_path}")
    return dest_path


def download_dataset_json(force: bool = False) -> Path:
    """Download the official MONAI BTCV dataset_0.json split file."""
    return download_file(DATASET_JSON_URL, DATASET_JSON_PATH, force=force)


def download_pretrained_weights(
    model_keys=("base", "small", "tiny"), force: bool = False
) -> dict[str, Path]:
    """Download the official pretrained checkpoints for Base, Small, and/or Tiny."""
    paths = {}
    for key in model_keys:
        cfg = MODEL_CONFIGS[key]
        dest = CHECKPOINTS_DIR / cfg["filename"]
        paths[key] = download_file(cfg["url"], dest, force=force)
    return paths
