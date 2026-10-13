"""Download and safely extract the labelled competition training images."""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import kagglehub


ROOT = Path(__file__).resolve().parent
DOWNLOAD_DIR = ROOT / "data" / "kaggle-download"
TRAIN_DIR = ROOT / "data" / "train"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_labeled_image(name: str) -> bool:
    basename = Path(name).name.lower()
    return (
        basename.startswith(("cat.", "dog."))
        and Path(basename).suffix in IMAGE_SUFFIXES
    )


def copy_images_from_zip(archive: zipfile.ZipFile) -> int:
    """Copy only labelled cat/dog images, never arbitrary archive paths."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for entry in archive.infolist():
        if entry.is_dir() or not is_labeled_image(entry.filename):
            continue
        target = TRAIN_DIR / Path(entry.filename).name
        with archive.open(entry) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        copied += 1
    return copied


def extract_training_archives(download_path: Path) -> int:
    """Support KaggleHub returning a directory or a competition ZIP archive."""
    total = 0
    if download_path.is_file() and zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path) as outer:
            train_archives = [
                entry for entry in outer.infolist()
                if not entry.is_dir() and Path(entry.filename).name.lower() == "train.zip"
            ]
            for entry in train_archives:
                with zipfile.ZipFile(io.BytesIO(outer.read(entry))) as train_archive:
                    total += copy_images_from_zip(train_archive)
            if not train_archives:
                total += copy_images_from_zip(outer)
        return total

    if not download_path.exists():
        raise FileNotFoundError(f"KaggleHub returned a missing path: {download_path}")

    for archive_path in download_path.rglob("train.zip"):
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                total += copy_images_from_zip(archive)

    if total == 0:
        for image_path in download_path.rglob("*"):
            if image_path.is_file() and is_labeled_image(image_path.name):
                TRAIN_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, TRAIN_DIR / image_path.name)
                total += 1
    return total


def main() -> None:
    if len(list(TRAIN_DIR.glob("cat.*"))) and len(list(TRAIN_DIR.glob("dog.*"))):
        print(f"Labelled training images already exist in {TRAIN_DIR}")
        return

    try:
        kagglehub.whoami(verbose=False)
    except Exception:
        print("Sign in to Kaggle using your API token when prompted.")
        kagglehub.login()

    print("Downloading the Dogs vs. Cats competition files from Kaggle...")
    result = kagglehub.competition_download(
        "dogs-vs-cats", path="train.zip", output_dir=str(DOWNLOAD_DIR)
    )
    count = extract_training_archives(Path(result))
    cats = len(list(TRAIN_DIR.glob("cat.*")))
    dogs = len(list(TRAIN_DIR.glob("dog.*")))
    if count == 0 or not cats or not dogs:
        raise RuntimeError(
            "KaggleHub downloaded the competition files but no labelled cat/dog "
            "training images were extracted. Check Kaggle competition access and "
            "that the competition rules have been accepted."
        )
    print(f"Prepared {cats} cat images and {dogs} dog images in {TRAIN_DIR}")


if __name__ == "__main__":
    main()

