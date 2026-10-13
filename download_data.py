"""Download a public cat-and-dog image dataset and prepare labelled images."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import kagglehub
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
DOWNLOAD_DIR = ROOT / "data" / "kaggle-download"
TRAIN_DIR = ROOT / "data" / "train"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DATASET_HANDLE = "tongpython/cat-and-dog"


def infer_class(path: Path) -> str | None:
    """Infer the class from either a class directory or a labelled filename."""
    name = path.name.lower()
    if name.startswith("cat."):
        return "cat"
    if name.startswith("dog."):
        return "dog"
    for parent in path.parents:
        folder = parent.name.lower()
        if folder in {"cat", "cats"}:
            return "cat"
        if folder in {"dog", "dogs"}:
            return "dog"
    return None


def prepare_local_archive(archive_path: Path) -> bool:
    """Use an already-downloaded Kaggle ZIP without uploading or copying it."""
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile):
        return False

    with archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
        has_expected_layout = any(
            "training_set" in Path(item.filename).parts
            and any(part.lower() in {"cats", "cat", "dogs", "dog"}
                    for part in Path(item.filename).parts[:-1])
            for item in entries
        )
        has_competition_names = any(
            Path(item.filename).name.lower().startswith(("cat.", "dog."))
            for item in entries
        )
        if not has_expected_layout and not has_competition_names:
            return False

        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        counts = {"cat": 0, "dog": 0}
        for item in entries:
            member = Path(item.filename)
            if member.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            parts = [part.lower() for part in member.parts]
            if "training_set" in parts:
                label = next(
                    ("cat" if part in {"cat", "cats"} else "dog"
                     for part in parts[:-1] if part in {"cat", "cats", "dog", "dogs"}),
                    None,
                )
            else:
                label = infer_class(member)
            if label is None:
                continue
            try:
                with archive.open(item) as source, Image.open(source) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError, ValueError, zipfile.BadZipFile):
                print(f"Skipping unreadable image: {member.name}")
                continue
            counts[label] += 1
            target = TRAIN_DIR / f"{label}.{counts[label]}{member.suffix.lower()}"
            with archive.open(item) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)

        if min(counts.values()) >= 2:
            print(
                f"Prepared {counts['cat']} cat images and {counts['dog']} dog images "
                f"from {archive_path}"
            )
            return True
        return False


def main() -> None:
    cats_existing = list(TRAIN_DIR.glob("cat.*"))
    dogs_existing = list(TRAIN_DIR.glob("dog.*"))
    if cats_existing and dogs_existing:
        print(f"Labelled training images already exist in {TRAIN_DIR}")
        return

    downloads = Path.home() / "Downloads"
    for archive_path in sorted(
        downloads.glob("*.zip"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        if prepare_local_archive(archive_path):
            return

    try:
        kagglehub.whoami(verbose=False)
    except Exception:
        print("Sign in to Kaggle using your API token when prompted.")
        kagglehub.login()

    print(f"Downloading public Kaggle dataset: {DATASET_HANDLE}")
    result = Path(
        kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(DOWNLOAD_DIR))
    )
    if not result.exists():
        raise FileNotFoundError(f"KaggleHub returned a missing path: {result}")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"cat": 0, "dog": 0}
    for source in result.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = infer_class(source)
        if label is None:
            continue
        try:
            with Image.open(source) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            print(f"Skipping unreadable image: {source.name}")
            continue
        counts[label] += 1
        target = TRAIN_DIR / f"{label}.{counts[label]}{source.suffix.lower()}"
        shutil.copy2(source, target)

    if min(counts.values()) < 2:
        raise RuntimeError(
            "The public Kaggle dataset download did not contain enough labelled "
            "cat and dog images. Check the Kaggle token and internet connection."
        )
    print(
        f"Prepared {counts['cat']} cat images and {counts['dog']} dog images "
        f"in {TRAIN_DIR}"
    )


if __name__ == "__main__":
    main()
