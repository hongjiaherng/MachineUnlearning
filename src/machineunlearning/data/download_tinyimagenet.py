import argparse
import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def log(msg: str, verbose: bool):
    if verbose:
        print(msg)


def is_dataset_ready(extracted_dir: Path) -> bool:
    return extracted_dir.exists() and (extracted_dir / "train").exists() and (extracted_dir / "val").exists()


def download(data_dir: Path, zip_path: Path, verbose: bool):
    data_dir.mkdir(parents=True, exist_ok=True)

    if zip_path.exists():
        log("Zip already downloaded, skipping.", verbose)
        return

    print("Downloading TinyImageNet...")
    urllib.request.urlretrieve(URL, zip_path)
    log(f"Downloaded to {zip_path}", verbose)


def extract(data_dir: Path, zip_path: Path, extracted_dir: Path, verbose: bool):
    if extracted_dir.exists():
        log("Dataset already extracted, skipping.", verbose)
        return

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    log(f"Extracted to {extracted_dir}", verbose)


def is_val_preprocessed(extracted_dir: Path) -> bool:
    return not (extracted_dir / "val" / "images").exists()


def preprocess_val(extracted_dir: Path, verbose: bool):
    if is_val_preprocessed(extracted_dir):
        log("Validation set already preprocessed, skipping.", verbose)
        return

    print("Preprocessing validation set...")

    root = extracted_dir / "val"
    annotations = root / "val_annotations.txt"
    images_root = root / "images"

    with open(annotations) as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            name, klass = row[0], row[1]

            src = images_root / name
            dst_dir = root / klass / "images"
            dst = dst_dir / name

            if src.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)

            if verbose and i % 500 == 0:
                print(f"Processed {i} images...")

    images_root.rmdir()
    log("Validation preprocessing complete.", verbose)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare TinyImageNet dataset")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./data"),
        help="Directory to store dataset (default: ./data)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    data_dir = args.root.expanduser().resolve()
    zip_path = data_dir / "tiny-imagenet-200.zip"
    extracted_dir = data_dir / "tiny-imagenet-200"

    if is_dataset_ready(extracted_dir) and is_val_preprocessed(extracted_dir):
        print("Dataset already ready!")
        print("Path:", extracted_dir)
        return

    log(f"Using data directory: {data_dir}", args.verbose)

    download(data_dir, zip_path, args.verbose)
    extract(data_dir, zip_path, extracted_dir, args.verbose)
    preprocess_val(extracted_dir, args.verbose)

    print("Done!")
    print("Dataset at:", extracted_dir)
