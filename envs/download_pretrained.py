#!/usr/bin/env python3

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PRETRAINED_URLS = [
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip",
    # "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    "https://huggingface.co/TencentARC/InstantMesh/resolve/main/diffusion_pytorch_model.bin",
    "https://huggingface.co/TencentARC/InstantMesh/resolve/main/instant_mesh_large.ckpt",
    "https://huggingface.co/Eyeline-Labs/Go-with-the-Flow/resolve/main/I2V5B_final_i38800_nearest_lora_weights.safetensors",
]

COGVIDEO_URL = "https://huggingface.co/zai-org/CogVideoX-5b-I2V"

SUPERGLUE_URLS = [
    "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_indoor.pth",
    "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_outdoor.pth",
    "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superpoint_v1.pth",
]


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    if not name:
        raise ValueError(f"Could not derive filename from URL: {url}")
    return name


def download_file(
    url: str,
    destination: Path,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if destination.exists() and not force:
        print(f"Skipping existing file: {destination}")
        return

    if dry_run:
        print(f"[dry-run] Download {url} -> {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    print(f"Downloading {url}")
    req = Request(url, headers={"User-Agent": "PSIVG-downloader/1.0"})
    with urlopen(req) as response, tmp_path.open("wb") as out_f:
        shutil.copyfileobj(response, out_f)
    tmp_path.replace(destination)
    print(f"Saved: {destination}")


def clone_repo(
    url: str,
    destination: Path,
    dry_run: bool = False,
) -> None:
    if destination.exists():
        print(f"Skipping existing repo: {destination}")
        return
    if dry_run:
        print(f"[dry-run] Clone {url} -> {destination}")
        return
    print(f"Cloning {url}")
    subprocess.run(["git", "clone", url, str(destination)], check=True)
    print(f"Cloned: {destination}")


def extract_zip(
    zip_path: Path,
    extract_to: Path,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if dry_run:
        print(f"[dry-run] Extract {zip_path} -> {extract_to}")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file does not exist: {zip_path}")

    if force:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                member_path = extract_to / member
                if member_path.exists():
                    if member_path.is_dir():
                        shutil.rmtree(member_path)
                    else:
                        member_path.unlink()

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Extracted: {zip_path} -> {extract_to}")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    pretrained_dir = repo_root / "pretrained_models"

    pretrained_dir.mkdir(parents=True, exist_ok=True)

    for url in PRETRAINED_URLS:
        out_name = filename_from_url(url)
        download_file(url, pretrained_dir / out_name)

    extract_zip(
        pretrained_dir / "big-lama.zip",
        pretrained_dir,
    )

    for url in SUPERGLUE_URLS:
        out_name = filename_from_url(url)
        download_file(url, pretrained_dir / out_name)

    clone_repo(COGVIDEO_URL, pretrained_dir / "CogVideoX-5b-I2V")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
