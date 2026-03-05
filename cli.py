import argparse
import hashlib
import json
import shutil
from pathlib import Path

from rich import print

import psivg.constants as C

SUPPORTED_VIDEO_FORMATS = {".mp4"}


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_same_video(video_path1: Path, video_path2: Path) -> bool:
    if video_path1 == video_path2:
        return True
    if not video_path1.is_file() or not video_path2.is_file():
        return False
    if video_path1.stat().st_size != video_path2.stat().st_size:
        return False
    if md5sum(video_path1) != md5sum(video_path2):
        return False
    return True


def valid_sample_id(sample_id: str) -> bool:
    if sample_id.isdigit() and len(sample_id) == 4:
        return True
    else:
        return False


def _create_empty_meta_json(meta_json_path: Path):
    with open(meta_json_path, "w") as f:
        json.dump(
            {
                "video_prompt": "",
                "fg_prompt": "",
                "primary": "",
            },
            f,
            indent=4,
        )


def _valid_meta_json(meta_json_path: Path):
    if not meta_json_path.exists():
        return False
    with open(meta_json_path, "r") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        return False
    if not all(key in meta for key in ["video_prompt", "fg_prompt", "primary"]):
        return False
    if (
        not isinstance(meta["video_prompt"], str)
        or not isinstance(meta["fg_prompt"], str)
        or not isinstance(meta["primary"], str)
    ):
        return False
    if meta["primary"] == "" or meta["fg_prompt"] == "" or meta["video_prompt"] == "":
        return False
    return True


def command_line_entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="Path to the video file (or the sample ID) to be processed",
        required=True,
    )
    args = parser.parse_args()

    if "." in args.video:
        # User provided a video file path directly
        input_video_path = Path(args.video).resolve()
        # Assume the sample ID is the video file name
        sample_id = input_video_path.stem

        # The sample ID must be a 4-digit number string
        if not valid_sample_id(sample_id):
            print(
                f"[red bold]✗[/red bold] Video file name '{input_video_path}' is not a 4-digit number string, please rename it to a 4-digit number string"
            )
            exit(1)

        assert (
            input_video_path.exists()
        ), f"Provided video file path '{input_video_path}' does not exist"
        assert (
            input_video_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS
        ), f"Provided video format '{input_video_path.suffix.lower()}' is not supported, please provide one of the following formats: {SUPPORTED_VIDEO_FORMATS}"

        # Check if the video file already exists under DATA_ROOT
        target_video_path = C.INPUT_VIDEOS_DIR / f"{sample_id}.mp4"

        if target_video_path.exists():
            if is_same_video(input_video_path, target_video_path):
                ...  # already taken as input before
            else:
                # there was a processed video with the same sample ID, but a different video
                print(
                    f"[red]✗[/red] Sample ID '{sample_id}' already occupied by a different video, try assigning a different ID"
                )
                exit(1)
        else:
            # copy the input video to the input videos directory under DATA_ROOT
            shutil.copy(input_video_path, target_video_path)

    else:
        # User provided a sample ID directly
        sample_id = str(args.video).strip()
        input_video_path = None
        if not valid_sample_id(sample_id):
            print(
                f"[red bold]✗[/red bold] Sample ID '{sample_id}' is not a 4-digit number string, please provide a 4-digit number string"
            )
            exit(1)

        # try to find the video file under DATA_ROOT given the user-provided sample ID
        target_video_path = C.INPUT_VIDEOS_DIR / f"{sample_id}.mp4"
        if not target_video_path.exists():
            print(
                f"[red bold]✗[/red bold] Sample ID '{sample_id}' not found, please provide a known sample ID"
            )

    meta_json_path = C.INPUT_META_DIR / f"{sample_id}.json"
    if not meta_json_path.exists():
        if isinstance(input_video_path, Path):
            fall_back_meta_json_path = input_video_path.parent / f"{sample_id}.json"
            if fall_back_meta_json_path.exists():
                shutil.copy(fall_back_meta_json_path, meta_json_path)

    if not meta_json_path.exists():
        _create_empty_meta_json(meta_json_path)
        print(
            f"[yellow bold]⚠[/yellow bold] Created empty metadata file at '{meta_json_path}'"
        )
        print(
            "[yellow bold]⚠[/yellow bold] Please fill in the metadata file according to instructions.md"
        )
        exit(1)

    # validate and process the metadata file
    if not _valid_meta_json(meta_json_path):
        print(f"[red bold]✗[/red bold] Metadata file is invalid: '{meta_json_path}'")
        print(
            "[yellow bold]⚠[/yellow bold] Please fill in the metadata file according to instructions.md"
        )
        exit(1)

    with open(meta_json_path, "r") as f:
        meta = json.load(f)
        video_prompt = meta["video_prompt"]
        fg_prompt = meta["fg_prompt"]

    # author prompt file required by part 3 and part 4
    with open(C.INPUT_PROMPTS_DIR / f"{sample_id}.txt", "w") as f:
        f.write(f"{video_prompt}")

    # author prompt file required by part 3 and part 4
    with open(C.INPUT_PROMPTS_DIR / f"{sample_id}_fg.txt", "w") as f:
        f.write(f"{fg_prompt}")

    return sample_id
