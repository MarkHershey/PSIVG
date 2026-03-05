import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from rich import print

try:
    import cv2
except ImportError:
    cv2 = None


def ffmpeg_is_available():
    return shutil.which("ffmpeg") is not None


def opencv_is_available():
    return cv2 is not None


def make_video_from_frames_dir(
    input_folder: str, output_path: str, fps: float = 10
) -> Tuple[int, Tuple[int, int]]:
    if ffmpeg_is_available():
        return make_video_from_frames_dir_ffmpeg(input_folder, output_path, fps)
    elif opencv_is_available():
        return make_video_from_frames_dir_opencv(input_folder, output_path, fps)
    else:
        print(
            "Neither [bold cyan]ffmpeg[/bold cyan] nor [bold cyan]OpenCV[/bold cyan] is available in the environment. Please install one of them and try again."
        )
        raise RuntimeError("Neither ffmpeg nor opencv is available.")


def make_video_from_frames_dir_opencv(
    input_folder: str, output_path: str, fps: float = 10
) -> Tuple[int, Tuple[int, int]]:
    """
    Build a video from a folder of RGB frame images (e.g., PNG/JPG), whose filenames
    contain sequence numbers (like 0001.png, frame_12.jpg, etc.).

    Required args:
        input_folder : str
            Directory containing the frame image files.
        fps : float
            Frames per second for the output video. Must be > 0.
        output_path : str
            Output video path. The codec is chosen based on the file extension.

    Behavior & assumptions:
      - Frames are discovered by image extensions (png/jpg/jpeg/bmp/tif/tiff).
      - Frames are sorted by the numeric parts of filenames (natural order).
      - If any frame's size differs from the first frame, it is resized to match.
      - Missing numbers are fine; only discovered files are used.
      - Common codecs are selected automatically based on output extension:
          .mp4/.mov/.m4v -> 'mp4v'
          .avi            -> 'XVID'
          otherwise       -> 'mp4v' (best-effort)

    Returns:
        (num_frames_written, (width, height))

    Raises:
        FileNotFoundError: if no readable image frames are found.
        ValueError: for invalid fps or if no frames could be written.
        RuntimeError: if the video writer cannot be opened.

    Example:
        make_video_from_frames("frames/", 30, "out.mp4")
    """
    # --- Validate inputs ---
    if fps is None or fps <= 0:
        raise ValueError(f"'fps' must be a positive number, got {fps!r}")

    input_dir = Path(input_folder)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input folder not found or not a directory: {input_folder}"
        )

    output_path = str(output_path)  # in case a Path was passed
    out_ext = Path(output_path).suffix.lower()

    # Choose a reasonable default codec based on extension
    if out_ext in {".mp4", ".mov", ".m4v"}:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif out_ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        # Fallback to mp4v; container support depends on your OpenCV build/FFmpeg
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # --- Gather & sort frames ---
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def numeric_key(p: Path) -> List[int | str]:
        # Extract all digit runs; sort lexicographically by those ints, then by name.
        nums = re.findall(r"\d+", p.stem)
        return [int(n) for n in nums] + [p.stem] if nums else [float("inf"), p.stem]

    frame_paths: List[Path] = sorted(
        (
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ),
        key=numeric_key,
    )

    if not frame_paths:
        raise FileNotFoundError(
            f"No image frames found in '{input_folder}'. "
            f"Supported extensions: {', '.join(sorted(valid_exts))}"
        )

    # --- Probe first frame to get size ---
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]

    # --- Prepare writer ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for '{output_path}'. "
            f"Try a different extension/codec (e.g., '.mp4', '.avi')."
        )

    # --- Write frames ---
    frames_written = 0

    # Write the first (already loaded) frame
    if first.shape[1] != width or first.shape[0] != height:
        first = cv2.resize(first, (width, height), interpolation=cv2.INTER_AREA)
    writer.write(first)
    frames_written += 1

    # Remaining frames
    for p in frame_paths[1:]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            # Skip unreadable frames
            continue
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(img)
        frames_written += 1

    writer.release()

    if frames_written == 0:
        raise ValueError("No frames were written to the video (all reads failed?).")

    return frames_written, (width, height)


def make_video_from_frames_dir_ffmpeg(
    input_folder: str, output_path: str, fps: float = 10
) -> Tuple[int, Tuple[int, int]]:
    """
    Build a video from a folder of RGB frame images and write a highly compatible MP4.

    Changes vs. the original:
      - Uses ffmpeg (libx264 + yuv420p) for broad player/browser support.
      - Moves 'moov' atom to the front (-movflags +faststart) for streaming/seekability.
      - Adds a silent AAC track to avoid audio-less edge cases.
      - Rescales frames to the first frame's size (same behavior as before).

    Returns:
        (num_frames_written, (width, height))

    Raises:
        FileNotFoundError, ValueError, RuntimeError
    """
    # --- Validate inputs ---
    if fps is None or fps <= 0:
        raise ValueError(f"'fps' must be a positive number, got {fps!r}")

    input_dir = Path(input_folder)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input folder not found or not a directory: {input_folder}"
        )

    output_path = str(output_path)  # in case a Path was passed
    out_ext = Path(output_path).suffix.lower()
    if out_ext not in {".mp4", ".mov", ".m4v"}:
        # You *can* target other containers, but H.264+yuv420p+faststart is designed for MP4/MOV/M4V.
        # Using other extensions may reduce compatibility.
        pass

    # --- Gather & sort frames ---
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def numeric_key(p: Path) -> List[int | str]:
        nums = re.findall(r"\d+", p.stem)
        return [int(n) for n in nums] + [p.stem] if nums else [float("inf"), p.stem]

    frame_paths: List[Path] = sorted(
        (
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ),
        key=numeric_key,
    )
    if not frame_paths:
        raise FileNotFoundError(
            f"No image frames found in '{input_folder}'. Supported extensions: {', '.join(sorted(valid_exts))}"
        )

    # --- Probe first frame for size ---
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]

    # --- Ensure ffmpeg is available ---
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is required for H.264 + faststart output but was not found on PATH. "
            "Please install ffmpeg (https://ffmpeg.org) and try again."
        )

    # --- Build the ffmpeg input list (sorted), set per-frame durations from fps ---
    n = len(frame_paths)
    per_frame_sec = 1.0 / float(fps)
    total_duration = n * per_frame_sec

    def _esc_for_concat(path: Path) -> str:
        # ffmpeg concat demuxer escaping: escape backslashes and single quotes, use absolute path.
        s = str(path.resolve())
        s = s.replace("\\", "\\\\").replace("'", r"'\''")
        return s

    # --- Assemble and run ffmpeg ---
    with tempfile.TemporaryDirectory() as tmpdir:
        list_txt = Path(tmpdir) / "frames.txt"

        if n == 1:
            # Single-frame video: loop the image for the desired duration.
            single = frame_paths[0].resolve()
            cmd = [
                ffmpeg_path,
                "-y",
                "-loop",
                "1",
                "-t",
                f"{total_duration:.9f}",
                "-i",
                str(single),
                "-f",
                "lavfi",
                "-t",
                f"{total_duration:.9f}",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-vf",
                f"scale={width}:{height}:flags=lanczos,setsar=1",
                "-r",
                f"{fps:.6f}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-level",
                "4.0",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
                "-shortest",
                output_path,
            ]
        else:
            # Concat demuxer with explicit durations for stable timing & ordering.
            with list_txt.open("w", encoding="utf-8", newline="\n") as f:
                for i, p in enumerate(frame_paths):
                    f.write(f"file '{_esc_for_concat(p)}'\n")
                    if i < n - 1:
                        f.write(f"duration {per_frame_sec:.9f}\n")
                # Repeat the last frame once (no duration) per concat demuxer rules.
                f.write(f"file '{_esc_for_concat(frame_paths[-1])}'\n")

            cmd = [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_txt),
                "-f",
                "lavfi",
                "-t",
                f"{total_duration:.9f}",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-vf",
                f"scale={width}:{height}:flags=lanczos,setsar=1",
                "-r",
                f"{fps:.6f}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-level",
                "4.0",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
                "-shortest",
                output_path,
            ]

        try:
            # Capture stderr for helpful errors if something goes wrong.
            proc = subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "ffmpeg failed while creating the video.\n\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"stderr:\n{e.stderr.decode(errors='replace')}"
            )

    # --- Return info consistent with original signature ---
    return n, (width, height)
