#!/usr/bin/env python3
"""
Script to extract frames from a video file - either specific frames or all frames.

Usage:
    # Extract specific frames
    python extract_frames.py <video_path> <output_folder> --indices <idx1> <idx2> <idx3> ... [--verbose]
    python extract_frames.py video.mp4 ./frames --indices 0 50 99 150
    python extract_frames.py video.mp4 ./frames --indices 0 50 99 150 --verbose

    # Extract all frames
    python extract_frames.py <video_path> <output_folder> --all [--verbose]
    python extract_frames.py video.mp4 ./frames --all
    python extract_frames.py video.mp4 ./frames --all --verbose

The script will:
- Extract either specific frame indices or all frames from the video
- Skip indices that don't exist in the video (when using --indices)
- Save frames with padded zero naming (e.g., 00000.jpg, 00001.jpg)
- Create the output directory if it doesn't exist
"""

import sys
from pathlib import Path
from typing import List, Tuple

from rich import print

try:
    import cv2
except ImportError:
    print(
        "Error: OpenCV is required. Install it with: pip install opencv-python",
        file=sys.stderr,
    )
    sys.exit(1)

# Constants
DEFAULT_PADDING = 5
DEFAULT_IMAGE_FORMAT = "jpg"
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
}


class FrameExtractionError(Exception):
    """Custom exception for frame extraction errors."""

    pass


def get_video_in_dir(video_dir: Path) -> List[Path]:
    """
    Get all video files in a directory.
    """
    return [
        f
        for f in video_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]


def validate_video_file(video_path: str) -> None:
    """
    Validate that the video file exists and has a supported extension.

    Args:
        video_path: Path to the video file

    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video file has an unsupported extension
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_ext = video_path.suffix.lower()
    if file_ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format: {file_ext}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )

    return video_path


def validate_frame_indices(frame_indices: List[int]) -> List[int]:
    """
    Validate and clean frame indices.

    Args:
        frame_indices: List of frame indices to validate

    Returns:
        List of unique, non-negative frame indices

    Raises:
        ValueError: If no valid indices are provided
    """
    if not frame_indices:
        raise ValueError("No frame indices provided")

    # Remove duplicates and negative indices, then sort
    valid_indices = sorted(set(idx for idx in frame_indices if idx >= 0))

    if not valid_indices:
        raise ValueError("No valid frame indices provided (all indices were negative)")

    return valid_indices


def extract_all_frames(
    video_path: str,
    output_folder: str = None,
    verbose: bool = True,
    image_format: str = DEFAULT_IMAGE_FORMAT,
) -> Tuple[int, int]:
    """
    Extract all frames from a video file.

    Args:
        video_path: Path to the input video file
        output_folder: Path to the output folder for saved frames
        verbose: Whether to print detailed progress information
        image_format: Output image format (jpg, png, etc.)

    Returns:
        Tuple of (success_count, total_video_frames)

    Raises:
        FrameExtractionError: If video cannot be opened or processed
        FileNotFoundError: If video file doesn't exist
        ValueError: If invalid parameters are provided
    """
    # Validate inputs
    video_path = validate_video_file(video_path)

    # Create output directory if it doesn't exist
    if not output_folder:
        video_name = video_path.stem
        output_folder = video_path.parent / video_name

    output_path = Path(output_folder)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise FrameExtractionError(
            f"Permission denied: Cannot create output directory {output_folder}"
        )
    except OSError as e:
        raise FrameExtractionError(
            f"Cannot create output directory {output_folder}: {e}"
        )

    # Open video file with proper resource management
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FrameExtractionError(f"Cannot open video file: {video_path}")

    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if verbose:
            print(
                f"Video properties: {total_frames} frames, {fps:.2f} FPS, {width}x{height}"
            )

        # Validate total frames
        if total_frames <= 0:
            raise FrameExtractionError(
                "Video appears to have no frames or frame count cannot be determined"
            )

        if verbose:
            print(f"Extracting all {total_frames} frames")

        # Determine padding for frame naming
        padding = max(len(str(total_frames - 1)), DEFAULT_PADDING)

        success_count = 0
        frame_index = 0

        # Process all frames sequentially
        while True:
            try:
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    # End of video reached
                    break

                # Validate frame data
                if frame is None or frame.size == 0:
                    if verbose:
                        print(f"Error: Frame {frame_index} is empty")
                    frame_index += 1
                    continue

                # Generate output filename with padded zeros
                filename = f"{frame_index:0{padding}d}.{image_format}"
                output_file_path = output_path / filename

                # Save the frame with error checking
                save_success = cv2.imwrite(str(output_file_path), frame)
                if save_success:
                    if verbose:
                        print(f"Saved frame {frame_index} -> {output_file_path}")
                    success_count += 1
                else:
                    if verbose:
                        print(
                            f"Error: Failed to save frame {frame_index} to {output_file_path}"
                        )

                frame_index += 1

            except Exception as e:
                if verbose:
                    print(f"Error processing frame {frame_index}: {e}")
                frame_index += 1
                continue

        return success_count, total_frames

    finally:
        # Ensure video capture is always released
        cap.release()


def extract_frames_by_indices(
    video_path: str,
    output_folder: str = None,
    frame_indices: List[int] = [0, 5, 10, 15, 20],
    verbose: bool = True,
    image_format: str = DEFAULT_IMAGE_FORMAT,
) -> Tuple[int, int, int]:
    """
    Extract specific frames from a video file based on frame indices.

    Args:
        video_path: Path to the input video file
        output_folder: Path to the output folder for saved frames
        frame_indices: List of frame indices to extract
        verbose: Whether to print detailed progress information
        image_format: Output image format (jpg, png, etc.)

    Returns:
        Tuple of (success_count, total_requested, total_video_frames)

    Raises:
        FrameExtractionError: If video cannot be opened or processed
        FileNotFoundError: If video file doesn't exist
        ValueError: If invalid parameters are provided
    """
    # Validate inputs
    video_path = validate_video_file(video_path)
    frame_indices = validate_frame_indices(frame_indices)

    # Create output directory if it doesn't exist
    if not output_folder:
        video_name = video_path.stem
        output_folder = video_path.parent / video_name

    output_path = Path(output_folder)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise FrameExtractionError(
            f"Permission denied: Cannot create output directory {output_folder}"
        )
    except OSError as e:
        raise FrameExtractionError(
            f"Cannot create output directory {output_folder}: {e}"
        )

    # Open video file with proper resource management
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FrameExtractionError(f"Cannot open video file: {video_path}")

    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if verbose:
            print(
                f"Video properties: {total_frames} frames, {fps:.2f} FPS, {width}x{height}"
            )

        # Validate total frames
        if total_frames <= 0:
            raise FrameExtractionError(
                "Video appears to have no frames or frame count cannot be determined"
            )

        # Filter out invalid indices
        valid_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]
        invalid_indices = [
            idx for idx in frame_indices if idx < 0 or idx >= total_frames
        ]

        if invalid_indices and verbose:
            print(
                f"Warning: Ignoring {len(invalid_indices)} invalid frame indices: {invalid_indices}"
            )

        if not valid_indices:
            if verbose:
                print("No valid frame indices to extract")
            return 0, len(frame_indices), total_frames

        if verbose:
            print(f"Extracting {len(valid_indices)} frames: {valid_indices}")

        # Determine padding for frame naming
        max_index = max(valid_indices)
        padding = max(len(str(max_index)), DEFAULT_PADDING)

        success_count = 0
        last_frame_pos = -1

        # Process frames in sorted order for efficiency
        for target_frame in sorted(valid_indices):
            try:
                # Only seek if necessary (optimization for sequential access)
                if target_frame != last_frame_pos + 1:
                    success = cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    if not success:
                        if verbose:
                            print(f"Warning: Could not seek to frame {target_frame}")
                        continue

                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    if verbose:
                        print(f"Error: Could not read frame {target_frame}")
                    continue

                # Validate frame data
                if frame is None or frame.size == 0:
                    if verbose:
                        print(f"Error: Frame {target_frame} is empty")
                    continue

                # Generate output filename with padded zeros
                filename = f"{target_frame:0{padding}d}.{image_format}"
                output_file_path = output_path / filename

                # Save the frame with error checking
                save_success = cv2.imwrite(str(output_file_path), frame)
                if save_success:
                    if verbose:
                        print(f"Saved frame {target_frame} -> {output_file_path}")
                    success_count += 1
                else:
                    if verbose:
                        print(
                            f"Error: Failed to save frame {target_frame} to {output_file_path}"
                        )

                last_frame_pos = target_frame

            except Exception as e:
                if verbose:
                    print(f"Error processing frame {target_frame}: {e}")
                continue

        return success_count, len(frame_indices), total_frames

    finally:
        # Ensure video capture is always released
        cap.release()
