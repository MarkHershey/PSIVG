import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lang_sam import LangSAM
from PIL import Image

##### this code is adapted from the original code by Luca Medeiros
##### https://github.com/luca-medeiros/lang-segment-anything


def extract_frames_from_video(video_path, output_dir, frame_rate=8):
    """
    Extract frames from a video at specified frame rate
    frame_rate: frames per second to extract (default: 8 fps)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(
        f"  Video FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s"
    )

    # Calculate frame interval based on desired frame rate
    frame_interval = int(fps / frame_rate)

    frame_count = 0
    saved_count = 0
    frames_info = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:04d}.png"
            frame_path = os.path.join(output_dir, frame_filename)

            # Save frame directly in BGR format (OpenCV's native format)
            # This preserves the original colors without conversion issues
            cv2.imwrite(frame_path, frame)

            frames_info.append(
                {
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps,
                    "filename": frame_filename,
                    "path": frame_path,
                }
            )

            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"  Extracted {saved_count} frames at {frame_rate} FPS")
    return frames_info


def save_masks_as_npy(results, output_dir="masks_npy", frame_prefix=""):
    """Save masks from results as .npy files"""
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        masks = result["masks"]
        if masks is not None and len(masks) > 0:
            # Save each mask separately
            for j, mask in enumerate(masks):
                mask_filename = f"{output_dir}/{frame_prefix}mask_{i:04d}_{j:02d}.npy"
                np.save(mask_filename, mask)
                print(f"    Saved mask to {mask_filename}")
        else:
            print(f"    No masks found in result {i}")


def visualize_masks(
    image_pil, results, output_dir="mask_visualizations", frame_prefix=""
):
    """Visualize masks and save as PNG files"""
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        masks = result["masks"]
        if masks is not None and len(masks) > 0:
            # Convert PIL image to numpy array
            image_np = np.array(image_pil)

            # Create a figure with subplots for each mask
            num_masks = len(masks)
            fig, axes = plt.subplots(1, num_masks + 1, figsize=(5 * (num_masks + 1), 5))

            # Show original image
            axes[0].imshow(image_pil)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Show each mask
            for j, mask in enumerate(masks):
                # Create colored mask overlay
                colored_mask = np.zeros_like(image_np)

                # Ensure mask is boolean and has the right shape
                if mask.dtype != bool:
                    mask = mask.astype(bool)

                # Apply mask to colored overlay
                colored_mask[mask] = [255, 0, 0]  # Red color for mask

                # Blend with original image
                alpha = 0.5
                blended = cv2.addWeighted(image_np, 1 - alpha, colored_mask, alpha, 0)

                axes[j + 1].imshow(blended)
                axes[j + 1].set_title(f"Mask {j+1}")
                axes[j + 1].axis("off")

            plt.tight_layout()
            mask_filename = f"{output_dir}/{frame_prefix}masks_{i:04d}.png"
            plt.savefig(mask_filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    Saved mask visualization to {mask_filename}")


def visualize_boxes(
    image_pil, results, output_dir="box_visualizations", frame_prefix=""
):
    """Visualize bounding boxes and save as PNG files"""
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        boxes = result["boxes"]
        if boxes is not None and len(boxes) > 0:
            # Convert PIL image to numpy array
            image_np = np.array(image_pil)

            # Create a copy for drawing
            image_with_boxes = image_np.copy()

            # Draw each bounding box
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                # Convert to integers for drawing
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw rectangle
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f"Box {j+1}"
                cv2.putText(
                    image_with_boxes,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Save the image with boxes
            box_filename = f"{output_dir}/{frame_prefix}boxes_{i:04d}.png"
            plt.figure(figsize=(10, 8))
            plt.imshow(image_with_boxes)
            plt.title(f"Bounding Boxes - Result {i}")
            plt.axis("off")
            plt.savefig(box_filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    Saved box visualization to {box_filename}")


def process_video_frames(
    video_path,
    model,
    text_prompt,
    frame_rate=8,
    base_output_dir="video_outputs",
    video_name=None,
):
    """Process a single video: extract frames and segment each frame"""
    if video_name is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing video: {video_name}")

    # Create video-specific output directory
    # video_output_dir = os.path.join(base_output_dir, video_name)
    video_output_dir = base_output_dir
    frames_dir = os.path.join(video_output_dir, "frames")

    # Extract frames from video
    print(f"  Extracting frames from video...")
    try:
        frames_info = extract_frames_from_video(video_path, frames_dir, frame_rate)
    except Exception as e:
        print(f"  Error extracting frames: {str(e)}")
        return

    if not frames_info:
        print(f"  No frames extracted from video")
        return

    # Process each frame
    print(f"  Processing {len(frames_info)} frames...")
    for frame_idx, frame_info in enumerate(frames_info):
        frame_path = frame_info["path"]
        frame_filename = frame_info["filename"]
        timestamp = frame_info["timestamp"]

        print(
            f"    Processing frame {frame_idx+1}/{len(frames_info)}: {frame_filename} (t={timestamp:.2f}s)"
        )

        try:
            # Load and process frame
            # Convert BGR (OpenCV) to RGB (PIL) for proper color handling
            frame_bgr = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)

            results = model.predict([image_pil], [text_prompt])

            # Create frame-specific output directories
            frame_base_name = os.path.splitext(frame_filename)[0]
            masks_dir = os.path.join(video_output_dir, "masks_npy")
            mask_viz_dir = os.path.join(video_output_dir, "mask_visualizations")
            box_viz_dir = os.path.join(video_output_dir, "box_visualizations")

            # Save masks as .npy files
            save_masks_as_npy(
                results, output_dir=masks_dir, frame_prefix=frame_base_name + "_"
            )

            # Visualize masks and save as PNG
            visualize_masks(
                image_pil,
                results,
                output_dir=mask_viz_dir,
                frame_prefix=frame_base_name + "_",
            )

            # Visualize bounding boxes and save as PNG
            visualize_boxes(
                image_pil,
                results,
                output_dir=box_viz_dir,
                frame_prefix=frame_base_name + "_",
            )

        except Exception as e:
            print(f"      Error processing frame {frame_filename}: {str(e)}")
            continue

    print(f"  Completed processing video: {video_name}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Segment video frames using LangSAM")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Directory containing subdirectories, each with an input_template video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed results",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        required=True,
        help="FG text prompt file for segmentation",
    )
    parser.add_argument(
        "--frame_rate",
        type=float,
        default=8.0,
        help="Frame extraction rate in FPS (default: 8.0)",
    )

    parser.add_argument(
        "--selected_vids_file", type=str, default="None", help="Selected vids file"
    )

    args = parser.parse_args()

    # Configuration from command line arguments
    base_input_dir = args.input_folder
    base_output_dir = args.output_dir
    text_prompt = args.text_prompt
    frame_rate = args.frame_rate
    selected_vids_file = args.selected_vids_file

    # Load text prompts from file
    if not os.path.exists(text_prompt):
        raise FileNotFoundError(f"Text prompt file not found: {text_prompt}")

    with open(text_prompt, "r") as f:
        text_prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(text_prompts)} text prompts from: {text_prompt}")

    # Initialize the model once
    print("Initializing LangSAM model...")
    model = LangSAM()
    print("Model initialized successfully!")

    # Load selected videos from file or get all subdirectories
    if selected_vids_file != "None":
        if not os.path.exists(selected_vids_file):
            raise FileNotFoundError(
                f"Selected videos file not found: {selected_vids_file}"
            )
        with open(selected_vids_file, "r") as f:
            subdirs = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(
            "Selected vids file is not specified. This is required, or we could also just get all subdirectories"
        )

    print(f"Found {len(subdirs)} subdirectories to process")
    print(f"Frame extraction rate: {frame_rate} FPS")

    # Process each subdirectory
    for i, subdir in enumerate(subdirs):

        input_dir = os.path.join(base_input_dir, subdir)

        # subdir_path = os.path.join(base_input_dir, subdir)
        print(f"\n{'='*60}")
        print(f"Processing subdirectory {i+1}/{len(subdirs)}: {subdir}")
        print(f"{'='*60}")

        # Look for input_template video file in the subdirectory
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
        input_template_video = None

        for ext in video_extensions:
            potential_video = os.path.join(
                input_dir, "output_noises_background", f"input_template{ext}"
            )
            # potential_video = os.path.join(subdir_path, f"input_template{ext}")
            if os.path.exists(potential_video):
                input_template_video = potential_video
                break

        if input_template_video is None:
            print(f"  No input_template video found in {subdir}, skipping...")
            continue

        print(f"  Found video: {os.path.basename(input_template_video)}")

        # Prompt selection follows file order:
        # line N in selected_vids_file uses line N in text_prompt file.
        prompt_idx = i

        if prompt_idx < len(text_prompts):
            current_text_prompt = text_prompts[prompt_idx]
        else:
            print("prompt_idx: ", prompt_idx)
            raise ValueError(
                f"No text prompt found for subdir index {prompt_idx} ({subdir})"
            )

        print(f"  Using text prompt: '{current_text_prompt}'")

        output_dir = os.path.join(base_output_dir, subdir, "video_segmentation_outputs")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Use subdirectory name as video name for output organization
            process_video_frames(
                input_template_video,
                model,
                current_text_prompt,
                frame_rate,
                output_dir,
                video_name=subdir,
            )
            print(f"Successfully completed processing: {subdir}")
        except Exception as e:
            print(f"Error processing subdirectory {subdir}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print(f"All {len(subdirs)} subdirectories processed!")
    print(f"Output saved to: {base_output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
