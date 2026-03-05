import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rp
import torch
from tqdm import tqdm

rp.r._pip_import_autoyes = True  # Automatically install missing packages

rp.pip_import("fire")
rp.git_import(
    "CommonSource"
)  # If missing, installs code from https://github.com/RyannDaGreat/CommonSource
import sys

import rp.git.CommonSource.noise_warp as nw

sys.path.append(rp.get_path_parent(__file__))
from rp.git.CommonSource.noise_warp import NoiseWarper


def load_segmentation_masks(masks_dir, num_frames):
    """
    Load segmentation masks for all frames.

    Args:
        masks_dir: Directory containing mask .npy files
        num_frames: Number of frames to load masks for

    Returns:
        List of lists: masks_per_frame[frame_idx][mask_idx] = mask_array
    """
    masks_per_frame = []

    for frame_idx in range(num_frames):
        frame_masks = []

        # Look for all masks for this frame
        mask_pattern = f"mask_{frame_idx:04d}_*.npy"
        mask_files = glob.glob(os.path.join(masks_dir, mask_pattern))

        if mask_files:
            # Sort by mask index
            mask_files.sort()

            for mask_file in mask_files:
                mask = np.load(mask_file)
                frame_masks.append(mask)

        masks_per_frame.append(frame_masks)

    return masks_per_frame


def create_combined_mask(masks, target_shape):
    """
    Combine multiple masks into a single foreground mask.

    Args:
        masks: List of individual masks
        target_shape: Target shape (H, W) for the combined mask

    Returns:
        Combined boolean mask of shape (H, W)
    """
    if not masks:
        return np.zeros(target_shape, dtype=bool)

    # Resize all masks to target shape and combine them
    combined_mask = np.zeros(target_shape, dtype=bool)

    for mask in masks:
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask.astype(bool)

        # Resize mask to target shape if needed
        if mask.shape != target_shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # Combine with OR operation
        combined_mask = combined_mask | mask

    return combined_mask


def load_video_masks(video_path, num_frames, target_shape):
    """
    Load masks from a video file where white pixels indicate foreground regions.

    Args:
        video_path: Path to the video file containing black/white masks
        num_frames: Number of frames to load
        target_shape: Target shape (H, W) for the masks

    Returns:
        List of boolean masks, one per frame
    """
    import cv2

    if not os.path.exists(video_path):
        raise ValueError(f"Video mask file not found: {video_path}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    masks = []
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Threshold to create binary mask (white pixels become True)
        # Assuming white pixels (high values) indicate foreground
        _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Convert to boolean and resize to target shape
        bool_mask = binary_mask > 127
        if bool_mask.shape != target_shape:
            bool_mask = cv2.resize(
                bool_mask.astype(np.uint8),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        masks.append(bool_mask)
        frame_count += 1

    cap.release()

    # If we didn't get enough frames, pad with empty masks
    while len(masks) < num_frames:
        masks.append(np.zeros(target_shape, dtype=bool))

    return masks[:num_frames]  # Return exactly num_frames masks


def create_flow_histogram(flow_values, title, output_path, bins=50):
    """
    Create and save a histogram of flow values.

    Args:
        flow_values: Array of flow values to plot
        title: Title for the histogram
        output_path: Path to save the histogram image
        bins: Number of bins for the histogram
    """
    plt.figure(figsize=(10, 6))
    plt.hist(flow_values.flatten(), bins=bins, alpha=0.7, edgecolor="black")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Flow Magnitude", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    mean_val = np.mean(flow_values)
    std_val = np.std(flow_values)
    median_val = np.median(flow_values)
    min_val = np.min(flow_values)
    max_val = np.max(flow_values)
    plt.text(
        0.02,
        0.98,
        f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to: {output_path}")


def create_comparison_histogram(
    background_flows, foreground_flows, title, output_path, bins=50
):
    """
    Create and save a comparison histogram of background vs foreground flow values.

    Args:
        background_flows: Array of background flow values
        foreground_flows: Array of foreground flow values
        title: Title for the histogram
        output_path: Path to save the histogram image
        bins: Number of bins for the histogram
    """
    plt.figure(figsize=(12, 8))

    # Plot both histograms
    plt.hist(
        background_flows.flatten(),
        bins=bins,
        alpha=0.6,
        edgecolor="black",
        label="Background Flows",
        color="blue",
        density=True,
    )

    plt.hist(
        foreground_flows.flatten(),
        bins=bins // 2,
        alpha=0.6,
        edgecolor="black",
        label="Foreground Flows",
        color="red",
        density=True,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Flow Magnitude", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    bg_mean = np.mean(background_flows)
    bg_std = np.std(background_flows)
    bg_min = np.min(background_flows)
    bg_max = np.max(background_flows)
    fg_mean = np.mean(foreground_flows)
    fg_std = np.std(foreground_flows)
    fg_min = np.min(foreground_flows)
    fg_max = np.max(foreground_flows)

    stats_text = f"Background:\nMean: {bg_mean:.4f}\nStd: {bg_std:.4f}\nMin: {bg_min:.4f}\nMax: {bg_max:.4f}\n\nForeground:\nMean: {fg_mean:.4f}\nStd: {fg_std:.4f}\nMin: {fg_min:.4f}\nMax: {fg_max:.4f}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison histogram to: {output_path}")


def merge_flows_with_dual_masks(
    flow1_path: str,
    flow2_path: str,
    background_mask_path: str,
    foreground_mask_video_path: str,
    output_folder: str = None,
    noise_channels: int = 16,
    visualize: bool = True,
    save_files: bool = True,
    device=None,
    progressive_noise_alpha=0,
    post_noise_alpha=0,
    warp_kwargs=dict(),
    flow_threshold: float = 5.0,
    save_histogram_files: bool = False,
):
    """
    Merge two flow files using dual masks for selective merging.

    Args:
        flow1_path (str): Path to the foreground flow file (flows_dxdy.npy)
        flow2_path (str): Path to the background flow file (flows_dxdy.npy)
        background_mask_path (str): Path to the background masks folder containing .npy files with naming pattern "frame_{4 digit frame number}_mask_0000_00.npy" - used to remove foreground from background flows
        foreground_mask_video_path (str): Path to the video file containing black/white masks - controls which parts of foreground flow to use
        output_folder (str, optional): Folder to save the output noise and visualization.
        noise_channels (int, optional): Number of channels in the generated noise. Defaults to 16.
        visualize (bool, optional): Whether to generate visualization images and video. Defaults to True.
        save_files (bool, optional): If True, will save files to disk. Defaults to True.
        device: Device to run on. If None, will auto-select.
        progressive_noise_alpha: For ryan, don't worry about it
        post_noise_alpha: For ryan, don't worry about it
        warp_kwargs (dict, optional): For experimental features. Don't worry about this if you're not Ryan Burgert.
        flow_threshold (float, optional): Threshold above which flow values will be set to 0. Defaults to 5.0. Set to 0 to disable thresholding.
        save_histogram_files (bool, optional): If True, save histogram images of flow magnitudes. Defaults to False.

    Returns:
        EasyDict: A dict containing the following keys:
            - 'numpy_noises' (np.ndarray): Generated noise with form [T, H, W, C].
            - 'numpy_flows' (np.ndarray): The merged flows with form [T-1, 2, H, W]
            - 'vis_frames' (np.ndarray): Visualization frames with form [T, H, W, C].
            - 'output_folder' (str): The path to the folder where outputs are saved (if save_files)
    """

    # Input assertions
    assert os.path.exists(flow1_path), f"Foreground flow file not found: {flow1_path}"
    assert os.path.exists(flow2_path), f"Background flow file not found: {flow2_path}"
    assert os.path.exists(
        background_mask_path
    ), f"Background mask folder not found: {background_mask_path}"
    assert os.path.exists(
        foreground_mask_video_path
    ), f"Foreground mask video file not found: {foreground_mask_video_path}"

    if device is None:
        if rp.currently_running_mac():
            device = "cpu"
        else:
            device = rp.select_torch_device(prefer_used=True)

    # Load the flow files
    rp.fansi_print(f"Loading foreground flow file: {flow1_path}", "yellow")
    flows_foreground = np.load(flow1_path)
    rp.fansi_print(f"Loading background flow file: {flow2_path}", "yellow")
    flows_background = np.load(flow2_path)

    # Check shapes and ensure compatibility
    assert (
        flows_foreground.shape == flows_background.shape
    ), f"Flow shapes must match: {flows_foreground.shape} vs {flows_background.shape}"
    assert (
        flows_foreground.ndim == 4
    ), f"Expected 4D flow array [T-1, 2, H, W], got shape {flows_foreground.shape}"
    assert (
        flows_foreground.shape[1] == 2
    ), f"Expected 2 channels for dx, dy, got {flows_foreground.shape[1]}"

    rp.fansi_print(
        f"Flow shape: {flows_foreground.shape}", "yellow"
    )  ## (48, 2, 240, 360)

    # Get dimensions
    T_minus_1, _, H, W = flows_foreground.shape

    # Load background masks from folder (multiple .npy files)
    rp.fansi_print(
        f"Loading background masks from folder: {background_mask_path}", "yellow"
    )
    background_masks = load_background_masks_from_folder(
        background_mask_path, T_minus_1 + 1, (H, W)
    )  # +1 because we have T-1 flows but T frames
    rp.fansi_print(f"Loaded {len(background_masks)} background masks", "yellow")  ## 49

    # Load foreground video masks
    rp.fansi_print(
        f"Loading foreground video masks from: {foreground_mask_video_path}", "yellow"
    )
    foreground_masks = load_video_masks(
        foreground_mask_video_path, T_minus_1 + 1, (H, W)
    )  # +1 because we have T-1 flows but T frames
    rp.fansi_print(f"Loaded {len(foreground_masks)} foreground video masks", "yellow")

    print("Background mask shape:", background_masks[0].shape)  ## (240, 360)
    print("Foreground masks shape:", foreground_masks[0].shape)  ## (240, 360)

    # Merge consecutive pairs of masks to reduce from 49 to 48 frames
    rp.fansi_print("Merging consecutive mask pairs to align with flow frames", "yellow")
    background_masks = merge_consecutive_mask_pairs(background_masks)
    foreground_masks = merge_consecutive_mask_pairs(foreground_masks)
    rp.fansi_print(
        f"After merging: {len(background_masks)} background masks, {len(foreground_masks)} foreground masks",
        "yellow",
    )

    # Merge flows using dual masks
    rp.fansi_print("Merging flows using dual masks", "yellow")
    merged_flows = flows_background.copy()  # Start with background flows

    for t in range(T_minus_1):
        # Get the masks for the current frame
        background_mask = background_masks[t]
        foreground_mask = foreground_masks[t]

        # Step 1: Remove foreground regions from background flows using background_mask
        # background_mask is True where we want to keep background flow (foreground regions are False)
        merged_flows[t, :, background_mask] = (
            0  # Set background flow to 0 in foreground regions
        )

        background_keep_mask = ~background_mask

        # Create histogram of background flow values
        if save_histogram_files:
            # Get the flow magnitude values for background regions (Euclidean norm of dx, dy)
            bg_flow_values = np.sqrt(
                merged_flows[t, 0, background_keep_mask] ** 2
                + merged_flows[t, 1, background_keep_mask] ** 2
            )

            # Create histogram folder if it doesn't exist
            histogram_folder = os.path.join(output_folder, "flow_histograms")
            os.makedirs(histogram_folder, exist_ok=True)

            # Create and save histogram
            histogram_path = os.path.join(
                histogram_folder, f"frame_{t:04d}_background_flows_histogram.png"
            )
            create_flow_histogram(
                bg_flow_values,
                f"Background Flow Magnitudes - Frame {t}",
                histogram_path,
            )

        # Apply threshold filtering to remove extreme flow values
        # This helps remove outliers and extreme flow magnitudes that could cause artifacts
        if flow_threshold > 0:
            # Get the absolute magnitude of the merged flows for this frame
            flow_magnitudes = np.sqrt(
                merged_flows[t, 0, :] ** 2 + merged_flows[t, 1, :] ** 2
            )
            # Create mask for values above threshold
            above_threshold_mask = flow_magnitudes > flow_threshold
            # Set flows above threshold to 0
            merged_flows[t, :, above_threshold_mask] = 0
            print(
                f"Frame {t}: Set {np.sum(above_threshold_mask)} pixels to 0 (above threshold {flow_threshold})"
            )

        merged_flows[t, :, foreground_mask] = flows_foreground[t, :, foreground_mask]

        # Create histogram of foreground flow values
        if save_histogram_files:
            # Get the flow magnitude values for foreground regions (Euclidean norm of dx, dy)
            fg_flow_values = np.sqrt(
                flows_foreground[t, 0, foreground_mask] ** 2
                + flows_foreground[t, 1, foreground_mask] ** 2
            )

            # Create and save histogram for foreground flows
            fg_histogram_path = os.path.join(
                histogram_folder, f"frame_{t:04d}_foreground_flows_histogram.png"
            )
            create_flow_histogram(
                fg_flow_values,
                f"Foreground Flow Magnitudes - Frame {t}",
                fg_histogram_path,
            )

            # Create comparison histogram for this frame
            comparison_histogram_path = os.path.join(
                histogram_folder, f"frame_{t:04d}_comparison_histogram.png"
            )
            create_comparison_histogram(
                bg_flow_values,
                fg_flow_values,
                f"Background vs Foreground Flow Comparison - Frame {t}",
                comparison_histogram_path,
            )

    # Decide the location of and create the output folder
    if save_files:
        if output_folder is None:
            output_folder = "dual_masked_merged_noise_outputs/dual_masked_merged_flows"
        os.makedirs(output_folder, exist_ok=True)
        rp.fansi_print("Output folder: " + output_folder, "green")

    def downscale_noise(noise):
        downscale_factor = 4  ## from the original size!
        down_noise = rp.torch_resize_image(
            noise, 1 / downscale_factor, interp="area"
        )  # Avg pooling
        down_noise = down_noise * downscale_factor  # Adjust for STD
        return down_noise

    with torch.no_grad():

        # Initialize the NoiseWarper
        warper = NoiseWarper(
            c=noise_channels,
            h=H,
            w=W,
            device=device,
            post_noise_alpha=post_noise_alpha,
            progressive_noise_alpha=progressive_noise_alpha,
            warp_kwargs=warp_kwargs,
        )

        # Get initial noise
        noise = warper.noise

        down_noise = downscale_noise(noise)
        numpy_noise = rp.as_numpy_image(down_noise).astype(np.float16)

        numpy_noises = [numpy_noise]
        vis_frames = []

        for index in tqdm(range(T_minus_1), desc="Generating dual masked merged noise"):
            # Get the merged flow for this timestep
            flow_t = merged_flows[index]  # Shape: [2, H, W]
            dx = torch.from_numpy(flow_t[0]).to(device)
            dy = torch.from_numpy(flow_t[1]).to(device)

            # Warp the noise using the merged flow
            noise = warper(dx, dy).noise

            down_noise = downscale_noise(noise)
            numpy_noise = rp.as_numpy_image(down_noise).astype(np.float16)

            numpy_noises.append(numpy_noise)

            if visualize:
                # Create visualization
                flow_rgb = rp.optical_flow_to_image(dx, dy)

                # Turn the noise into a numpy HWC RGB array for visualization
                noise_image = np.zeros((*numpy_noise.shape[:2], 3))
                noise_image_c = min(noise_channels, 3)
                noise_image[:, :, :noise_image_c] = numpy_noise[:, :, :noise_image_c]

                # Normalize noise for visualization
                noise_vis = noise_image / 3 + 0.5

                # Get the masks for this frame
                background_mask = background_masks[index]
                foreground_mask = foreground_masks[index]
                background_mask_rgb = np.stack(
                    [background_mask, background_mask, background_mask], axis=2
                ).astype(np.float32)
                foreground_mask_rgb = np.stack(
                    [foreground_mask, foreground_mask, foreground_mask], axis=2
                ).astype(np.float32)

                down_size = rp.get_image_dimensions(noise_vis)
                down_flow_rgb, down_background_mask_rgb, down_foreground_mask_rgb = (
                    rp.resize_images(
                        flow_rgb,
                        background_mask_rgb,
                        foreground_mask_rgb,
                        size=down_size,
                    )
                )

                visualization = rp.as_byte_image(
                    rp.tiled_images(
                        rp.labeled_images(
                            [
                                noise_vis,
                                down_flow_rgb,
                                down_background_mask_rgb,  # Show background mask
                                down_foreground_mask_rgb,  # Show foreground mask
                            ],
                            [
                                "Dual Masked Merged Noise",
                                "Dual Masked Merged Flow",
                                "Background Mask",
                                "Foreground Mask",
                            ],
                            font="G:Zilla Slab",
                        )
                    )
                )

                vis_frames.append(visualization)

    # Create histogram of final merged flows
    if save_histogram_files:
        print("Creating histograms of merged flows...")

        for t in range(T_minus_1):
            # Get the merged flow magnitude values for this frame (Euclidean norm of dx, dy)
            merged_flow_values = np.sqrt(
                merged_flows[t, 0, :] ** 2 + merged_flows[t, 1, :] ** 2
            )

            # Create and save histogram for merged flows
            merged_histogram_path = os.path.join(
                histogram_folder, f"frame_{t:04d}_merged_flows_histogram.png"
            )
            create_flow_histogram(
                merged_flow_values,
                f"Merged Flow Magnitudes - Frame {t}",
                merged_histogram_path,
            )

        # Create summary histogram across all frames
        print("Creating summary histogram across all frames...")
        # Calculate flow magnitudes for all frames
        all_merged_flows = []
        for t in range(T_minus_1):
            flow_magnitudes = np.sqrt(
                merged_flows[t, 0, :] ** 2 + merged_flows[t, 1, :] ** 2
            )
            all_merged_flows.append(flow_magnitudes)
        all_merged_flows = np.concatenate(all_merged_flows)

        summary_histogram_path = os.path.join(
            histogram_folder, "summary_all_frames_merged_flows_histogram.png"
        )
        create_flow_histogram(
            all_merged_flows,
            "Merged Flow Magnitudes - All Frames Combined",
            summary_histogram_path,
            bins=100,  # More bins for the combined data
        )

        # Create summary comparison histogram across all frames
        print("Creating summary comparison histogram across all frames...")
        all_bg_flows = []
        all_fg_flows = []

        for t in range(T_minus_1):
            bg_mask = background_masks[t]
            fg_mask = foreground_masks[t]
            # Calculate flow magnitudes for background and foreground
            bg_flow_magnitudes = np.sqrt(
                flows_background[t, 0, ~bg_mask] ** 2
                + flows_background[t, 1, ~bg_mask] ** 2
            )
            fg_flow_magnitudes = np.sqrt(
                flows_foreground[t, 0, fg_mask] ** 2
                + flows_foreground[t, 1, fg_mask] ** 2
            )
            all_bg_flows.append(bg_flow_magnitudes)
            all_fg_flows.append(fg_flow_magnitudes)

        all_bg_flows = np.concatenate(all_bg_flows)
        all_fg_flows = np.concatenate(all_fg_flows)

        summary_comparison_path = os.path.join(
            histogram_folder, "summary_all_frames_comparison_histogram.png"
        )
        create_comparison_histogram(
            all_bg_flows,
            all_fg_flows,
            "Background vs Foreground Flow Comparison - All Frames Combined",
            summary_comparison_path,
            bins=100,
        )

    print("Finished processing merged flows with dual masks")
    numpy_noises = np.stack(numpy_noises).astype(np.float16)
    if vis_frames:
        vis_frames = np.stack(vis_frames)

    # print("numpy noises shape:", numpy_noises.shape) ## (48, 240, 360, 16)

    if save_files and len(vis_frames):
        vis_img_folder = rp.make_directory(output_folder + "/visualization_images")
        vis_img_paths = rp.path_join(vis_img_folder, "visual_%05i.png")
        rp.save_images(vis_frames, vis_img_paths, show_progress=True)

        if "ffmpeg" in rp.get_system_commands():
            vis_mp4_path = rp.path_join(
                output_folder, "dual_masked_merged_noise_visualization.mp4"
            )
            noise_mp4_path = rp.path_join(
                output_folder, "dual_masked_merged_noise_video.mp4"
            )
            rp.save_video_mp4(
                vis_frames,
                vis_mp4_path,
                video_bitrate="max",
                framerate=30,
            )
            rp.save_video_mp4(
                (numpy_noises / 4 + 0.5)[:, :, :, :3],
                noise_mp4_path,
                video_bitrate="max",
                framerate=30,
            )
        else:
            rp.fansi_print(
                "Please install ffmpeg! We won't save an MP4 this time - please try again."
            )

    if save_files:
        noises_path = rp.path_join(output_folder, "dual_masked_merged_noises.npy")
        flows_path = rp.path_join(output_folder, "dual_masked_merged_flows_dxdy.npy")
        np.save(noises_path, numpy_noises)
        rp.fansi_print(
            "Saved " + noises_path + " with shape " + str(numpy_noises.shape), "green"
        )
        np.save(flows_path, merged_flows)
        rp.fansi_print(
            "Saved " + flows_path + " with shape " + str(merged_flows.shape), "green"
        )

        rp.fansi_print(
            rp.get_file_name(__file__)
            + ": Done merging flows with dual masks, results are at "
            + rp.get_absolute_path(output_folder),
            "green",
            "bold",
        )


def merge_consecutive_mask_pairs(masks):
    """
    Merge consecutive pairs of masks by taking their union.
    Reduces the number of masks from N to N-1.

    Args:
        masks: List of boolean masks, each of shape (H, W)

    Returns:
        List of merged boolean masks, each of shape (H, W)
    """
    if len(masks) < 2:
        return masks

    merged_masks = []
    for i in range(len(masks) - 1):
        # Take union of consecutive pair (OR operation)
        merged_mask = masks[i] | masks[i + 1]
        merged_masks.append(merged_mask)

    return merged_masks


def load_static_mask(mask_path, target_shape):
    """
    Load a static mask from a file (supports .npy, .png, .jpg, etc.)

    Args:
        mask_path: Path to the mask file
        target_shape: Target shape (H, W) for the mask

    Returns:
        Boolean mask of shape (H, W)
    """
    if not os.path.exists(mask_path):
        raise ValueError(f"Mask file not found: {mask_path}")

    # Determine file type and load accordingly
    file_ext = os.path.splitext(mask_path)[1].lower()

    if file_ext == ".npy":
        # Load numpy array
        mask = np.load(mask_path)
    elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
        # Load image file
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load image file: {mask_path}")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    # Ensure mask is 2D
    if mask.ndim > 2:
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"Mask should be 2D, got shape {mask.shape}")

    # Convert to boolean (assuming non-zero values are True)
    bool_mask = mask.astype(bool)

    # Resize to target shape if needed
    if bool_mask.shape != target_shape:
        bool_mask = cv2.resize(
            bool_mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return bool_mask


def load_background_masks_from_folder(masks_dir, num_frames, target_shape):
    """
    Load background masks from a folder containing .npy files with naming pattern:
    frame_{4 digit frame number}_mask_0000_00.npy

    Args:
        masks_dir: Directory containing mask .npy files
        num_frames: Number of frames to load masks for
        target_shape: Target shape (H, W) for the masks

    Returns:
        List of boolean masks, one per frame
    """
    if not os.path.exists(masks_dir):
        raise ValueError(f"Masks directory not found: {masks_dir}")

    masks = []

    for frame_idx in range(num_frames):
        # Look for mask file with the specific naming pattern
        mask_pattern = f"frame_{frame_idx:04d}_mask_0000_00.npy"
        mask_file = os.path.join(masks_dir, mask_pattern)

        if os.path.exists(mask_file):
            # Load the mask
            mask = np.load(mask_file)

            # Ensure mask is 2D
            if mask.ndim > 2:
                if mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                else:
                    raise ValueError(f"Mask should be 2D, got shape {mask.shape}")

            # Convert to boolean (assuming non-zero values are True)
            bool_mask = mask.astype(bool)

            # Resize to target shape if needed
            if bool_mask.shape != target_shape:
                bool_mask = cv2.resize(
                    bool_mask.astype(np.uint8),
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        else:
            # If mask file doesn't exist, create an empty mask
            print(f"Warning: Mask file not found: {mask_file}, using empty mask")
            bool_mask = np.zeros(target_shape, dtype=bool)

        masks.append(bool_mask)

    return masks


def merge_rgb_videos_with_masks(
    foreground_video_path: str,
    background_template_video_path: str,
    mask_video_path: str,
    output_folder: str,
    template_video_video_mask_path: str = None,
    target_num_frames: int = 49,
    target_height: int = 480,
    target_width: int = 720,
):
    """
    Merge two RGB videos (foreground and background) according to per-frame masks and
    save the composited result as "video_movingbg.mp4".

    Foreground is taken from `warped_from_first.mp4` (or given path), and background is
    taken from a template video. Both videos are loaded and preprocessed the same way
    as in make_warped_noise_for_background_hh.py (resize to 49 frames, fit to 480x720,
    then center crop to 480x720).

    Args:
        foreground_video_path: Path to the RGB foreground video (e.g., warped_from_first.mp4)
        background_template_video_path: Path to the RGB background template video
        mask_video_path: Path to the black/white mask video (white = foreground)
        output_folder: Directory to save the merged result
        template_video_video_mask_path: Optional path to per-frame .npy masks for the template video; used to remove the template's original foreground from the background before compositing
        target_num_frames: Number of frames to align to (default: 49)
        target_height: Target video height after preprocessing (default: 480)
        target_width: Target video width after preprocessing (default: 720)
    """
    if not os.path.exists(foreground_video_path):
        raise FileNotFoundError(f"Foreground video not found: {foreground_video_path}")
    if not os.path.exists(background_template_video_path):
        raise FileNotFoundError(
            f"Background template video not found: {background_template_video_path}"
        )
    if not os.path.exists(mask_video_path):
        raise FileNotFoundError(f"Mask video not found: {mask_video_path}")

    os.makedirs(output_folder, exist_ok=True)

    # Load and preprocess foreground and background videos (match the background noise script)
    fg_video = rp.load_video(foreground_video_path)
    bg_video = rp.load_video(background_template_video_path)

    fg_video = rp.resize_list(fg_video, length=target_num_frames)
    bg_video = rp.resize_list(bg_video, length=target_num_frames)

    fg_video = rp.resize_images_to_hold(
        fg_video, height=target_height, width=target_width
    )
    bg_video = rp.resize_images_to_hold(
        bg_video, height=target_height, width=target_width
    )

    fg_video = rp.crop_images(
        fg_video, height=target_height, width=target_width, origin="center"
    )
    bg_video = rp.crop_images(
        bg_video, height=target_height, width=target_width, origin="center"
    )

    fg_video = rp.as_numpy_array(fg_video)  # THWC, uint8
    bg_video = rp.as_numpy_array(bg_video)  # THWC, uint8

    T = min(len(fg_video), len(bg_video), target_num_frames)
    H, W = target_height, target_width

    # Load masks aligned to target shape and frame count
    masks = load_video_masks(mask_video_path, T, (H, W))
    if len(masks) != T:
        masks = masks[:T]

    # Optionally remove the template's original foreground from background frames using per-frame masks
    if template_video_video_mask_path is not None:
        try:
            template_masks = load_background_masks_from_folder(
                template_video_video_mask_path, T, (H, W)
            )
            for t in range(T):
                mask_bool = template_masks[t]
                # Zero-out background where the template mask marks foreground
                bg_video[t][mask_bool] = 0
        except Exception as e:
            print(
                f"Warning: Failed to apply template video masks from {template_video_video_mask_path}: {e}"
            )

    # Composite frames
    merged_frames = []
    for t in range(T):
        mask_bool = masks[t]
        # Ensure mask is float [0,1] with shape HxW -> HxWx1
        mask_f = mask_bool.astype(np.float32)[..., None]
        fg = fg_video[t].astype(np.float32) / 255.0
        bg = bg_video[t].astype(np.float32) / 255.0
        comp = mask_f * fg + (1.0 - mask_f) * bg
        merged_frames.append(rp.as_byte_image(comp))

    merged_frames = rp.as_numpy_array(merged_frames)

    # Save the merged video
    output_video_path = os.path.join(output_folder, "video_movingbg.mp4")
    rp.save_video_mp4(
        merged_frames, output_video_path, framerate=30, video_bitrate="max"
    )
    print(f"Saved merged RGB video to: {output_video_path}")

    return output_video_path


def process_flow_pairs_with_segmentation(
    input_dir: str,
    flow_threshold: float = 2.0,
    selected_vids_file: str = None,
    input_folder_templatevideo: str = None,
    save_histogram_files: bool = False,
):
    """
    Process pairs of flow files from two input folders and merge them using segmentation masks.

    Args:
        input_folder1: Path to the first folder containing flow files (foreground)
        input_folder2: Path to the second folder containing flow files (background)
        segmentation_dir: Path to the segmentation outputs directory
        output_base_folder: Base folder where outputs will be saved
        foreground_mask_video_path: Path to the video file containing black/white masks - controls which parts of foreground flow to use
        total_augmentations: Total number of augmentations to process (default: 4)
        flow_threshold: Threshold above which flow values will be set to 0 (default: 5.0)
        selected_vids_file: Path to file containing selected video folder names (one per line). If None, process all folders.
        save_histogram_files: If True, save histogram images of flow magnitudes (default: False)
    """

    # Load selected video folders if specified
    selected_folders = None
    if selected_vids_file and selected_vids_file != "None":
        if not os.path.exists(selected_vids_file):
            raise FileNotFoundError(
                f"Selected videos file not found: {selected_vids_file}"
            )

        with open(selected_vids_file, "r") as f:
            selected_folders = [
                line.strip() for line in f if line.strip()
            ]  # Remove empty lines

        if not selected_folders:
            raise ValueError("No video names found in the selected videos file")

        print(f"Found {len(selected_folders)} selected video folders to process")

    # for subfolder in common_subfolders:
    for subfolder in selected_folders:
        print(f"\nProcessing subfolder: {subfolder}")

        input_folder1 = os.path.join(input_dir, subfolder, "noise_output")
        input_folder2 = os.path.join(input_dir, subfolder, "output_noises_background")
        segmentation_dir = os.path.join(
            input_dir, subfolder, "video_segmentation_outputs"
        )
        output_base_folder = os.path.join(input_dir, subfolder, "output_merged_noises")

        flow1_path = os.path.join(input_folder1, "flows_dxdy.npy")
        flow2_path = os.path.join(input_folder2, "flows_dxdy.npy")
        masks_dir = os.path.join(segmentation_dir, "masks_npy")

        foreground_mask_video_path = os.path.join(
            input_dir, subfolder, "obj_masks_updated.mp4"
        )

        # Check if all required files exist
        if not os.path.exists(flow1_path):
            print(f"Warning: Foreground flow file not found at {flow1_path}")
            continue
        if not os.path.exists(flow2_path):
            print(f"Warning: Background flow file not found at {flow2_path}")
            continue
        if not os.path.exists(masks_dir):
            print(f"Warning: Masks directory not found at {masks_dir}")
            continue
        if not os.path.exists(foreground_mask_video_path):
            print(
                f"Warning: Foreground mask video file not found at {foreground_mask_video_path}"
            )
            continue

        output_folder = os.path.join(output_base_folder)

        print(f"Foreground flow: {flow1_path}")
        print(f"Background flow: {flow2_path}")
        print(f"Masks directory: {masks_dir}")
        print(f"Foreground mask video path: {foreground_mask_video_path}")
        print(f"Output folder: {output_folder}")

        # After merging flows, also merge RGB videos using masks
        foreground_rgb_path = os.path.join(
            os.path.dirname(input_folder1), "obj_only_video.mp4"
        )

        background_rgb_path = os.path.join(input_folder2, "input_template.mp4")

        print("foreground_rgb_path for video: ", foreground_rgb_path)
        print("background_rgb_path for video: ", background_rgb_path)

        template_video_video_mask_path = os.path.join(
            os.path.dirname(input_folder1), "video_segmentation_outputs", "masks_npy"
        )
        print("template_video_video_mask_path: ", template_video_video_mask_path)
        if os.path.exists(foreground_rgb_path) and os.path.exists(background_rgb_path):
            try:
                merge_rgb_videos_with_masks(
                    foreground_video_path=foreground_rgb_path,
                    background_template_video_path=background_rgb_path,
                    mask_video_path=foreground_mask_video_path,
                    output_folder=output_folder,
                    template_video_video_mask_path=template_video_video_mask_path,
                )
            except Exception as me:
                print(f"Warning: Failed to merge RGB videos for {subfolder}: {me}")
        else:
            if not os.path.exists(foreground_rgb_path):
                print(
                    f"Warning: Foreground RGB video not found at {foreground_rgb_path}"
                )
            if not os.path.exists(background_rgb_path):
                print(
                    f"Warning: Background template RGB video not found at {background_rgb_path}"
                )

        try:

            result = merge_flows_with_dual_masks(
                flow1_path=flow1_path,
                flow2_path=flow2_path,
                background_mask_path=masks_dir,
                foreground_mask_video_path=foreground_mask_video_path,
                output_folder=output_folder,
                visualize=True,
                save_files=True,
                flow_threshold=flow_threshold,
                save_histogram_files=save_histogram_files,
            )

        except Exception as e:
            # print(f"Error processing {subfolder} augmentation {aug_id}: {str(e)}")
            print(f"Error processing {subfolder}: {str(e)}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Merge two flow files using segmentation masks, video masks, or dual masks for selective merging."
    )
    parser.add_argument(
        "--flow1_path",
        type=str,
        help="Path to the foreground flow file (flows_dxdy.npy)",
    )
    parser.add_argument(
        "--flow2_path",
        type=str,
        help="Path to the background flow file (flows_dxdy.npy)",
    )
    parser.add_argument(
        "--masks_dir", type=str, help="Directory containing segmentation masks"
    )
    parser.add_argument(
        "--mask_video_path",
        type=str,
        help="Path to video file containing black/white masks",
    )
    parser.add_argument(
        "--background_mask_path",
        type=str,
        help='Path to background masks folder containing .npy files with naming pattern "frame_{4 digit frame number}_mask_0000_00.npy" - used to remove foreground from background flows',
    )
    parser.add_argument(
        "--foreground_mask_video_path",
        type=str,
        help="Path to video file containing black/white masks - controls which parts of foreground flow to use",
    )
    parser.add_argument(
        "--output_folder", type=str, help="Output folder for merged results"
    )
    parser.add_argument(
        "--noise_channels", type=int, default=16, help="Number of noise channels"
    )

    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=2.0,
        help="Threshold above which flow values will be set to 0 (default: 2.0). To be more resistant to noise.",
    )
    parser.add_argument(
        "--selected_vids_file", type=str, default="None", help="Selected vids file"
    )

    parser.add_argument("--input_dir", type=str, default="None", help="Input directory")

    parser.add_argument(
        "--input_folder_templatevideo",
        type=str,
        default="None",
        help="Input folder template video",
    )

    parser.add_argument(
        "--save_histogram_files",
        action="store_true",
        default=False,
        help="Save histogram images of flow magnitudes (default: False)",
    )

    args = parser.parse_args()

    process_flow_pairs_with_segmentation(
        input_dir=args.input_dir,
        flow_threshold=args.flow_threshold,
        selected_vids_file=args.selected_vids_file,
        input_folder_templatevideo=args.input_folder_templatevideo,
        save_histogram_files=args.save_histogram_files,
    )


if __name__ == "__main__":
    main()
