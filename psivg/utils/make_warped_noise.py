# Code is adapted from the original Go-with-the-flow code by Ryan Burgert 2024

import argparse
import glob
import os

import cv2
import numpy as np
import rp

# from rp import *

rp.r._pip_import_autoyes = True  # Automatically install missing packages

rp.pip_import("fire")
rp.git_import(
    "CommonSource"
)  # If missing, installs code from https://github.com/RyannDaGreat/CommonSource
import rp.git.CommonSource.noise_warp as nw


def process_video(video_path: str, output_folder: str, flow, frame):
    """
    Process a single video file and generate warped noise.
    Args:
        video_path: Path to the video file
        output_folder: Path to save the output
    """
    # if rp.folder_exists(output_folder):
    #     raise RuntimeError(f"The given output_folder={repr(output_folder)} already exists! To avoid clobbering what might be in there, please specify a folder that doesn't exist so I can create one for you. Alternatively, you could delete that folder if you don't care whats in it.")

    FRAME = (
        2**-1
    )  # We immediately resize the input frames by this factor, before calculating optical flow
    # The flow is calulated at (input size) × FRAME resolution.
    # Higher FLOW values result in slower optical flow calculation and higher intermediate noise resolution
    # Larger is not always better - watch the preview in Jupyter to see if it looks good!

    FLOW = (
        2**3
    )  # Then, we use bilinear interpolation to upscale the flow by this factor
    # We warp the noise at (input size) × FRAME × FLOW resolution
    # The noise is then downsampled back to (input size)
    # Higher FLOW values result in more temporally consistent noise warping at the cost of higher VRAM usage and slower inference time
    LATENT = 8  # We further downsample the outputs by this amount - because 8 pixels wide corresponds to one latent wide in Stable Diffusion
    # The final output size is (input size) ÷ LATENT regardless of FRAME and FLOW

    FLOW = flow
    FRAME = frame

    # LATENT = 1    #Uncomment this line for a prettier visualization! But for latent diffusion models, use LATENT=8

    # You can also use video files or URLs
    # video = "https://www.shutterstock.com/shutterstock/videos/1100085499/preview/stock-footage-bremen-germany-october-old-style-carousel-moving-on-square-in-city-horses-on-traditional.webm"

    video = rp.load_video(video_path)

    # Preprocess the video
    video = rp.resize_list(
        video, length=49
    )  # Stretch or squash video to 49 frames (CogVideoX's length)
    video = rp.resize_images_to_hold(video, height=480, width=720)
    video = rp.crop_images(
        video, height=480, width=720, origin="center"
    )  # Make the resolution 480x720 (CogVideoX's resolution)
    video = rp.as_numpy_array(video)

    # See this function's docstring for more information!
    noise_output_folder = os.path.join(output_folder, "noise_output")
    output = nw.get_noise_from_video(
        video,
        remove_background=False,  # Set this to True to matte the foreground - and force the background to have no flow
        visualize=True,  # Generates nice visualization videos and previews in Jupyter notebook
        save_files=True,  # Set this to False if you just want the noises without saving to a numpy file
        noise_channels=16,
        # output_folder=output_folder,
        output_folder=noise_output_folder,
        resize_frames=FRAME,
        resize_flow=FLOW,
        downscale_factor=round(FRAME * FLOW) * LATENT,
    )

    output.first_frame_path = rp.save_image(
        video[0], rp.path_join(output_folder, "first_frame.png")
    )

    rp.save_video_mp4(
        video,
        rp.path_join(output_folder, "input.mp4"),
        framerate=12,
        video_bitrate="max",
    )
    # rp.save_video_mp4(video, os.path.join(output_folder, 'input.mp4'), framerate=12, video_bitrate='max')

    print("Noise shape:", output.numpy_noises.shape)
    print("Flow shape:", output.numpy_flows.shape)
    print("Output folder:", output.output_folder)


def main(
    input_folder: str,
    output_base_folder: str,
    selected_vids_file: str,
    flow,
    frame,
    first_frame_folder: str,
    mask_firstframe_folder: str,
    newload,
    multiobject,
):
    """
    Process videos based on paths specified in the selected_vids_file.
    Each line in the file should contain a two-level path (e.g., "0017_static/batch_run_14_SEP").

    Args:
        input_folder: Path to the base folder containing video subfolders
        output_base_folder: Base folder where outputs will be saved
        selected_vids_file: Path to file containing selected video paths
    """
    # Create output base folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)

    # Read the selected video paths from the file
    if not os.path.exists(selected_vids_file):
        raise FileNotFoundError(f"Selected vids file not found: {selected_vids_file}")

    with open(selected_vids_file, "r") as f:
        selected_paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(selected_paths)} video paths to process")

    # Process each selected video path
    for video_path_line in selected_paths:
        # Construct the full path to the video folder
        video_folder_path = os.path.join(input_folder, video_path_line)

        if not os.path.exists(video_folder_path):
            print(
                f"Warning: Video folder not found at {video_folder_path}, skipping..."
            )
            continue

        # Look for obj_only folder containing PNG frames
        obj_only_folder = os.path.join(video_folder_path, "obj_only")

        if not os.path.exists(obj_only_folder):
            print(
                f"Warning: obj_only folder not found at {obj_only_folder}, skipping..."
            )
            continue

        # Check if there are PNG frames in the obj_only folder
        frame_files = glob.glob(os.path.join(obj_only_folder, "*.png"))
        if not frame_files:
            print(f"Warning: No PNG frames found in {obj_only_folder}, skipping...")
            continue
        frame_files.sort()

        #### getting the first 49 frames for now due to gwtf... Can change this logic if need something more complicated
        frame_files = frame_files[:49]

        ### loading the first frame
        # Extract first path segment (before '/') and then take substring before '_'
        if newload:
            digits = "".join(ch for ch in video_path_line if ch.isdigit())
            video_name = "0" + digits[:3]
        elif not newload:
            _normalized_path = video_path_line.replace("\\", "/")
            _first_segment = _normalized_path.split("/", 1)[0]
            _segment_prefix = _first_segment.split("_", 1)[0]
            video_name = _segment_prefix
        first_frame_path = os.path.join(first_frame_folder, f"{video_name}.png")
        first_frame = rp.load_image(first_frame_path)

        print(f"First frame path: {first_frame_path}")

        # Looading the mask folder. this one uses the simulator's provided mask.
        obj_mask_folder = os.path.join(video_folder_path, "obj_mask")

        if multiobject:
            ### loading from the first frame mask, processed by segment_frames.py
            mask_img1 = np.load(
                os.path.join(
                    mask_firstframe_folder, f"{video_name}/mask_obj01_0000_00.npy"
                )
            )
            mask_img2 = np.load(
                os.path.join(
                    mask_firstframe_folder, f"{video_name}/mask_obj02_0000_00.npy"
                )
            )

            total_mask = (mask_img1 > 0.5) | (mask_img2 > 0.5)
            background = rp.cv_inpaint_image(first_frame, mask=total_mask)

        elif not multiobject:
            ### loading from the first frame mask, processed by segment_frames.py
            mask_img = np.load(
                os.path.join(mask_firstframe_folder, f"{video_name}/mask_0000_00.npy")
            )

            total_mask = mask_img > 0.5
            background = rp.cv_inpaint_image(first_frame, mask=total_mask)

        print(f"Background shape: {background.shape}")

        # Create output folder maintaining the same structure
        output_folder = os.path.join(output_base_folder, video_path_line)

        os.makedirs(output_folder, exist_ok=True)
        background_path = os.path.join(output_folder, "background.png")
        rp.save_image(background, background_path)

        # Overlay foreground over background for each frame using masks from obj_mask
        overlaid_folder = os.path.join(output_folder, "overlaid_frames")
        os.makedirs(overlaid_folder, exist_ok=True)

        # Compile obj_mask/*.png into obj_mask.mp4 in the output folder. (for the final dataset)
        mask_frame_files = glob.glob(os.path.join(obj_mask_folder, "*.png"))
        if mask_frame_files:
            mask_frame_files.sort()
            obj_mask_video_path = os.path.join(output_folder, "obj_mask.mp4")
            try:
                rp.save_video_mp4(
                    mask_frame_files,
                    obj_mask_video_path,
                    framerate=12,
                    video_bitrate="max",
                )
                print(f"Saved mask video: {obj_mask_video_path}")
            except Exception as e:
                print(
                    f"Warning: failed to save mask video at {obj_mask_video_path}: {str(e)}"
                )

        # Ensure background matches frame size (based on first frame)
        _first_frame_img = rp.load_image(frame_files[0])
        if background.shape != _first_frame_img.shape:
            background_resized = cv2.resize(
                background,
                (_first_frame_img.shape[1], _first_frame_img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            background_resized = background

        for frame_path in frame_files:
            frame_img = rp.load_image(frame_path)

            # Resize background if current frame size differs
            if background_resized.shape != frame_img.shape:
                bg_img = cv2.resize(
                    background,
                    (frame_img.shape[1], frame_img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                bg_img = background_resized

            mask_name = os.path.basename(frame_path)
            mask_path = os.path.join(obj_mask_folder, mask_name)
            if not os.path.exists(mask_path):
                print(
                    f"Warning: mask not found for frame {mask_name}; skipping overlay for this frame"
                )
                continue

            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                print(f"Warning: could not read mask {mask_path}; skipping")
                continue

            if (mask_gray.shape[0] != frame_img.shape[0]) or (
                mask_gray.shape[1] != frame_img.shape[1]
            ):
                mask_gray = cv2.resize(
                    mask_gray,
                    (frame_img.shape[1], frame_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask_bool = mask_gray > 127  # white = foreground
            result = np.where(mask_bool[..., None], frame_img, bg_img)

            out_name = os.path.basename(frame_path)
            out_path = os.path.join(overlaid_folder, out_name)
            rp.save_image(result, out_path)

        # Check if there are PNG frames in the obj_only folder
        frame_files_overlaid = glob.glob(os.path.join(overlaid_folder, "*.png"))
        if not frame_files_overlaid:
            print(f"Warning: No PNG frames found in {overlaid_folder}, skipping...")
            continue
        frame_files_overlaid.sort()

        # Create the output video path
        video_name = "obj_only_video"
        output_video_path = os.path.join(output_folder, f"{video_name}.mp4")

        print(f"\nProcessing video: {video_name}")
        print(f"Frames folder: {obj_only_folder}")
        print(f"Output video: {output_video_path}")
        print(f"Output folder: {output_folder}")

        try:
            # First, combine the PNG frames into an MP4 video
            print(f"Combining {len(frame_files_overlaid)} frames into MP4 video...")
            output_video_file = rp.save_video_mp4(
                frame_files_overlaid, output_video_path, video_bitrate="max"
            )
            print("Combined frames into MP4 video")

            # Then process the created MP4 video
            process_video(output_video_path, output_folder, flow, frame)
            print(f"Successfully processed {video_name}")
        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Make warped noise for videos in a folder."
    )
    parser.add_argument(
        "--selected_vids_file",
        type=str,
        default="testing_output",
        help="Selected vids file",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="testing_output",
        help="Input folder containing video subfolders",
    )
    parser.add_argument(
        "--output_folder", type=str, default="testing_output", help="Output Directory"
    )
    parser.add_argument(
        "--flow", type=int, default=2**3, help="FLOW parameter"
    )  ## FLOW = 2**3
    parser.add_argument(
        "--frame", type=float, default=2**-1, help="FRAME parameter"
    )  ## FRAME = 2**-1

    parser.add_argument(
        "--first_frame_folder",
        type=str,
        default="testing_output",
        help="First frame folder",
    )
    parser.add_argument(
        "--mask_firstframe_folder",
        type=str,
        default="testing_output",
        help="Mask first frame folder",
    )

    parser.add_argument(
        "--newload",
        action="store_true",
        default=False,
        help="Whether to load the paths the new way",
    )
    parser.add_argument(
        "--multiobject",
        action="store_true",
        default=False,
        help="Whether to segment multiple objects",
    )

    args = parser.parse_args()

    input_folder = args.input_folder
    output_base_folder = args.output_folder

    main(
        input_folder=input_folder,
        output_base_folder=output_base_folder,
        selected_vids_file=args.selected_vids_file,
        flow=args.flow,
        frame=args.frame,
        first_frame_folder=args.first_frame_folder,
        mask_firstframe_folder=args.mask_firstframe_folder,
        newload=args.newload,
        multiobject=args.multiobject,
    )
