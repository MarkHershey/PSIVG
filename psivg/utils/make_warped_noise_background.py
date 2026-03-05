# Code is adapted from the original Go-with-the-flow code by Ryan Burgert 2024

import argparse
import glob
import os

import rp

rp.r._pip_import_autoyes = True  # Automatically install missing packages

rp.pip_import("fire")
rp.git_import(
    "CommonSource"
)  # If missing, installs code from https://github.com/RyannDaGreat/CommonSource
import sys

import numpy as np
import rp.git.CommonSource.noise_warp as nw
import torch
from tqdm import tqdm

sys.path.append(rp.get_path_parent(__file__))
import raft
from background_remover import BackgroundRemover
from rp.git.CommonSource.noise_warp import NoiseWarper


def get_noise_from_video(
    video_path: str,
    noise_channels: int = 3,
    output_folder: str = None,
    visualize: bool = True,
    resize_frames: tuple = None,
    resize_flow: int = 1,
    downscale_factor: int = 1,
    device=None,
    video_preprocessor=None,
    save_files=True,
    progressive_noise_alpha=0,
    post_noise_alpha=0,
    remove_background=False,
    visualize_flow_sensitivity=None,
    warp_kwargs=dict(),
):

    # Input assertions
    assert isinstance(resize_flow, int) and resize_flow >= 1, resize_flow

    if device is None:
        if rp.currently_running_mac():
            device = "cpu"
        else:
            device = rp.select_torch_device(prefer_used=True)

    raft_model = raft.RaftOpticalFlow(device, "large")

    # Load video frames into a [T, H, W, C] numpy array, where C=3 and values are between 0 and 1
    # Can be specified as an MP4, a folder that contains images, or a glob like /path/to/*.png
    assert rp.is_numpy_array(video_path) or isinstance(video_path, str), type(
        video_path
    )
    if rp.is_video_file(video_path) or rp.is_valid_url(video_path):
        video_frames = rp.load_video(video_path)
    elif rp.is_numpy_array(video_path):
        # We can also pass a numpy video as an input in THWC form
        video_frames = video_path
        assert video_frames.ndim == 4, video_frames.ndim
        video_path = rp.get_unique_copy_path("noisewarp_video.mp4")
    else:
        if rp.is_a_folder(video_path):
            frame_paths = rp.get_all_image_files(video_path, sort_by="number")
        else:
            frame_paths = glob.glob(video_path)
            frame_paths = sorted(sorted(frame_paths), key=len)
            if not frame_paths:
                raise ValueError(
                    video_path
                    + " is not a video file, a folder of images, or a glob containing images"
                )
        video_frames = rp.load_images(frame_paths, show_progress=True)

    if video_preprocessor is not None:
        assert callable(video_preprocessor), type(video_preprocessor)
        video_frames = rp.as_numpy_array(video_frames)
        video_frames = video_preprocessor(video_frames)

    # If resize_frames is specified, resize all frames to that (height, width)
    if resize_frames is not None:
        rp.fansi_print(
            "Resizing all input frames to size %s" % str(resize_frames), "yellow"
        )
        video_frames = rp.resize_images(video_frames, size=resize_frames, interp="area")

    if remove_background:
        alphas = []

        background_remover = BackgroundRemover(device)

        if visualize and rp.running_in_jupyter_notebook():
            alpha_display_channel = rp.JupyterDisplayChannel()
            alpha_display_channel.display()

        for video_frame in rp.eta(video_frames, title="Removing Backgrounds"):
            rgba_image = background_remover(video_frame)
            alpha = rp.get_alpha_channel(rgba_image)
            alpha = rp.as_float_image(alpha)
            alphas.append(alpha)

            if visualize and rp.running_in_jupyter_notebook():
                alpha_display_channel.update(
                    rp.horizontally_concatenated_images(
                        rp.with_alpha_checkerboard(rgba_image), alpha
                    )
                )

        del background_remover  # Free GPU usage

    video_frames = rp.as_rgb_images(video_frames)
    video_frames = np.stack(video_frames)
    video_frames = video_frames.astype(np.float16) / 255
    _, h, w, _ = video_frames.shape
    rp.fansi_print(f"Input video shape: {video_frames.shape}", "yellow")

    if h % downscale_factor or w % downscale_factor:
        rp.fansi_print(
            "WARNING: height {h} or width{w} is not divisible by the downscale_factor {downscale_factor}. This will lead to artifacts in the noise."
        )

    def downscale_noise(noise):
        down_noise = rp.torch_resize_image(
            noise, 1 / downscale_factor, interp="area"
        )  # Avg pooling
        down_noise = down_noise * downscale_factor  # Adjust for STD
        return down_noise

    # Decide the location of and create the output folder
    if save_files:
        if output_folder is None:
            output_folder = "outputs/" + rp.get_file_name(
                video_path, include_file_extension=False
            )
        output_folder = rp.make_directory(rp.get_unique_copy_path(output_folder))
        rp.fansi_print("Output folder: " + output_folder, "green")

    with torch.no_grad():

        if visualize and rp.running_in_jupyter_notebook():
            # For previewing results in Jupyter notebooks, if applicable
            display_channel = rp.JupyterDisplayChannel()
            display_channel.display()

        warper = NoiseWarper(
            c=noise_channels,
            h=resize_flow * h,
            w=resize_flow * w,
            device=device,
            post_noise_alpha=post_noise_alpha,
            progressive_noise_alpha=progressive_noise_alpha,
            warp_kwargs=warp_kwargs,
        )

        prev_video_frame = video_frames[0]
        noise = warper.noise

        down_noise = downscale_noise(noise)
        numpy_noise = rp.as_numpy_image(down_noise).astype(
            np.float16
        )  # In HWC form. Using float16 to save RAM, but it might cause problems on come CPU

        numpy_noises = [numpy_noise]
        numpy_flows = []
        vis_frames = []

        try:
            for index, video_frame in enumerate(tqdm(video_frames[1:])):

                dx, dy = raft_model(prev_video_frame, video_frame)
                noise = warper(dx, dy).noise
                prev_video_frame = video_frame

                numpy_flow = np.stack(
                    [
                        rp.as_numpy_array(dx).astype(np.float16),
                        rp.as_numpy_array(dy).astype(np.float16),
                    ]
                )
                numpy_flows.append(numpy_flow)

                down_noise = downscale_noise(noise)

                numpy_noise = rp.as_numpy_image(down_noise).astype(np.float16)

                if remove_background:
                    if "background_noise" not in dir():
                        background_noise = np.random.randn(*numpy_noise.shape)
                    numpy_noise_alpha = alphas[index]
                    numpy_noise_alpha = rp.cv_resize_image(
                        numpy_noise_alpha, numpy_noise.shape[:2]
                    )
                    numpy_noise = blend_noise(
                        background_noise, numpy_noise, numpy_noise_alpha[:, :, None]
                    )

                numpy_noises.append(numpy_noise)

                if visualize:
                    flow_rgb = rp.optical_flow_to_image(
                        dx, dy, sensitivity=visualize_flow_sensitivity
                    )

                    # Turn the noise into a numpy HWC RGB array
                    down_noise_image = np.zeros((*numpy_noise.shape[:2], 3))
                    down_noise_image_c = min(noise_channels, 3)
                    down_noise_image[:, :, :down_noise_image_c] = numpy_noise[
                        :, :, :down_noise_image_c
                    ]

                    down_size = rp.get_image_dimensions(down_noise_image)
                    down_video_frame, down_flow_rgb = rp.resize_images(
                        video_frame, flow_rgb, size=down_size
                    )

                    optional_images = []
                    optional_labels = []
                    if remove_background:
                        alpha = alphas[index]
                        down_alpha = rp.cv_resize_image(alpha, down_size)

                        optional_images.append(down_alpha)
                        optional_labels.append("Alpha")

                        optional_images.append(
                            rp.with_alpha_checkerboard(
                                rp.with_image_alpha(down_video_frame, down_alpha)
                            )
                        )
                        optional_labels.append("RGBA")

                    visualization = rp.as_byte_image(
                        rp.tiled_images(
                            rp.labeled_images(
                                [
                                    down_noise_image / 3 + 0.5,
                                    down_video_frame,
                                    down_flow_rgb,
                                    down_noise_image / 5 + down_video_frame,
                                ]
                                + optional_images,
                                [
                                    "Warped Noise",
                                    "Input Video",
                                    "Optical Flow",
                                    "Overlaid",
                                ]
                                + optional_labels,
                                font="G:Zilla Slab",
                            )
                        )
                    )

                    if rp.running_in_jupyter_notebook():
                        display_channel.update(visualization)

                    vis_frames.append(visualization)

        except KeyboardInterrupt:
            rp.fansi_print(
                "Interrupted! Returning %i noises" % len(numpy_noises), "cyan", "bold"
            )
            pass

    numpy_noises = np.stack(numpy_noises).astype(np.float16)
    numpy_flows = np.stack(numpy_flows).astype(np.float16)
    if vis_frames:
        vis_frames = np.stack(vis_frames)

    if save_files and len(vis_frames):
        vis_img_folder = rp.make_directory(output_folder + "/visualization_images")
        vis_img_paths = rp.path_join(vis_img_folder, "visual_%05i.png")
        rp.save_images(vis_frames, vis_img_paths, show_progress=True)

        if "ffmpeg" in rp.get_system_commands():
            vis_mp4_path = rp.path_join(output_folder, "visualization_video.mp4")
            noise_mp4_path = rp.path_join(output_folder, "noise_video.mp4")
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
            if rp.is_video_file(video_path):
                try:
                    # If possible, try to add the original audio and framerate back again
                    # Only makes sense if the input was an MP4 file and not a folder of images etc
                    for output_video_path in [vis_mp4_path, noise_mp4_path]:
                        rp.fansi_print(
                            "Added audio to output at: "
                            + rp.add_audio_to_video_file(
                                rp.printed(
                                    rp.change_video_file_framerate(
                                        output_video_path,
                                        rp.get_video_file_framerate(video_path),
                                    )
                                ),
                                video_path,
                            ),
                            "green",
                            "bold",
                        )
                except Exception:
                    rp.print_stack_trace()
        else:
            rp.fansi_print(
                "Please install ffmpeg! We won't save an MP4 this time - please try again."
            )

    if save_files:
        noises_path = rp.path_join(output_folder, "noises.npy")
        flows_path = rp.path_join(output_folder, "flows_dxdy.npy")
        np.save(noises_path, numpy_noises)
        rp.fansi_print(
            "Saved " + noises_path + " with shape " + str(numpy_noises.shape), "green"
        )
        np.save(flows_path, numpy_flows)
        rp.fansi_print(
            "Saved " + flows_path + " with shape " + str(numpy_flows.shape), "green"
        )

        rp.fansi_print(
            rp.get_file_name(__file__)
            + ": Done warping noise, results are at "
            + rp.get_absolute_path(output_folder),
            "green",
            "bold",
        )

    return rp.gather_vars("numpy_noises numpy_flows vis_frames output_folder")


def process_video(video_path: str, output_folder: str, flow, frame):
    """
    Process a single video file and generate warped noise.
    Args:
        video_path: Path to the video file
        output_folder: Path to save the output
    """
    if rp.folder_exists(output_folder):
        raise RuntimeError(
            f"The given output_folder={repr(output_folder)} already exists! To avoid clobbering what might be in there, please specify a folder that doesn't exist so I can create one for you. Alternatively, you could delete that folder if you don't care whats in it."
        )

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
    output = get_noise_from_video(
        video,
        remove_background=False,  # Set this to True to matte the foreground - and force the background to have no flow
        visualize=True,  # Generates nice visualization videos and previews in Jupyter notebook
        save_files=True,  # Set this to False if you just want the noises without saving to a numpy file
        noise_channels=16,
        output_folder=output_folder,
        resize_frames=FRAME,
        resize_flow=FLOW,
        downscale_factor=round(FRAME * FLOW) * LATENT,
    )

    output.first_frame_path = rp.save_image(
        video[0], rp.path_join(output_folder, "first_frame.png")
    )

    rp.save_video_mp4(
        video,
        rp.path_join(output_folder, "input_template.mp4"),
        framerate=8,
        video_bitrate="max",
    )

    print("Noise shape:", output.numpy_noises.shape)
    print("Flow shape:", output.numpy_flows.shape)
    print("Output folder:", output.output_folder)


def main(
    input_folder: str, output_base_folder: str, flow, frame, selected_vids_file, newload
):
    """
    Process all videos in subfolders of the input folder.
    Each subfolder should contain a video file with the same name as the subfolder.

    Args:
        input_folder: Path to the folder containing subfolders with videos
        output_base_folder: Base folder where outputs will be saved
    """
    # Create output base folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)

    if selected_vids_file != "None":
        if not os.path.exists(selected_vids_file):
            raise FileNotFoundError(
                f"Selected videos file not found: {selected_vids_file}"
            )
        with open(selected_vids_file, "r") as f:
            # files = [line.strip().split('_')[0] for line in f if line.strip()]
            files = [line.strip() for line in f if line.strip()]

    elif selected_vids_file == "None":
        files = [
            f
            for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f))
        ]

    files = sorted(files)

    for file in files:
        print(f"Processing file: {file}")

        if newload:
            digits = "".join(ch for ch in file if ch.isdigit())
            video_name = "0" + digits[:3]
            print("DEBUG: video_name", video_name)
        else:
            video_name = file.split("_")[0]

        video_path = os.path.join(input_folder, f"{video_name}.mp4")

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found at {video_path}, skipping...")
            continue

        output_folder = os.path.join(
            output_base_folder, file, "output_noises_background"
        )

        print(f"\nProcessing video: {video_name}")
        print(f"Input video: {video_path}")
        print(f"Output folder: {output_folder}")

        try:
            process_video(video_path, output_folder, flow, frame)
            print(f"Successfully processed {video_name}")
        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Make warped noise for videos in a folder."
    )
    parser.add_argument(
        "--output_dir", type=str, default="testing_output", help="Output Directory"
    )
    parser.add_argument(
        "--flow", type=int, default=2**3, help="FLOW parameter"
    )  ## FLOW = 2**3
    parser.add_argument(
        "--frame", type=float, default=2**-1, help="FRAME parameter"
    )  ## FRAME = 2**-1

    parser.add_argument(
        "--input_folder_templatevideo",
        type=str,
        default="testing_output",
        help="Input folder containing the template videos",
    )

    parser.add_argument(
        "--selected_vids_file", type=str, default="None", help="Selected vids file"
    )
    parser.add_argument(
        "--newload",
        action="store_true",
        default=False,
        help="Whether to load the paths the new way",
    )

    args = parser.parse_args()

    output_base_folder = os.path.join(args.output_dir)

    main(
        input_folder=args.input_folder_templatevideo,
        output_base_folder=output_base_folder,
        flow=args.flow,
        frame=args.frame,
        selected_vids_file=args.selected_vids_file,
        newload=args.newload,
    )
