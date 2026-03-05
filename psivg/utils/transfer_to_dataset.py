import argparse
import os
import shutil


def main():
    input_dir = args.input_dir
    output_dir = args.output_dataset_dir
    selected_vids_file = args.selected_vids_file
    prompt_file = args.prompt_file
    prompt_fg_file = args.prompt_fg_file
    if selected_vids_file == "None" or not os.path.exists(selected_vids_file):
        raise ValueError(
            f"Selected vids file '{selected_vids_file}' does not exist or not specified."
        )

    if prompt_file == "None" or not os.path.exists(prompt_file):
        raise ValueError(
            f"Prompt file '{prompt_file}' does not exist or not specified."
        )
    if prompt_fg_file == "None" or not os.path.exists(prompt_fg_file):
        raise ValueError(
            f"Prompt fg file '{prompt_fg_file}' does not exist or not specified."
        )

    with open(prompt_file, "r", encoding="utf-8") as pf:
        prompts = [line.strip() for line in pf]
    with open(prompt_fg_file, "r", encoding="utf-8") as pff:
        prompts_fg = [line.strip() for line in pff]

    with open(selected_vids_file, "r", encoding="utf-8") as sf:
        selected_lines = [line.strip() for line in sf if line.strip()]

    for index, line in enumerate(selected_lines):
        # Determine source directory; support absolute paths or paths relative to input_dir
        potential_src = line
        if not os.path.isabs(potential_src) and not os.path.exists(potential_src):
            potential_src = os.path.join(input_dir, line)
        src_dir = potential_src
        if not os.path.isdir(src_dir):
            print(
                f"Warning: source directory '{src_dir}' does not exist or is not a directory. Skipping."
            )
            continue

        base_parts = line.split("/")
        base_name = base_parts[0]
        prompt_index = index

        if (
            prompt_index < 0
            or prompt_index >= len(prompts)
            or prompt_index >= len(prompts_fg)
        ):
            print(
                f"Warning: prompt index {prompt_index} out of range for prompts files. Skipping '{line}'."
            )
            continue

        base_name = base_parts[0]
        dest_root = os.path.join(output_dir, base_name)

        # 1) Copy noises and write noises.txt
        noises_dir = os.path.join(dest_root, "noises")
        os.makedirs(noises_dir, exist_ok=True)

        src_noises = os.path.join(src_dir, "noise_output", "noises.npy")
        dst_noises = os.path.join(noises_dir, "noises.npy")
        if os.path.exists(src_noises):
            shutil.copy2(src_noises, dst_noises)
        else:
            print(f"Warning: {src_noises} does not exist.")
        with open(os.path.join(dest_root, "noises.txt"), "w", encoding="utf-8") as f:
            f.write("noises/noises.npy\n")

        # 1a) Copy merged noises and write merged_noises.txt
        if args.with_merged_noises:
            merged_noises_dir = os.path.join(dest_root, "merged_noises")
            os.makedirs(merged_noises_dir, exist_ok=True)

            merged_noises_src = os.path.join(
                src_dir, "output_merged_noises", "dual_masked_merged_noises.npy"
            )
            merged_noises_dst = os.path.join(merged_noises_dir, "noises.npy")
            if os.path.exists(merged_noises_src):
                shutil.copy2(merged_noises_src, merged_noises_dst)
            else:
                print(
                    f"Warning: {merged_noises_src} does not exist, but with_merged_noises is True."
                )
            with open(
                os.path.join(dest_root, "merged_noises.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("merged_noises/noises.npy\n")

        ##### To transfer over the video masks with moving camera
        if args.with_merged_noises:
            video_movingbg_dir = os.path.join(dest_root, "video_movingbg")
            os.makedirs(video_movingbg_dir, exist_ok=True)

            video_movingbg_src = os.path.join(
                src_dir, "output_merged_noises", "video_movingbg.mp4"
            )
            video_movingbg_dst = os.path.join(video_movingbg_dir, "video_movingbg.mp4")
            if os.path.exists(video_movingbg_src):
                shutil.copy2(video_movingbg_src, video_movingbg_dst)
            else:
                print(
                    f"Warning: {video_movingbg_src} does not exist, but with_video_movingbg is True."
                )
            with open(
                os.path.join(dest_root, "video_movingbg.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("video_movingbg/video_movingbg.mp4\n")

        # 2) Copy video and write simulator_videos.txt
        videos_dir = os.path.join(dest_root, "simulator_videos")
        os.makedirs(videos_dir, exist_ok=True)

        src_video = os.path.join(src_dir, "input.mp4")
        dst_video_rel = os.path.join("simulator_videos", f"{base_name}.mp4")
        dst_video = os.path.join(dest_root, dst_video_rel)
        if os.path.exists(src_video):
            shutil.copy2(src_video, dst_video)
        else:
            print(f"Warning: {src_video} does not exist.")
        with open(
            os.path.join(dest_root, "simulator_videos.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(f"{dst_video_rel}\n")

        # 2a) Copy video and write videos.txt, from warped_from_first.mp4
        videos_dir = os.path.join(dest_root, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        src_video = os.path.join(src_dir, "warped_from_first.mp4")
        dst_video_rel = os.path.join("videos", f"{base_name}.mp4")
        dst_video = os.path.join(dest_root, dst_video_rel)
        if os.path.exists(src_video):
            shutil.copy2(src_video, dst_video)
        else:
            print(f"Warning: {src_video} does not exist.")
        with open(os.path.join(dest_root, "videos.txt"), "w", encoding="utf-8") as f:
            f.write(f"{dst_video_rel}\n")

        # 2b) Copy mask and write masks.txt, from the updated masks
        masks_dir = os.path.join(dest_root, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        src_mask = os.path.join(src_dir, "convex_hull_masks.mp4")
        dst_mask_rel_mp4 = os.path.join("masks", f"{base_name}.mp4")
        dst_mask_mp4 = os.path.join(dest_root, dst_mask_rel_mp4)
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask_mp4)
        else:
            print(f"Warning: {src_mask} does not exist.")
        with open(os.path.join(dest_root, "masks.txt"), "w", encoding="utf-8") as f:
            f.write(f"masks/{base_name}.mp4\n")

        # 3) Write prompt files (single first line)
        with open(os.path.join(dest_root, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(f"{prompts[prompt_index]}\n")
        with open(os.path.join(dest_root, "prompt_fg.txt"), "w", encoding="utf-8") as f:
            f.write(f"{prompts_fg[prompt_index]}\n")

        # 4) Copy input image if provided
        if args.image_folder is not None and args.image_folder != "None":
            input_image_dir = os.path.join(dest_root, "input_image")
            os.makedirs(input_image_dir, exist_ok=True)
            src_image = os.path.join(args.image_folder, f"{base_name}.png")
            dst_image = os.path.join(input_image_dir, f"{base_name}.png")
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
            else:
                print(f"Warning: {src_image} does not exist.")

        # 5) Copy corr_files if requested
        if args.with_correspondences:
            corr_files_dir = os.path.join(dest_root, "corr_files")
            os.makedirs(corr_files_dir, exist_ok=True)

            src_corr_files = os.path.join(src_dir, "corr_files")
            if os.path.exists(src_corr_files) and os.path.isdir(src_corr_files):
                # Copy all JSON files from corr_files directory
                json_files = []
                for file in os.listdir(src_corr_files):
                    if file.endswith(".json"):
                        src_file = os.path.join(src_corr_files, file)
                        dst_file = os.path.join(corr_files_dir, file)
                        shutil.copy2(src_file, dst_file)
                        json_files.append(f"corr_files/{file}")

                # Write corr_files.txt with list of JSON files
                with open(
                    os.path.join(dest_root, "corr_files.txt"), "w", encoding="utf-8"
                ) as f:
                    for json_file in sorted(json_files):
                        f.write(f"{json_file}\n")
            else:
                print(
                    f"Warning: {src_corr_files} does not exist or is not a directory."
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Transfer and rename processed data to new dataset structure."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing preprocessing results.",
    )
    parser.add_argument(
        "--output_dataset_dir", type=str, help="Output dataset directory."
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        default="None",
        help="Path to the prompt file. If default, just use the name inside the selected vids file",
    )
    parser.add_argument(
        "--total_augmentations",
        type=int,
        default=0,
        help="Total number of augmentations.",
    )

    parser.add_argument(
        "--prompt_fg_file",
        type=str,
        default="None",
        help="Path to the prompt fg file. If default, just use the name inside the selected vids file",
    )
    parser.add_argument(
        "--with_correspondences",
        action="store_true",
        default=False,
        help="Whether to also transfer the correspondences.",
    )

    parser.add_argument(
        "--with_merged_noises",
        action="store_true",
        default=False,
        help="Whether to also transfer the merged noises.",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=0,
        help="Number of layers to generate. If 0, then we are not using layers.",
    )

    parser.add_argument(
        "--selected_vids_file", type=str, default="None", help="Selected vids file."
    )

    parser.add_argument(
        "--image_folder", type=str, default="None", help="Image folder."
    )

    args = parser.parse_args()

    main()
