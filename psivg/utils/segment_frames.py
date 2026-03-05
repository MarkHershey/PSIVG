import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lang_sam import LangSAM
from PIL import Image

##### this segmentation code is based on the original code by Luca Medeiros
##### https://github.com/luca-medeiros/lang-segment-anything


def save_masks_as_npy(results, output_dir="masks_npy", object_name=None):
    """Save masks from results as .npy files"""
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        masks = result["masks"]
        if masks is not None and len(masks) > 0:
            # Save each mask separately
            for j, mask in enumerate(masks):
                if object_name is not None:
                    mask_filename = (
                        f"{output_dir}/mask_{object_name}_{i:04d}_{j:02d}.npy"
                    )
                else:
                    mask_filename = f"{output_dir}/mask_{i:04d}_{j:02d}.npy"
                np.save(mask_filename, mask)
                print(f"Saved mask to {mask_filename}")
        else:
            print(f"No masks found in result {i}")


def visualize_masks(
    image_pil,
    results,
    output_dir="mask_visualizations",
    combined_output_dir=None,
    base_name=None,
    object_name=None,
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
            axes[0].imshow(image_np)
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
            if object_name is not None:
                mask_filename = f"{output_dir}/masks_{object_name}_{i:04d}.png"
            else:
                mask_filename = f"{output_dir}/masks_{i:04d}.png"
            plt.savefig(mask_filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved mask visualization to {mask_filename}")

            # Also save to combined directory if specified
            if combined_output_dir is not None and base_name is not None:
                os.makedirs(combined_output_dir, exist_ok=True)
                if object_name is not None:
                    combined_filename = (
                        f"{combined_output_dir}/{base_name}_{object_name}_masks.png"
                    )
                else:
                    combined_filename = f"{combined_output_dir}/{base_name}_masks.png"
                plt.figure(figsize=(5 * (num_masks + 1), 5))
                fig, axes = plt.subplots(
                    1, num_masks + 1, figsize=(5 * (num_masks + 1), 5)
                )

                # Show original image
                axes[0].imshow(image_np)
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
                    blended = cv2.addWeighted(
                        image_np, 1 - alpha, colored_mask, alpha, 0
                    )

                    axes[j + 1].imshow(blended)
                    axes[j + 1].set_title(f"Mask {j+1}")
                    axes[j + 1].axis("off")

                plt.tight_layout()
                plt.savefig(combined_filename, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved combined mask visualization to {combined_filename}")


def visualize_boxes(
    image_pil, results, output_dir="box_visualizations", object_name=None
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
            if object_name is not None:
                box_filename = f"{output_dir}/boxes_{object_name}_{i:04d}.png"
            else:
                box_filename = f"{output_dir}/boxes_{i:04d}.png"
            plt.figure(figsize=(10, 8))
            plt.imshow(image_with_boxes)
            plt.title(f"Bounding Boxes - Result {i}")
            plt.axis("off")
            plt.savefig(box_filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved box visualization to {box_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Segment static images using LangSAM.")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing PNG images to process",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Base directory to save outputs"
    )
    parser.add_argument(
        "--selected_videos",
        type=str,
        required=True,
        help="Path to file listing selected video ids (e.g., 0009_static/...)",
    )
    parser.add_argument(
        "--text_prompts",
        type=str,
        required=True,
        help="Either a path to a text file with one prompt per line, or a literal prompt string",
    )
    return parser.parse_args()


def read_selected_ids(selected_videos_path):
    with open(selected_videos_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    # Extract the 4-digit id prefix from each line (before '_' or path separator)
    ids = []
    for line in lines:
        token = os.path.basename(line)
        # token may still contain suffix; prefer the leading 4 digits in the original line
        # Find first 4 consecutive digits

        digits = "".join(ch for ch in line if ch.isdigit())
        candidate = None
        # print("digits: ", digits)

        if len(digits) >= 4:
            candidate = digits[:4]

        else:
            # Fallback: strip at '_' and take first part
            candidate = token.split("_")[0]
        ids.append(candidate)
    return lines, ids


def read_prompts(text_prompts_arg):
    if os.path.isfile(text_prompts_arg):
        with open(text_prompts_arg, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    # Treat as literal single prompt
    return [text_prompts_arg]


def process_multiobject_image(image_pil, text_prompt, model, base_name, output_base):
    """Process an image with multiple objects separated by :::"""
    # Split prompt by ::: separator
    objects = [obj.strip() for obj in text_prompt.split(":::") if obj.strip()]

    if len(objects) == 1:
        # Single object - use original behavior for backward compatibility
        results = model.predict([image_pil], [text_prompt])

        masks_dir = os.path.join(output_base, "masks_npy", base_name)
        mask_viz_dir = os.path.join(output_base, "mask_visualizations", base_name)
        box_viz_dir = os.path.join(output_base, "box_visualizations", base_name)
        combined_masks_dir = os.path.join(output_base, "combined_masks")

        print(f"  Saving masks as .npy files...")
        save_masks_as_npy(results, output_dir=masks_dir)

        print(f"  Creating mask visualizations...")
        visualize_masks(
            image_pil,
            results,
            output_dir=mask_viz_dir,
            combined_output_dir=combined_masks_dir,
            base_name=base_name,
        )

        print(f"  Creating bounding box visualizations...")
        visualize_boxes(image_pil, results, output_dir=box_viz_dir)

    else:
        # Multiple objects - process each separately
        print(f"  Processing {len(objects)} objects: {objects}")

        for obj_idx, obj_prompt in enumerate(objects):
            print(f"    Processing object {obj_idx+1}/{len(objects)}: '{obj_prompt}'")

            # Use shared directories; disambiguate by object tag in filenames
            obj_name = f"obj{obj_idx+1:02d}"
            masks_dir = os.path.join(output_base, "masks_npy", base_name)
            mask_viz_dir = os.path.join(output_base, "mask_visualizations", base_name)
            box_viz_dir = os.path.join(output_base, "box_visualizations", base_name)
            combined_masks_dir = os.path.join(output_base, "combined_masks")

            try:
                # Process this object
                results = model.predict([image_pil], [obj_prompt])

                print(f"      Saving masks as .npy files...")
                save_masks_as_npy(results, output_dir=masks_dir, object_name=obj_name)

                print(f"      Creating mask visualizations...")
                visualize_masks(
                    image_pil,
                    results,
                    output_dir=mask_viz_dir,
                    combined_output_dir=combined_masks_dir,
                    base_name=base_name,
                    object_name=obj_name,
                )

                print(f"      Creating bounding box visualizations...")
                visualize_boxes(
                    image_pil, results, output_dir=box_viz_dir, object_name=obj_name
                )

            except Exception as e:
                print(f"      Error processing object '{obj_prompt}': {str(e)}")
                continue


if __name__ == "__main__":
    args = parse_args()

    image_dir = args.image_dir
    output_base = args.output_dir
    os.makedirs(output_base, exist_ok=True)

    selected_lines, selected_ids = read_selected_ids(args.selected_videos)
    print(f"Selected lines: {selected_lines}")
    print(f"Selected ids: {selected_ids}")

    prompts = read_prompts(args.text_prompts)

    # Build processing list in the order of selected_videos
    tasks = []
    for idx, vid_id in enumerate(selected_ids):
        image_filename = f"{vid_id}.png"
        image_path = os.path.join(image_dir, image_filename)
        if not os.path.isfile(image_path):
            print(f"Skipping {selected_lines[idx]} -> missing image {image_filename}")
            continue
        # Determine prompt by sequence order in selected_videos.
        # If a single prompt is provided, apply it to all entries.
        prompt = (
            prompts[0]
            if len(prompts) == 1
            else (prompts[idx] if idx < len(prompts) else prompts[-1])
        )
        tasks.append((image_filename, image_path, prompt))

    print(
        f"Found {len(tasks)} images to process (from {len(selected_ids)} selected entries)"
    )

    # Initialize the model once
    model = LangSAM()

    print("tasks: ", tasks)
    # Process each image in the derived order
    for i, (image_file, image_path, text_prompt) in enumerate(tasks):
        print(f"\nProcessing image {i+1}/{len(tasks)}: {image_file}")
        print("text_prompt: ", text_prompt)
        try:
            image_pil = Image.open(image_path).convert("RGB")
            base_name = os.path.splitext(image_file)[0]

            # Use the new multi-object processing function
            process_multiobject_image(
                image_pil, text_prompt, model, base_name, output_base
            )

            print(f"  Completed processing {image_file}")
        except Exception as e:
            print(f"  Error processing {image_file}: {str(e)}")
            continue

    print(f"\nAll {len(tasks)} images processed!")
