import json
import os
from pathlib import Path

import cv2
import groundingdino.datasets.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import psivg.constants as C
import torch
import trimesh
import yaml
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image
from psivg.helpers import print_section
from rich import print
from segment_anything import SamPredictor, sam_model_registry

from .lama_inpaint import inpaint_img_with_lama
from .my_instant_mesh import MyInstantMesh
from .rotation_est import get_object_rotation


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    local_model_path = "pretrained_models/bert-base-uncased"
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    if hasattr(args, "text_encoder_type") and os.path.exists(local_model_path):
        args.text_encoder_type = local_model_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def get_box_from_mask(mask: np.ndarray, threshold: float = 0.5):
    """
    Compute the tight bounding box of non-black pixels in an RGBA image.

    Parameters
    ----------
    mask : np.ndarray
        Float image of shape (H, W, 1) with values in [0, 1].
    threshold : float, optional
        Pixel is considered non-black if any of R,G,B > threshold.
        Use a small value like 1e-6 to ignore tiny noise. Default 0.0.

    Returns
    -------
    tuple | None
        (y_min, x_min, y_max, x_max) with inclusive min, exclusive max indices;
        or None if the image is entirely black.

    Notes
    -----
    - Pure-black background means RGB == 0 (within the threshold).
    - Alpha channel is ignored for “colorfulness” unless you want to
      consider visible (alpha>0) pixels; in that case see the variant below.
    """
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    assert mask.ndim == 3 and mask.shape[-1] == 1, "mask must have shape (H, W, 1)"
    # normalize mask to [0, 1]
    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255.0
    elif mask.dtype == bool:
        mask = mask.astype(np.float32)
    elif mask.dtype == np.float32:
        mask = mask.clip(0, 1)
    else:
        raise ValueError(f"Unsupported mask dtype: {mask.dtype}")

    # Mask of any non-black RGB
    mask = (mask > threshold).any(axis=-1)  # shape (H, W), dtype=bool

    if not mask.any():
        return None  # pure black

    # Rows and cols that contain at least one colorful pixel
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]

    y_min, y_max = ys[0], ys[-1] + 1  # exclusive max
    x_min, x_max = xs[0], xs[-1] + 1

    # return (int(y_min), int(x_min), int(y_max), int(x_max))
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def normalize_mesh(
    mesh_path: str = None,
    side=0.1,
    center_mode="bbox",
    out_path=None,
):
    """
    Center the mesh and uniformly scale it to fit inside a cube of given side length,
    centered at the origin. Returns (normalized_mesh, 4x4_transform).

    center_mode: "bbox" (AABB center) or "centroid" (mass center)
    """
    assert mesh_path is not None, "mesh_path is required"
    mesh_path = Path(mesh_path)
    assert mesh_path.exists(), f"Mesh file not found at {mesh_path}"
    m = trimesh.load(mesh_path)

    if center_mode == "bbox":
        bounds = m.bounds  # shape (2,3): [min, max]
        c = bounds.mean(axis=0)  # AABB center
        extent = (bounds[1] - bounds[0]).max()
    elif center_mode == "centroid":
        c = m.centroid  # center of mass (uniform density)
        extent = m.extents.max()  # AABB largest side (same as above but recomputed)
    else:
        raise ValueError("center_mode must be 'bbox' or 'centroid'")

    # Build transform: first translate by -c, then uniform scale so max side == side
    s = 1.0 if extent == 0 else side / float(extent)

    T = np.eye(4)
    T[:3, 3] = -c

    S = np.eye(4)
    S[:3, :3] *= s

    M = S @ T  # scale ∘ translate
    m.apply_transform(M)

    if out_path is not None:
        m.export(out_path)

    # return m, M
    return out_path


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def extract_masked_region(image, mask):
    assert image.shape[:2] == mask.shape, "Image and mask dimensions do not match"
    white_background = np.ones_like(image) * 255
    black_background = np.zeros_like(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    white_background[mask == 1] = masked_image[mask == 1]
    black_background[mask == 1] = masked_image[mask == 1]

    # Create transparent background version
    # Convert to RGBA if image is RGB
    if image.shape[2] == 3:
        transparent_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        transparent_image = image.copy()

    # Set alpha channel based on mask
    alpha_channel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    alpha_channel[mask == 1] = 255  # Fully opaque for masked regions
    alpha_channel[mask == 0] = 0  # Fully transparent for background

    # Apply mask to image and set alpha
    transparent_image = cv2.bitwise_and(
        transparent_image, transparent_image, mask=mask.astype(np.uint8)
    )
    transparent_image[:, :, 3] = alpha_channel

    return white_background, black_background, transparent_image


def extract_label(input_string):
    # match = re.match(r"([a-zA-Z]+)\(\d+\.\d+\)", input_string)
    # if match:
    #     return match.group(1)
    # else:
    #     return "other"
    if "yo - yo" in input_string:
        input_string = input_string.replace("yo - yo", "yo-yo")

    if "(" in input_string:
        tmp = input_string.split("(")[0].strip()
        tmp = tmp.replace(" ", "_")
        return tmp
    else:
        return "other"


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background
    extracted_label_list = [extract_label(label) for label in label_list]

    mask_img = torch.zeros(mask_list.shape[-2:])
    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)
    for idx, mask in enumerate(mask_list):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask_np = mask[0]
        mask_img[mask_np > 0] = value + idx + 1
        label = extracted_label_list[idx]
        cv2.imwrite(
            os.path.join(mask_dir, f"mask_{label}.jpg"),
            (mask_np * 255).astype(np.uint8),
        )
    cv2.imwrite(
        os.path.join(mask_dir, "mask.jpg"), (mask_img.numpy() * 255).astype(np.uint8)
    )

    json_data = [{"value": value, "label": "background"}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]  # the last is ')'
        if isinstance(box, torch.Tensor):
            _box = box.cpu().numpy().tolist()
        elif isinstance(box, np.ndarray):
            _box = box.tolist()
        else:
            _box = box
        json_data.append(
            {
                "value": value,
                "label": name,
                "logit": float(logit),
                "box": _box,
            }
        )
    with open(os.path.join(mask_dir, "mask.json"), "w") as f:
        json.dump(json_data, f)


def get_white_mask_area(mask_img):
    mask_array = np.array(mask_img)
    white_threshold = 250
    white_mask = mask_array >= white_threshold
    return white_mask


def main(sample_id: str, overwrite: bool = False):

    with open(C.CONFIGS_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ###########################################################################
    print_section("Starting perception pipeline")
    # cfg
    config_file = C.CONFIGS_DIR / "GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = C.PRETRAINED_MODELS_DIR / "groundingdino_swint_ogc.pth"
    sam_version = "vit_h"
    sam_checkpoint = C.PRETRAINED_MODELS_DIR / "sam_vit_h_4b8939.pth"
    OVERWRITE = overwrite

    all_sample_ids = [x.stem for x in C.INPUT_FRAMES_DIR.iterdir() if x.is_dir()]
    assert (
        sample_id in all_sample_ids
    ), f"Sample id {sample_id} not found in {C.INPUT_FRAMES_DIR}"

    # find gt_mask data if any
    gt_mask = None
    gt_mask_path = C.INPUT_MASKS_DIR / f"{sample_id}.npz"
    if gt_mask_path.exists():
        _data = np.load(gt_mask_path)
        if "mask" in _data:
            gt_mask = _data["mask"]  # (V, N, H, W) boolean arra

    obj_info_path = C.INPUT_META_DIR / f"{sample_id}.json"

    obj_info = json.load(open(obj_info_path)) if obj_info_path.exists() else {}
    obj_primary = obj_info.get("primary", "ball")
    obj_secondary = obj_info.get("secondary", None)
    if obj_secondary is not None:
        text_prompt = f"{obj_primary}.{obj_secondary}"
    else:
        text_prompt = obj_primary

    KEY_FRAME_IDXES = [0, 5]
    all_frame_idxes = [
        int(x.stem)
        for x in (C.INPUT_FRAMES_DIR / sample_id).iterdir()
        if x.is_file() and x.suffix == ".jpg"
    ]

    for frame_idx in KEY_FRAME_IDXES:
        frame_idx_str = f"{frame_idx:05d}"
        image_path = C.INPUT_FRAMES_DIR / sample_id / f"{frame_idx_str}.jpg"
        assert image_path.is_file(), f"Image file not found: {image_path}"
        output_dir = C.OUT_PERCEPTION_DIR / sample_id / frame_idx_str
        if output_dir.exists() and not OVERWRITE:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        box_threshold = 0.3
        text_threshold = 0.25
        device = "cuda"

        # make dir
        print(f"Output directory: {output_dir}")
        os.makedirs(output_dir / "objects", exist_ok=True)
        os.makedirs(output_dir / "inpaint", exist_ok=True)
        # os.makedirs(output_dir / "depth", exist_ok=True)
        # load image
        image_pil, image = load_image(image_path)

        ###########################################################################
        if gt_mask is None:
            print_section("Running Grounding DINO")
            # load grounding dino model
            dino_model = load_model(config_file, grounded_checkpoint, device=device)
            # visualize raw image
            image_pil.save(str(output_dir / "raw_image.jpg"))
            # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model=dino_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )

        ###########################################################################
        no_object_detected = False
        if gt_mask is not None:
            masks = gt_mask[frame_idx]  # take mask for this frame
            if masks.sum() == 0:
                no_object_detected = True
            pred_phrases = [f"object_{i+1}(1.0)" for i in range(masks.shape[0])]
            boxes_filt = [get_box_from_mask(mask) for mask in masks]
            boxes_filt = [torch.tensor(box) for box in boxes_filt]
            image = cv2.imread(image_path)
            masks = np.expand_dims(masks, axis=1)
            masks = torch.tensor(masks)
        else:
            print_section("Running SAM")
            # initialize SAM
            predictor = SamPredictor(
                sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
            )
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes_filt, image.shape[:2]
            ).to(device)

            try:
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(device),
                    multimask_output=False,
                )
            except Exception as e:
                print(f"Error predicting masks: {e}")
                masks = []
                no_object_detected = True

            del predictor

        ###########################################################################
        if not no_object_detected:
            print_section("Drawing output image")
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                show_box(box.numpy(), plt.gca(), label)
            plt.axis("off")
            plt.savefig(
                str(output_dir / "grounded_sam_output.jpg"),
                bbox_inches="tight",
                dpi=300,
                pad_inches=0.0,
            )
            save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

            # draw seperated objects
            label_count = {}
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            object_names = []
            for mask, box, phrase in zip(masks, boxes_filt, pred_phrases):
                label = extract_label(phrase)
                if label not in label_count:
                    label_count[label] = 1
                else:
                    label_count[label] += 1
                masked_white, masked_black, masked_transparent = extract_masked_region(
                    image, mask.cpu().numpy()[0]
                )
                x1, y1, x2, y2 = map(int, box.numpy().tolist())
                cropped_white = masked_white[y1:y2, x1:x2]
                cropped_black = masked_black[y1:y2, x1:x2]
                cropped_transparent = masked_transparent[y1:y2, x1:x2]
                object_id = (
                    f"{label}_{label_count[label]}" if label_count[label] > 1 else label
                )
                object_names.append(object_id)
                cv2.imwrite(
                    str(output_dir / "objects" / f"{object_id}.jpg"),
                    cropped_white,
                )
                cv2.imwrite(
                    str(output_dir / "objects" / f"{object_id}_black.jpg"),
                    cropped_black,
                )
                # Save transparent version as PNG
                cv2.imwrite(
                    os.path.join(
                        output_dir,
                        "objects",
                        f"{object_id}_transparent.png",
                    ),
                    cropped_transparent,
                )
            print("objects: ", label_count)

        ###########################################################################
        if not no_object_detected:
            print_section("Inpaint background using LaMa")
            torch.set_grad_enabled(True)
            img = load_img_to_array(image_path)
            last_img = img
            for i, mask in enumerate(masks):
                print(f"Inpainting mask {i+1} of {len(masks)}")
                mask = mask[0].cpu().numpy()
                img_inpainted_p = str(
                    output_dir / "inpaint" / f"inpainted_with_mask{i}.jpg"
                )
                img_inpainted = inpaint_img_with_lama(
                    img,
                    mask,
                    str(C.CONFIGS_DIR / "lama-prediction.yaml"),
                    str(C.PRETRAINED_MODELS_DIR / "big-lama"),
                    device=device,
                    dilation=50,
                    find_shade=False,
                    out_path=str(output_dir / "mask" / f"mask_{i}_final.jpg"),
                )
                last_img = inpaint_img_with_lama(
                    last_img,
                    mask,
                    str(C.CONFIGS_DIR / "lama-prediction.yaml"),
                    str(C.PRETRAINED_MODELS_DIR / "big-lama"),
                    device=device,
                    dilation=50,
                    find_shade=False,
                    out_path=str(output_dir / "mask" / f"mask_{i}_final.jpg"),
                )
                save_array_to_img(img_inpainted, img_inpainted_p)
                save_array_to_img(
                    last_img,
                    str(output_dir / "inpaint" / f"inpainted_with_mask{i}_cumu.jpg"),
                )
            final_path = str(output_dir / "inpaint" / f"inpainted_all.jpg")
            last_img = cv2.resize(
                last_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            save_array_to_img(last_img, final_path)

        ###########################################################################
        print_section("Reconstruct object mesh using InstantMesh")
        if frame_idx > 0 or no_object_detected:
            print(f"Skipping InstantMesh for frame {frame_idx}")
        else:
            mesh_model = MyInstantMesh(
                str(C.CONFIGS_DIR / "instant-mesh-large.yaml"),
                input_path=str(output_dir / "objects"),
                output_path=output_dir,
                seed=42,
            )
            mesh_model.multiview_generation(diffusion_steps=175)
            mesh_model.reconstruction(1.0, 6, 4.5, True, False)

        # overall success
        suucess_file = output_dir / "success.txt"
        suucess_file.touch()

    print_section("Estimating object rotation axis")
    object_rotation = get_object_rotation(sample_id)
    print(f"Object rotation axis: {object_rotation}")
