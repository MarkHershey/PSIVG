import json
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
import psivg.constants as C
import torch
from PIL import Image
from rich import print

from .ext.matcher.matching import Matching
from .ext.matcher.utils import make_matching_plot

Match = Tuple[float, float, float, float]  # (x, y, x', y')


def get_object_rotation(sample_id: str) -> str:
    meta_json_path = C.INPUT_META_DIR / f"{sample_id}.json"
    metadata = json.load(open(meta_json_path)) if meta_json_path.exists() else {}
    # read value from cache if exists
    rot_axis = metadata.get("primary_obj_rot_axis")
    if rot_axis is not None:
        # return cached value
        return rot_axis

    primary_obj_name = metadata.get("primary", "ball")

    frame_0_jpg_path = C.INPUT_FRAMES_DIR / sample_id / "00000.jpg"
    frame_5_jpg_path = C.INPUT_FRAMES_DIR / sample_id / "00005.jpg"
    frame_0_mask_path = (
        C.OUT_PERCEPTION_DIR
        / sample_id
        / "00000"
        / "mask"
        / f"mask_{primary_obj_name}.jpg"
    )
    frame_5_mask_path = (
        C.OUT_PERCEPTION_DIR
        / sample_id
        / "00005"
        / "mask"
        / f"mask_{primary_obj_name}.jpg"
    )
    all_exists = all(
        [
            frame_0_jpg_path.exists(),
            frame_5_jpg_path.exists(),
            frame_0_mask_path.exists(),
            frame_5_mask_path.exists(),
        ]
    )
    # print(f"all_exists: {all_exists}")
    if all_exists:
        list_of_matches = _matcher(
            frame_0_jpg_path,
            frame_5_jpg_path,
            frame_0_mask_path,
            frame_5_mask_path,
            output_dir=C.OUT_PERCEPTION_DIR / sample_id / "rot_est",
            verbose=False,
        )
        result = _estimate_rotation(list_of_matches)
    else:
        result = "None"

    cases = {
        "Roll Left (CCW)": "z-",
        "Roll Right (CW)": "z",
        "Pitch Inwards": "x-",
        "Pitch Outwards": "x",
        "Yaw Left": "y-",
        "Yaw Right": "y",
    }
    rot_axis = cases.get(result, "None")
    metadata["primary_obj_rot_axis"] = rot_axis
    json.dump(metadata, open(meta_json_path, "w"), indent=4)

    return rot_axis


def _matcher(
    image_i: str,
    image_j: str,
    mask_i=None,
    mask_j=None,
    output_dir: str = "tmp_demo_output",
    verbose: bool = False,
):
    if mask_i is None and mask_j is None:
        image_i = cv2.imread(image_i, cv2.IMREAD_GRAYSCALE)
        image_j = cv2.imread(image_j, cv2.IMREAD_GRAYSCALE)
        return _image_pair_matching(image_i, image_j, output_dir, verbose=verbose)
    else:
        # load image and mask
        image_i = Image.open(image_i).convert("RGB")
        image_j = Image.open(image_j).convert("RGB")
        mask_i = Image.open(mask_i).convert("L")
        mask_j = Image.open(mask_j).convert("L")
        # apply mask onto a black canvas so background pixels are zeroed out
        image_i_masked = Image.new("RGB", image_i.size, (0, 0, 0))
        image_i_masked.paste(image_i, mask=mask_i)
        image_j_masked = Image.new("RGB", image_j.size, (0, 0, 0))
        image_j_masked.paste(image_j, mask=mask_j)

        # match in grayscale to align with the unmasked path
        image_i_gray = cv2.cvtColor(np.array(image_i_masked), cv2.COLOR_RGB2GRAY)
        image_j_gray = cv2.cvtColor(np.array(image_j_masked), cv2.COLOR_RGB2GRAY)
        return _image_pair_matching(
            image_i_gray, image_j_gray, output_dir, verbose=verbose
        )


def _frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None, None].to(device)


@torch.no_grad()
def _image_pair_matching(
    input_image,
    ref_image,
    output_dir,
    resize=[-1],
    resize_float=False,
    superglue="indoor",
    max_keypoints=1024,
    keypoint_threshold=0.005,
    nms_radius=4,
    sinkhorn_iterations=20,
    match_threshold=0.2,
    viz=True,
    fast_viz=False,
    cache=False,
    show_keypoints=False,
    viz_extension="png",
    save=True,
    verbose: bool = False,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print('Running inference on device "{}"'.format(device))
    config = {
        "superpoint": {
            "nms_radius": nms_radius,
            "keypoint_threshold": keypoint_threshold,
            "max_keypoints": max_keypoints,
        },
        "superglue": {
            "weights": superglue,
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold,
        },
    }
    matching = Matching(config).eval().to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    if verbose:
        print('Will write matches to directory "{}"'.format(output_dir))
    if viz:
        if verbose:
            print(
                'Will write visualization images to directory "{}"'.format(output_dir)
            )

    match_nums = 0
    match_result = None

    matches_path = output_dir / "matches.npz"
    viz_path = output_dir / "matches.{}".format(viz_extension)

    do_match = True
    do_viz = viz
    if cache:
        if matches_path.exists():
            try:
                match_result = np.load(matches_path)
            except:
                raise IOError("Cannot load matches .npz file: %s" % matches_path)

            kpts0, kpts1 = match_result["keypoints0"], match_result["keypoints1"]
            matches, conf = match_result["matches"], match_result["match_confidence"]
            do_match = False

    rot0, rot1 = 0, 0

    if do_match:
        pred = matching(
            {
                "image0": _frame2tensor(input_image, device),
                "image1": _frame2tensor(ref_image, device),
            }
        )
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches, conf = pred["matches0"], pred["matching_scores0"]

        match_result = {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "matches": matches,
            "match_confidence": conf,
        }
        if save:
            np.savez(str(matches_path), **match_result)

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    match_nums = len(mkpts0)

    if match_nums > 10:
        # select the best 10 matches
        best_matches = np.argsort(mconf)[-10:]
        mkpts0 = mkpts0[best_matches]
        mkpts1 = mkpts1[best_matches]
        mconf = mconf[best_matches]
        match_nums = len(mkpts0)
    elif match_nums < 3:
        raise ValueError("Too few matches")
    else:
        # sort by mconf
        best_matches = np.argsort(mconf)[::-1]
        mkpts0 = mkpts0[best_matches]
        mkpts1 = mkpts1[best_matches]
        mconf = mconf[best_matches]
        match_nums = len(mkpts0)

    if verbose:
        print(f"match_nums: {match_nums}")
        # print(f"valid: {valid}")
        print(f"mkpts0: {mkpts0}, {len(mkpts0)}")
        print(f"mkpts1: {mkpts1}, {len(mkpts1)}")
        print(f"mconf: {mconf}, {len(mconf)}")
    all_matches: list[Match] = []
    for i in range(match_nums):
        _match = (mkpts0[i][0], mkpts0[i][1], mkpts1[i][0], mkpts1[i][1])
        all_matches.append(_match)

    if do_viz:
        color = cm.jet(mconf)
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
            "Matches: {}".format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append("Rotation: {}:{}".format(rot0, rot1))

        k_thresh = matching.superpoint.config["keypoint_threshold"]
        m_thresh = matching.superglue.config["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
        ]

        make_matching_plot(
            input_image,
            ref_image,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            viz_path,
            show_keypoints,
            fast_viz,
            small_text,
        )

        # draw matched keypoints on the image
        input_image_copy = input_image.copy()
        for i in range(match_nums)[:3]:
            cv2.circle(
                input_image_copy,
                (int(mkpts0[i][0]), int(mkpts0[i][1])),
                5,
                (0, 0, 255),
                -1,
            )
            # add label to the keypoint
            cv2.putText(
                input_image_copy,
                f"({int(mkpts0[i][0])}, {int(mkpts0[i][1])})",
                (int(mkpts0[i][0]), int(mkpts0[i][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(output_dir / "matched_keypoints.png", input_image_copy)

    del matching

    return all_matches


def _estimate_rotation(matches, min_points=4, threshold=0.1):
    """
    Estimates the coarse 3D rotation of an object given 2D keypoint matches
    between two frames.

    Args:
        matches (list): A list of tuples, where each tuple contains four floats:
                        (x, y, x_prime, y_prime)
                        (x, y) are coordinates in frame i
                        (x_prime, y_prime) are coordinates in frame j
        min_points (int): The minimum number of valid points required to
                          make an estimate.
        threshold (float): The minimum absolute score required to declare
                           a dominant rotation.

    Returns:
        str: A string describing the dominant rotation
             (e.g., "Roll Left (CCW)", "Pitch Inwards", "Yaw Right",
             "Undetermined").
    """

    # --- Step 0: Data Prep ---
    try:
        P_i = np.array([match[0:2] for match in matches], dtype=float)
        P_j = np.array([match[2:4] for match in matches], dtype=float)
    except (ValueError, IndexError):
        return "Error: Invalid 'matches' format."

    if len(P_i) < min_points:
        return f"Undetermined (Insufficient points: {len(P_i)} < {min_points})"

    # --- Step 1: Isolate Rotational Motion (De-translation) ---
    C_i = np.mean(P_i, axis=0)
    C_j = np.mean(P_j, axis=0)
    V_i = P_i - C_i
    V_j = P_j - C_j

    # --- Step 2: Calculate Rotation Scores ---
    V_i_mags = np.linalg.norm(V_i, axis=1)
    V_j_mags = np.linalg.norm(V_j, axis=1)

    valid_mask = V_i_mags > 1e-6

    if np.sum(valid_mask) < min_points:
        return f"Undetermined (Insufficient valid points after filtering: {np.sum(valid_mask)} < {min_points})"

    x = V_i[valid_mask, 0]
    y = V_i[valid_mask, 1]

    F = V_j[valid_mask] - V_i[valid_mask]
    fx = F[:, 0]
    fy = F[:, 1]

    S = V_j_mags[valid_mask] / V_i_mags[valid_mask]

    # 1. Roll Score (Z-axis Curl)
    curl_components = (x * fy) - (y * fx)
    roll_score = np.mean(curl_components)

    # 2. Pitch Score (X-axis Divergence)
    pitch_corr_matrix = np.corrcoef(y, S)
    pitch_score = pitch_corr_matrix[0, 1]

    # 3. Yaw Score (Y-axis Divergence)
    yaw_corr_matrix = np.corrcoef(x, S)
    yaw_score = yaw_corr_matrix[0, 1]

    scores = {
        "roll": 0.0 if np.isnan(roll_score) else roll_score,
        "pitch": 0.0 if np.isnan(pitch_score) else pitch_score,
        "yaw": 0.0 if np.isnan(yaw_score) else yaw_score,
    }

    # --- Step 3: Interpret the Results ---
    dominant_axis = max(scores, key=lambda k: abs(scores[k]))
    dominant_value = scores[dominant_axis]

    if abs(dominant_value) < threshold:
        return "None"

    if dominant_axis == "roll":
        return "Roll Left (CCW)" if dominant_value < 0 else "Roll Right (CW)"

    elif dominant_axis == "pitch":
        return "Pitch Outwards" if dominant_value > 0 else "Pitch Inwards"

    elif dominant_axis == "yaw":
        return "Yaw Right" if dominant_value > 0 else "Yaw Left"

    return "None"


def _perspective_project(points_3d, f=500, c=(500, 500)):
    """Helper for mock data: Simple 3D to 2D perspective projection."""
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z_cam = z + 10.0
    x_proj = (f * x / z_cam) + c[0]
    y_proj = (-f * y / z_cam) + c[1]
    return np.vstack([x_proj, y_proj]).T


if __name__ == "__main__":
    ...
