"""
Export the vipe results.

NOTE that for Viser and Vipe results, it use the RH COLMAP/OpenCV convention, where
- +X points to RIGHT
- +Y points to DOWN
- +Z points to FORWARD (along viewing direction)

For Mitsuba, it uses is RH as well, but with
- +X points to LEFT
- +Y points to UP
- +Z points to FORWARD (along viewing direction)

"""

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
import trimesh
from arrgh import arrgh
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
from psivg.constants import (
    INPUT_META_DIR,
    OUT_PERCEPTION_DIR,
    VIPE_EXPORT_DIR,
    VIPE_RAW_DIR,
)
from rich import print
from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_instance_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)
from vipe.utils.PSIVG_pcp import advanced_trim_outliers, simple_trim_outliers

import viser
import viser.transforms as tf

from .PSIVG_grid_io import save_grid


def load_mask_map(txt_path: Path) -> dict[int, str]:
    with open(txt_path, "r") as f:
        lines = f.readlines()
    mask_map = {}
    for line in lines:
        mask_id, mask_name = line.strip().split(": ")
        mask_map[int(mask_id)] = mask_name
    return mask_map


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
        return out_path
    else:
        return m


def get_points_stats(pcd: np.ndarray) -> dict:
    stats = dict(
        min=np.min(pcd, axis=0),
        max=np.max(pcd, axis=0),
        mean=np.mean(pcd, axis=0),
        std=np.std(pcd, axis=0),
        median=np.median(pcd, axis=0),
        q25=np.percentile(pcd, 25, axis=0),
        q75=np.percentile(pcd, 75, axis=0),
        q90=np.percentile(pcd, 90, axis=0),
        q95=np.percentile(pcd, 95, axis=0),
        q99=np.percentile(pcd, 99, axis=0),
    )
    print(f"stats: {json.dumps(stats, indent=2, default=str)}")
    return stats


def transform_pcd(pcd: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    Assumes pcd has dimension of (N, 3) or (..., 3)
    Assumes c2w has dimension of (4, 4)
    """
    assert pcd.shape[-1] == 3, "pcd must have dimension of (..., 3)"
    assert c2w.shape == (4, 4), "c2w must be a 2D array of shape (4, 4)"
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    if pcd.ndim != 2:
        original_shape = pcd.shape[:]
        out = pcd.reshape(-1, 3) @ R.T + t
        return out.reshape(original_shape)
    else:
        return pcd @ R.T + t


###############################################################################
### Code used to transform the world frame
###############################################################################


def apply_Rts_to_pcd(
    pcd: np.ndarray, R: np.ndarray, t: np.ndarray, s: float = 1.0
) -> np.ndarray:
    """
    Apply a 3x3 rotation matrix, a 3D translation vector, and a scaling factor to a point cloud.
    """
    assert pcd.shape[-1] == 3, "pcd must have dimension of (..., 3)"
    assert R.shape == (3, 3), "R must be a 2D array of shape (3, 3)"
    assert t.shape == (3,), "t must be a 1D array of shape (3,)"
    assert s > 0, "s must be a positive float"

    return (pcd @ R.T) * s + t


def transform_world(
    pcd: np.ndarray,
    current_box_center: np.ndarray,
    new_box_center: np.ndarray,
    R: np.ndarray,
    s: float = 1.0,
) -> np.ndarray:
    """
    Transform a point cloud from the current world frame to the new world frame.

    current_box_center indicate the center of rotation for the R rotation
    """
    assert pcd.shape[-1] == 3, "pcd must have dimension of (..., 3)"
    assert current_box_center.shape == (
        3,
    ), "current_box_center must be a 1D array of shape (3,)"
    assert new_box_center.shape == (
        3,
    ), "new_box_center must be a 1D array of shape (3,)"
    assert R.shape == (3, 3), "R must be a 2D array of shape (3, 3)"
    assert s > 0, "s must be a positive float"

    # first, translate the point cloud to the current box center
    pcd = pcd - current_box_center
    # then, apply the rotation
    pcd = pcd @ R.T
    # then, apply the scaling
    pcd = pcd * s
    # then, translate the point cloud to the new box center
    pcd = pcd + new_box_center
    return pcd


_EPS = 1e-12


def _as_hom(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack 3x3 R, 3-vector t into a 4x4 homogeneous transform."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _from_hom(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack 4x4 homogeneous transform into (R, t)."""
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def build_transform(p0, px, py, pz):
    """
    Construct a numerically stable Sim(3) that maps NEW coords -> OLD coords.

    Inputs (3-vectors in *old* world coordinates):
      p0 : where new (0,0,0) should be
      px : where new (1,0,0) should be
      py : where new (0,1,0) should be
      pz : where new (0,0,1) should be

    Returns dict with:
      - 'F': 4x4 mapping new_world -> old_world  (F = [s R | t])
      - 'M': 4x4 mapping old_world -> new_world  (M = F^{-1})
      - 's','R','t': scalar scale, rotation (3x3), translation (3,)
      - 'B_raw': 3x3 basis before snapping (columns = px-p0, py-p0, pz-p0)
      - 'B_sim': 3x3 = s*R (snapped similarity basis)
    """
    p0 = np.array(p0, dtype=float)
    px = np.array(px, dtype=float)
    py = np.array(py, dtype=float)
    pz = np.array(pz, dtype=float)
    assert p0.shape == px.shape == py.shape == pz.shape == (3,)

    # Raw basis (old-frame coordinates of the new unit axes)
    vx, vy, vz = px - p0, py - p0, pz - p0
    B_raw = np.column_stack([vx, vy, vz])  # shape (3,3)

    # SVD-based polar “snap” to Sim(3): B_raw ≈ s * R
    # Closest proper rotation R = U * D * V^T (with det(R)=+1).
    U, S, Vt = np.linalg.svd(B_raw)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt

    # Uniform scale minimizing ||B_raw - s R||_F is s = trace(S*D)/3
    # (Umeyama-style). Ensure positive s (degenerate/reflective cases check).
    s = float(np.trace(np.diag(S) @ D) / 3.0)
    if s <= _EPS:
        raise ValueError("Degenerate basis: estimated uniform scale is non-positive.")

    B_sim = s * R
    t = p0.copy()

    # Homogeneous transforms:
    F = np.eye(4, dtype=float)
    F[:3, :3] = B_sim
    F[:3, 3] = t

    # Analytic inverse for numerical stability:
    # M = [ (1/s) R^T  |  -(1/s) R^T t ]
    M = np.eye(4, dtype=float)
    M[:3, :3] = (1.0 / s) * R.T
    M[:3, 3] = -M[:3, :3] @ t

    return {"F": F, "M": M, "s": s, "R": R, "t": t, "B_raw": B_raw, "B_sim": B_sim}


def transform_points(X_old: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Transform points from OLD world to NEW world using 4x4 M (old->new).

    Accepts X_old with shape (..., 3) or (3,), returns same shape.

    Implementation uses row-vector form: X_new = X_old @ R^T + t,
    where R = M[:3,:3], t = M[:3,3].
    """
    X = np.asarray(X_old, dtype=float)
    if X.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {X.shape}")
    R = M[:3, :3]
    t = M[:3, 3]
    # matmul broadcasts over leading dims; result keeps the same shape as X
    return X @ R.T + t


# --- Camera transforms -------------------------------------------------------


def transform_pose_cw(
    R_cw: np.ndarray, t_cw: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update camera *pose* (camera-to-world): X_w = R_cw X_c + t_cw
    under the world change new->old given by F.
    New pose is T'_cw = F * T_cw.
    """
    T_cw = _as_hom(R_cw, t_cw)
    T_cw_new = F @ T_cw
    return _from_hom(T_cw_new)


def transform_pose_c2w_hom(c2w: np.ndarray, M: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Transform a camera-to-world pose from the old world to the new world.
    """
    assert c2w.shape == (4, 4), "c2w must be a 2D array of shape (4, 4)"
    R_c2w, t_old = _from_hom(c2w)
    R_new = R.T @ R_c2w
    t_new = M[:3, :3] @ t_old + M[:3, 3]
    return _as_hom(R_new, t_new)


def transform_extrinsics_wc(
    R_wc: np.ndarray, t_wc: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update world-to-camera extrinsics: X_c = R_wc X_w + t_wc
    when world is remapped by X_old = F * X_new  (i.e., X_new = M * X_old).
    New extrinsics are T'_wc = T_wc * F.
    """
    T_wc = _as_hom(R_wc, t_wc)
    T_wc_new = T_wc @ F
    return _from_hom(T_wc_new)


def update_intrinsics_for_image_scale(K: np.ndarray, alpha: float) -> np.ndarray:
    """
    If (and only if) you rescale the camera image by 'alpha',
    scale intrinsics accordingly. Otherwise, K stays the same.
    """
    K = np.array(K, dtype=float)
    K_new = K.copy()
    K_new[0, 0] *= alpha  # fx
    K_new[1, 1] *= alpha  # fy
    K_new[0, 2] *= alpha  # cx
    K_new[1, 2] *= alpha  # cy
    return K_new


# --- Optional convenience: convert between forms -----------------------------


def wc_to_cw(R_wc: np.ndarray, t_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World->camera to camera->world."""
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw


def cw_to_wc(R_cw: np.ndarray, t_cw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera->world to world->camera."""
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    return R_wc, t_wc


###############################################################################
### above Code used to transform the world frame
###############################################################################


def get_video_static_bg_pcd(
    video_data: list[dict],
    max_num_points: int = 1_000_000,
    trim_floating_outliers: bool = True,
    return_unreliable: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    bg_pcd = []
    bg_rgb = []
    unreliable_bg_pcd = []
    unreliable_bg_rgb = []
    for frame_dict in video_data:
        bg_mask = frame_dict["bg_mask"].reshape(-1)  # (H, W) -> (H * W)
        depth_mask = frame_dict["depth_mask"].reshape(-1)  # (H, W) -> (H * W)
        combined_mask = bg_mask & depth_mask  # (H * W)

        full_pcd = frame_dict["full_pcd"].reshape(-1, 3)  # (H, W, 3) -> (H * W, 3)
        full_rgb = frame_dict["full_rgb"].reshape(-1, 3)  # (H, W, 3) -> (H * W, 3)
        bg_pcd.append(full_pcd[combined_mask])
        bg_rgb.append(full_rgb[combined_mask])

        if return_unreliable:
            _mask = ~bg_mask & ~depth_mask
            unreliable_bg_pcd.append(full_pcd[_mask])
            unreliable_bg_rgb.append(full_rgb[_mask])

    bg_pcd = np.concatenate(bg_pcd, axis=0)
    bg_rgb = np.concatenate(bg_rgb, axis=0)

    if return_unreliable:
        unreliable_bg_pcd = np.concatenate(unreliable_bg_pcd, axis=0)
        unreliable_bg_rgb = np.concatenate(unreliable_bg_rgb, axis=0)

    _count = bg_pcd.shape[0]
    if _count > max_num_points:
        random_indices = np.random.choice(
            bg_pcd.shape[0], max_num_points, replace=False
        )
        bg_pcd = bg_pcd[random_indices]
        bg_rgb = bg_rgb[random_indices]
        print(f"Randomly downsampled from {_count:,} to {bg_pcd.shape[0]:,} points")

    if return_unreliable:
        if unreliable_bg_pcd.shape[0] > max_num_points:
            random_indices = np.random.choice(
                unreliable_bg_pcd.shape[0], max_num_points, replace=False
            )
            unreliable_bg_pcd = unreliable_bg_pcd[random_indices]
            unreliable_bg_rgb = unreliable_bg_rgb[random_indices]

    ### Trim floating outliers
    if trim_floating_outliers:
        mask1 = simple_trim_outliers(bg_pcd)
        mask2 = advanced_trim_outliers(bg_pcd)
        mask = mask1 & mask2
        bg_pcd = bg_pcd[mask]
        bg_rgb = bg_rgb[mask]
        removed = mask.shape[0] - mask.sum()
        print(f"Removed outliers {removed:,} points")
        print(f"Final background points: {mask.sum():,}\n")

    if return_unreliable:
        return bg_pcd, bg_rgb, unreliable_bg_pcd, unreliable_bg_rgb
    else:
        return bg_pcd, bg_rgb


def print_1D_line(min, median, mean, p75, p85, p95, p99, max):
    _range = max - min
    name = ["min", "50", "μ", "75", "85", "95", "99", "max"]
    original_values = [min, median, mean, p75, p85, p95, p99, max]
    norm_values = [(v - min) / _range for v in original_values]
    norm_100 = [int(v * 100) for v in norm_values]
    for i in range(100):
        if i in norm_100:
            _idx = norm_100.index(i)
            print(name[_idx], end="")
        else:
            print("-", end="")
    print("max")


def process_perframe_foreground_pcd(
    depth: np.ndarray, fg_mask: np.ndarray, threshold: float = 85
) -> np.ndarray:
    """
    Process the per-frame foreground point clouds.
    Takes in depth, foreground mask

    check depth distribution
    return a new foreground mask
    """
    # print(type(depth), type(fg_mask))
    # print(depth.shape, fg_mask.shape)
    # depth_copy = deepcopy(depth).numpy().reshape(-1)
    # fg_mask_copy = deepcopy(fg_mask).reshape(-1)
    # fg_points = depth_copy[fg_mask_copy].reshape(-1)

    # min_depth = np.min(fg_points)
    # median_depth = np.median(fg_points)
    # mean_depth = np.mean(fg_points)
    # p75_depth = np.percentile(fg_points, 75)
    # p85_depth = np.percentile(fg_points, 80)
    # p95_depth = np.percentile(fg_points, 95)
    # p99_depth = np.percentile(fg_points, 99)
    # max_depth = np.max(fg_points)
    # print_1D_line(min_depth, median_depth, mean_depth, p75_depth, p85_depth, p95_depth, p99_depth, max_depth)\

    assert 50 <= threshold <= 100, "Threshold must be between 50 and 100"
    _depth = depth.numpy() if isinstance(depth, torch.Tensor) else depth
    _fg_mask = fg_mask.numpy() if isinstance(fg_mask, torch.Tensor) else fg_mask
    assert _depth.shape == _fg_mask.shape

    threshold_depth = np.percentile(_depth[_fg_mask], threshold)
    new_fg_mask = _fg_mask & (_depth < threshold_depth)
    return new_fg_mask


def get_pts_centroid(pcd: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    assert pcd.shape[-1] == 3, "The last dimension of pcd must be 3"
    if pcd.ndim != 2:
        _pcd = pcd.reshape(-1, 3)
    else:
        _pcd = pcd
    if mask is not None:
        _mask = mask.reshape(-1)

    if mask is None:
        return np.mean(_pcd, axis=0)  # (3,)
    else:
        return np.mean(_pcd[_mask], axis=0)  # (3,)


def get_ball_radius_old(depth: np.ndarray, fg_mask: np.ndarray) -> float:
    _max_depth = np.max(depth[fg_mask])
    _min_depth = np.min(depth[fg_mask])
    _radius = _max_depth - _min_depth
    return float(_radius)


def get_ball_radius(camera_space_pcd: np.ndarray, fg_mask: np.ndarray) -> float:
    _pts = camera_space_pcd[fg_mask].reshape(-1, 3)
    _max = np.max(_pts, axis=0)
    _min = np.min(_pts, axis=0)
    _width = _max[0] - _min[0]
    _height = _max[1] - _min[1]
    _avg_length = (_width + _height) / 2
    return float(_avg_length / 2)


def export_video_data(
    current_artifact: ArtifactPath,
    spatial_subsample: int = 1,
    temporal_subsample: int = 1,
):
    sample_id = current_artifact.artifact_name
    print(
        f"Exporting data from artifact: [green bold]{sample_id}[/green bold] with {spatial_subsample=} and {temporal_subsample=}"
    )
    meta_file = INPUT_META_DIR / f"{sample_id}.json"

    if meta_file.exists():
        with open(meta_file, "r") as f:
            obj_info = json.load(f)
        primary_obj = obj_info.get("primary")
        secondary_obj = obj_info.get("secondary")
        if not primary_obj:
            raise ValueError(f"No primary object found in {meta_file}")
    else:
        raise ValueError(f"No object info file found for {sample_id}")

    rays: np.ndarray | None = None

    def none_it(inner_it):
        try:
            for item in inner_it:
                yield item
        except FileNotFoundError:
            while True:
                yield None, None

    mask_map: dict[int, str] = load_mask_map(current_artifact.mask_phrase_path)

    background_mask_id = None
    primary_mask_ids = []
    secondary_mask_ids = []
    sky_mask_ids = []
    for mask_id, mask_name in mask_map.items():

        if mask_name == primary_obj:
            primary_mask_ids.append(mask_id)
        elif mask_name == secondary_obj:
            secondary_mask_ids.append(mask_id)
        elif mask_name == "background":
            background_mask_id = mask_id
        elif mask_name == "sky":
            sky_mask_ids.append(mask_id)
        else:
            pass
            print(f"Ignoring mask name: {mask_name}")

    if len(primary_mask_ids) == 0:
        print(f"mask_map: {mask_map}")
        raise ValueError(f"Primary object not detected by ViPE")
    if background_mask_id != 0:
        if not background_mask_id:
            raise ValueError(
                f"Background mask not found in {current_artifact.mask_phrase_path}"
            )
        else:
            print(f"Unexpected background mask ID: {background_mask_id}")

    if secondary_obj and len(secondary_mask_ids) == 0:
        raise ValueError(f"Secondary object not detected by ViPE")

    print(f"background_mask_id: {background_mask_id}")
    print(f"sky_mask_ids      : {sky_mask_ids}\n")
    print(f"primary_mask_ids  : {primary_mask_ids}")
    print(f"secondary_mask_ids: {secondary_mask_ids}")

    video_data: list[dict] = []

    for frame_idx, (
        c2w,
        (_, rgb),
        intr,
        camera_type,
        (_, depth),
        (_, masks),
    ) in enumerate(
        zip(
            read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy(),
            read_rgb_artifacts(current_artifact.rgb_path),
            *read_intrinsics_artifacts(
                current_artifact.intrinsics_path, current_artifact.camera_type_path
            )[1:3],
            none_it(read_depth_artifacts(current_artifact.depth_path)),
            read_instance_artifacts(current_artifact.mask_path),
        )
    ):
        if frame_idx % temporal_subsample != 0:
            continue

        print(f"Loading frame {frame_idx}", end="\r", flush=True)

        pinhole_intr = camera_type.build_camera_model(intr).pinhole().intrinsics
        frame_height, frame_width = rgb.shape[:2]
        frame_aspect_ratio = frame_width / frame_height
        fov = 2 * np.arctan2(frame_height / 2, pinhole_intr[0].item())

        sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
        sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

        if rays is None:
            camera_model = camera_type.build_camera_model(intr)
            disp_v, disp_u = torch.meshgrid(
                torch.arange(frame_height).float()[::spatial_subsample],
                torch.arange(frame_width).float()[::spatial_subsample],
                indexing="ij",
            )
            if camera_type == CameraType.PANORAMA:
                disp_v = disp_v / (frame_height - 1)
                disp_u = disp_u / (frame_width - 1)
            disp = torch.ones_like(disp_v)
            pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
            rays = pts[..., :3].numpy()
            if camera_type != CameraType.PANORAMA:
                rays /= rays[..., 2:3]

        if depth is not None:
            depth_np = depth.numpy()[::spatial_subsample, ::spatial_subsample]
            pcd = rays * depth_np[..., None]

            depth_mask = reliable_depth_mask_range(depth)[
                ::spatial_subsample, ::spatial_subsample
            ].numpy()
            masks = masks[::spatial_subsample, ::spatial_subsample].numpy()

            # process background mask
            background_mask = masks == background_mask_id  # (H, W)

            # process primary object mask
            primary_obj_mask = np.zeros_like(masks, dtype=bool)  # (H, W)
            for mask_id in primary_mask_ids:
                primary_obj_mask |= masks == mask_id
            if primary_obj_mask.sum() > 0:
                primary_obj_mask = process_perframe_foreground_pcd(
                    depth_np, primary_obj_mask, threshold=85
                )
                primary_obj_radius = get_ball_radius(pcd, primary_obj_mask)
            else:
                primary_obj_radius = 0.0

            # process secondary object mask
            secondary_obj_mask = np.zeros_like(masks, dtype=bool)  # (H, W)
            for mask_id in secondary_mask_ids:
                secondary_obj_mask |= masks == mask_id
            if secondary_obj_mask.sum() > 0:
                secondary_obj_mask = process_perframe_foreground_pcd(
                    depth_np, secondary_obj_mask, threshold=85
                )
                secondary_obj_radius = get_ball_radius(pcd, secondary_obj_mask)
            else:
                secondary_obj_radius = 0.0

            # process sky mask
            sky_mask = np.zeros_like(masks, dtype=bool)  # (H, W)
            for mask_id in sky_mask_ids:
                sky_mask |= masks == mask_id

            # Transform the frame-level point cloud from camera space to world space
            pcd = transform_pcd(pcd, c2w)  # (H, W, 3)

            # Estimate the centroid of the primary object
            if primary_obj_mask.sum() > 0:
                primary_obj_centroid = get_pts_centroid(pcd[primary_obj_mask])  # (3,)
            else:
                primary_obj_centroid = np.array([0.0, 0.0, 0.0])

            # Estimate the centroid of the secondary object
            if secondary_obj_mask.sum() > 0:
                secondary_obj_centroid = get_pts_centroid(
                    pcd[secondary_obj_mask]
                )  # (3,)
            else:
                secondary_obj_centroid = np.array([0.0, 0.0, 0.0])

            # make sure all masks and rgb have the same shape
            assert (
                background_mask.shape
                == primary_obj_mask.shape
                == secondary_obj_mask.shape
                == depth_mask.shape
                == sampled_rgb.shape[:2]
            )

        else:
            pcd, depth_mask, background_mask, primary_obj_mask, secondary_obj_mask = (
                None,
                None,
                None,
                None,
                None,
            )

        thumbnail_rgb = make_thumbnail(sampled_rgb)

        frame_dict = dict(
            idx=frame_idx,  # int
            c2w=c2w,  # (4, 4) numpy array
            K=intr.numpy(),  # (4, ) numpy array
            fov=fov.item(),  # float
            thumbnail_rgb=thumbnail_rgb,  # (H, W, 3)
            full_rgb=sampled_rgb,  # (H, W, 3)
            full_pcd=pcd,  # (H, W, 3)
            full_depth=depth,  # (H, W)
            depth_mask=depth_mask,  # (H, W)
            bg_mask=background_mask,  # (H, W)
            # fg_mask=foreground_mask,  # (H, W)
            primary_obj_name=primary_obj,  # str
            primary_obj_mask=primary_obj_mask,  # (H, W)
            secondary_obj_name=secondary_obj,  # str
            secondary_obj_mask=secondary_obj_mask,  # (H, W)
            frame_aspect_ratio=frame_aspect_ratio,  # float
            # ball_radius=ball_radius,  # float
            # ball_centroid=ball_centroid,  # (3,)
            primary_obj_radius=primary_obj_radius,  # float
            secondary_obj_radius=secondary_obj_radius,  # float
            primary_obj_centroid=primary_obj_centroid,  # (3,) numpy array
            secondary_obj_centroid=secondary_obj_centroid,  # (3,) numpy array
        )

        video_data.append(frame_dict)

    return video_data


def make_thumbnail(rgb: np.ndarray, max_height: int = 200) -> np.ndarray:
    thumbnail = Image.fromarray(rgb)
    H, W = thumbnail.size
    H = min(H, max_height)
    W = int(W * (H / H))
    thumbnail.thumbnail((W, H), Image.Resampling.LANCZOS)
    return np.array(thumbnail)


def get_rainbow_color(N: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(int(N)):
        color = cm.jet(i / N)[:3]  # r,g,b in [0,1]
        color = tuple((int(c * 255) for c in color))  # r,g,b in [0,255]
        colors.append(color)
    return colors


def is_json_serializable(
    value: Any, *, strict: bool = True, _seen: set[int] | None = None
) -> bool:
    """
    Returns True if `value` can be encoded by the stdlib `json` module
    (without custom encoders).

    strict=True  -> JSON spec:
        - dict keys must be str
        - floats must be finite (no NaN/Infinity)
    strict=False -> match Python's json more closely:
        - dict keys may be str|int|float|bool (still must be finite if float)
        - NaN/Infinity allowed (like json.dumps default)
    Cycles are rejected.
    """
    if _seen is None:
        _seen = set()

    # Track containers to break cycles
    if isinstance(value, (list, tuple, dict)):
        obj_id = id(value)
        if obj_id in _seen:
            return False
        _seen.add(obj_id)

    # Scalars
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value) if strict else True

    # Sequences (JSON arrays); tuples serialize as lists
    if isinstance(value, (list, tuple)):
        return all(is_json_serializable(v, strict=strict, _seen=_seen) for v in value)

    # Mappings (JSON objects)
    if isinstance(value, dict):
        keys = value.keys()
        if strict:
            if not all(isinstance(k, str) for k in keys):
                return False
        else:
            for k in keys:
                if not isinstance(k, (str, int, float, bool)):
                    return False
                if isinstance(k, float) and not math.isfinite(k):
                    return False
        return all(
            is_json_serializable(v, strict=strict, _seen=_seen) for v in value.values()
        )

    # Everything else (set, bytes, Decimal, dataclass, numpy scalars, etc.)
    return False


def make_json_serializable(value: Any, numpy_size_threshold: int = 64) -> Any:
    """
    A recursive function that converts any value or container to a JSON serializable value (from top to bottom).
    This function is primarily used to handle non-serializable arrays (e.g., numpy arrays, torch tensors) by converting them to lists,
    or discard them if they are larger than the threshold (numpy_size_threshold).
    """
    if is_json_serializable(value):
        return value
    elif isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
        value = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
        if value.size <= numpy_size_threshold:
            return value.tolist()
        else:
            print(f"Array too big to save: {value.shape}")
            return f"Array too big to save: {value.shape}"
    elif isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list) or isinstance(value, tuple):
        return [make_json_serializable(v) for v in value]
    else:
        return value


def save_data_dict(
    data_dict: dict,
    data_name: str,
    save_root: Path,
    overwrite: bool = True,
    use_grid_io: bool = True,
    numpy_size_threshold: int = 64,
):
    assert isinstance(data_dict, dict), "data_dict must be a dictionary"
    assert isinstance(data_name, str), "data_name must be a string"
    assert isinstance(
        numpy_size_threshold, int
    ), "numpy_size_threshold must be an integer"

    _start_time = time.time()

    save_root = Path(save_root).resolve()
    save_dir = save_root / data_name
    if save_dir.exists():
        if overwrite:
            print(f"[yellow]Overwriting save directory[/yellow] {save_dir}")
            shutil.rmtree(save_dir)
        else:
            print(f"[yellow]Skipping existing save directory[/yellow] {save_dir}")
            return
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / f"{data_name}.json"

    if use_grid_io:
        json_dict = dict()  # to put serializable data
        for key, value in data_dict.items():
            if is_json_serializable(value):
                json_dict[key] = value
            elif isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                value = (
                    value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                )
                # if number of entries is less than 64, save as a list
                if value.size <= numpy_size_threshold:
                    # to save as a list
                    json_dict[key] = value.tolist()
                else:
                    save_grid(save_dir / f"{key}.npz", value)
                    json_dict[key] = f"{key}.npz"
            else:
                print(
                    f"[red]Key '{key}' is ignored because its value type ({type(value)}) is not expected by save_data_dict"
                )
    else:
        json_dict = make_json_serializable(
            data_dict, numpy_size_threshold=numpy_size_threshold
        )

    with json_path.open("w") as f:
        json.dump(json_dict, f, indent=2)

    time_taken = time.time() - _start_time
    print(f"[green]✔[/green] Data saved to {save_dir} (took {time_taken:.1f} seconds)")
    return save_dir


def axis_reflection(axis="z"):
    if axis == "x":
        return np.diag([-1, 1, 1])
    if axis == "y":
        return np.diag([1, -1, 1])
    if axis == "z":
        return np.diag([1, 1, -1])
    raise ValueError("axis must be 'x', 'y', or 'z'")


def convert_pose_LH_RH(R, t, axis="z"):
    """
    R: (3,3) rotation matrix (LH)
    t: (3,) translation vector (LH)
    returns R_RH, t_RH
    """
    S = axis_reflection(axis)
    R_new = S @ R @ S
    t_new = S @ t
    return R_new, t_new


def convert_homogeneous_LH_RH(H, axis="z"):
    """
    H: (4,4) homogeneous transform (LH)
    returns H_RH
    """
    S = axis_reflection(axis)
    C = np.eye(4)
    C[:3, :3] = S
    return C @ H @ C  # since C^{-1}=C


def convert_pts8_to_line12(pts8: np.ndarray) -> np.ndarray:
    """
    Take the corner 8 points of a 3D bounding box and convert them to 12 line segments that draw the bounding box.
    Args:
        - pts8: (8,3) array of 8 points
    Returns:
        - lines12: (12,2,3) array of 12 line segments that draw the bounding box
    """
    assert isinstance(pts8, np.ndarray), "pts8 must be a numpy array"
    assert pts8.shape == (8, 3), "pts8 must have shape (8, 3)"

    # Strategy: Identify the two parallel faces of the bounding box
    # and create the proper cube wireframe connectivity

    # Find the center of all points
    center = np.mean(pts8, axis=0)

    # Find the principal axes by computing the covariance matrix
    centered_pts = pts8 - center
    cov_matrix = np.cov(centered_pts.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues (largest variation first)
    sort_idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, sort_idx]

    # Project points onto the principal axes
    projected = centered_pts @ eigenvecs

    # For each axis, split points into two groups (positive/negative side)
    # We'll use the axis with the smallest variation to split into two faces
    split_axis = 2  # axis with smallest variation (most likely the "height")

    # Split points based on the sign of projection on the split axis
    mask = projected[:, split_axis] > 0
    face1_indices = np.where(mask)[0]
    face2_indices = np.where(~mask)[0]

    # If we don't get 4 points on each face, use a different strategy
    if len(face1_indices) != 4 or len(face2_indices) != 4:
        # Fallback: split based on the coordinate with largest range
        ranges = np.ptp(pts8, axis=0)  # peak-to-peak (max - min) for each coordinate
        split_coord = np.argmin(ranges)  # coordinate with smallest range
        median_val = np.median(pts8[:, split_coord])
        mask = pts8[:, split_coord] > median_val
        face1_indices = np.where(mask)[0]
        face2_indices = np.where(~mask)[0]

    # Create edges within each face (4 edges per face = 8 edges total)
    def get_face_edges(face_indices):
        face_pts = pts8[face_indices]
        face_center = np.mean(face_pts, axis=0)

        # Sort points in clockwise/counterclockwise order around the face center
        # Project to 2D by removing the coordinate with smallest variation
        ranges = np.ptp(face_pts, axis=0)
        remove_coord = np.argmin(ranges)
        coords_2d = np.delete(face_pts - face_center, remove_coord, axis=1)

        # Sort by angle
        angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_face_indices = face_indices[sorted_indices]

        # Create edges between consecutive points (and close the loop)
        face_edges = []
        for i in range(4):
            face_edges.append(
                (sorted_face_indices[i], sorted_face_indices[(i + 1) % 4])
            )
        return face_edges

    # Get edges for both faces
    face1_edges = get_face_edges(face1_indices)
    face2_edges = get_face_edges(face2_indices)

    # Create vertical edges connecting corresponding vertices of the two faces
    # Match points based on their 2D positions after projecting out the split coordinate
    face1_pts = pts8[face1_indices]
    face2_pts = pts8[face2_indices]

    # Project both faces to 2D (remove the split coordinate)
    ranges = np.ptp(pts8, axis=0)
    split_coord = np.argmin(ranges)
    face1_2d = np.delete(face1_pts, split_coord, axis=1)
    face2_2d = np.delete(face2_pts, split_coord, axis=1)

    # Find closest point correspondences
    vertical_edges = []
    for i, pt1_2d in enumerate(face1_2d):
        distances = np.sum((face2_2d - pt1_2d) ** 2, axis=1)
        closest_j = np.argmin(distances)
        vertical_edges.append((face1_indices[i], face2_indices[closest_j]))

    # Combine all edges
    all_edges = face1_edges + face2_edges + vertical_edges
    lines12 = np.stack([pts8[[i, j], :] for (i, j) in all_edges], axis=0)
    return lines12


def process_video_data(
    video_data: list[dict],
    sample_id: str,
    export_base_path: Path = VIPE_EXPORT_DIR,
    force_overwrite: bool = True,
    export_all_frames: bool = False,
) -> dict:
    """
    Further process the per-frame video data to get the scene-level information.

    Args:
        - video_data: a list of frame data (dict) exported from export_video_data
    """
    N_frames = len(video_data)
    first_frame_y = video_data[0]["c2w"][:3, 1]

    ###########################################################################
    ### Process static background points
    ###########################################################################

    # Aggregate static points from all frames
    # bg_pcd, bg_rgb = get_video_static_bg_pcd(video_data, return_unreliable=False)
    ubg_pcd, ubg_rgb = (
        None,
        None,
    )
    # bg_pcd, bg_rgb, ubg_pcd, ubg_rgb = get_video_static_bg_pcd(video_data, return_unreliable=True)
    bg_pcd, bg_rgb = get_video_static_bg_pcd(video_data, return_unreliable=False)

    # convert bg_pcd to Open3D pcd
    bg_pcd_o3d = o3d.geometry.PointCloud()
    bg_pcd_o3d.points = o3d.utility.Vector3dVector(bg_pcd)
    bg_pcd_o3d.colors = o3d.utility.Vector3dVector(bg_rgb)

    # aabb: axis-aligned bounding box
    aabb = bg_pcd_o3d.get_axis_aligned_bounding_box()
    aabb_points = np.asarray(aabb.get_box_points())  # 8 corners of bbox
    aa_min = aabb_points.min(axis=0)
    aa_max = aabb_points.max(axis=0)
    print(f"aa_min: {aa_min}, aa_max: {aa_max}")

    # obb: oriented bounding box
    obb = bg_pcd_o3d.get_oriented_bounding_box()
    obb_points = np.asarray(obb.get_box_points())  # 8 corners of bbox

    # Use o3d to fit the ground plane
    # NOTE: it is more robust than my custom plane fitting
    plane_model, inliers = bg_pcd_o3d.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    [a, b, c, d] = plane_model
    # Note that this plane is defined in the original world frame before normalization
    # thus we don't use it at simulation
    # but we still need this step to compute the ground inliers points to derive a proper domain box
    print(f"Plane equation: [red]{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # convert inliers to a mask
    ground_inliers_mask = np.zeros(len(bg_pcd), dtype=bool)
    ground_inliers_mask[inliers] = True
    ground_outlier_mask = ~ground_inliers_mask
    print(
        f"Inliers mask: {ground_inliers_mask.shape}, Outlier mask: {ground_outlier_mask.shape}"
    )
    print(
        f"Inliers mask sum: {ground_inliers_mask.sum()}, Outlier mask sum: {ground_outlier_mask.sum()}"
    )
    # exit()
    # bg_rgb[outlier_mask,:] = [255, 0, 0]

    ground_pcd_o3d = bg_pcd_o3d.select_by_index(inliers)
    ground_aabb = ground_pcd_o3d.get_axis_aligned_bounding_box()
    ground_aabb_points = np.asarray(ground_aabb.get_box_points())  # 8 corners of bbox
    # ground_aa_min = ground_aabb_points.min(axis=0)
    # ground_aa_max = ground_aabb_points.max(axis=0)
    # print(f"ground_aa_min: {ground_aa_min}, ground_aa_max: {ground_aa_max}")

    ###########################################################################
    ### Process dynamic foreground points
    ###########################################################################
    # all_fq_pcd = [
    #     frame_dict["full_pcd"][frame_dict["fg_mask"]] for frame_dict in video_data
    # ]
    # all_fq_pcd = np.concatenate(all_fq_pcd, axis=0).reshape(-1, 3)

    all_fq_pcd = [
        frame_dict["full_pcd"][frame_dict["primary_obj_mask"]]
        for frame_dict in video_data
    ]
    if "multi" in sample_id:
        all_secondary_obj_pcd = [
            frame_dict["full_pcd"][frame_dict["secondary_obj_mask"]]
            for frame_dict in video_data
        ]
        all_fq_pcd = all_fq_pcd + all_secondary_obj_pcd

    all_fq_pcd = np.concatenate(all_fq_pcd, axis=0).reshape(-1, 3)

    # convert to Open3D pcd
    all_fq_pcd_o3d = o3d.geometry.PointCloud()
    all_fq_pcd_o3d.points = o3d.utility.Vector3dVector(all_fq_pcd)
    all_fq_aabb = all_fq_pcd_o3d.get_axis_aligned_bounding_box()
    all_fq_aabb_points = np.asarray(all_fq_aabb.get_box_points())  # 8 corners of bbox

    ###########################################################################
    ### Prepare for Physics simulation
    ###########################################################################
    # get all camera positions
    all_cam_pos = [frame_dict["c2w"][:3, 3] for frame_dict in video_data]
    all_cam_pos = np.stack(all_cam_pos, axis=0)  # (N_frames, 3)
    assert all_cam_pos.shape == (
        N_frames,
        3,
    ), f"Unexpected shape error: {all_cam_pos.shape} != ({N_frames}, 3)"

    # compute appropriate domain box for Physics simulation
    domain_defining_pts = np.concatenate(
        [all_cam_pos, all_fq_aabb_points, ground_aabb_points], axis=0
    )
    domain_pcd_o3d = o3d.geometry.PointCloud()
    domain_pcd_o3d.points = o3d.utility.Vector3dVector(domain_defining_pts)
    domain_aabb = domain_pcd_o3d.get_axis_aligned_bounding_box()
    domain_aabb_points = np.asarray(domain_aabb.get_box_points())  # 8 corners of bbox
    # Turn a cuboid domain box into a cube domain box
    domain_min = domain_aabb_points.min(axis=0)
    domain_max = domain_aabb_points.max(axis=0)
    # take the longest side as the side length for the cube
    domain_side_length = np.max(domain_max - domain_min)
    # use the old cuboid center as the center of the cube
    domain_center = (domain_min + domain_max) / 2
    domain_aabb_points = []
    # create 8 corners of the cube
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            for k in range(-1, 2, 2):
                domain_aabb_points.append(
                    domain_center + domain_side_length * 0.5 * np.array([i, j, k])
                )
    domain_aabb_points = np.array(domain_aabb_points)

    target_domain_size = 2.0
    target_domain_center = np.array([1.0, 1.0, 1.0])

    # get scale factor to scale the domain box to length of 2
    domain_scale = target_domain_size / domain_side_length
    # domain_translate = target_domain_center - domain_center
    # domain_rotate = np.array([
    #         [-1, 0, 0],
    #         [0, -1, 0],
    #         [0, 0, 1]]) # rot_180_round_z

    # NOTE: We identify the where the new basis are at in the current world frame
    # we can do this because the domain box is constructed such that
    # domain_aabb_points[6] is the near bottom right corner of the box, and
    # we make it the new origin.
    new_origin = domain_aabb_points[6]
    new_x = (domain_aabb_points[2] + new_origin) / 2
    new_y = (domain_aabb_points[4] + new_origin) / 2
    new_z = (domain_aabb_points[7] + new_origin) / 2

    transform_dict = build_transform(p0=new_origin, px=new_x, py=new_y, pz=new_z)

    # Sanity Check
    # check derived scale is expected
    _scale = 1 / transform_dict["s"]
    if np.abs(_scale - domain_scale) > 1e-6:
        print(
            f"[red bold] WARNING: [/red bold]Derived scale {_scale} is not expected {domain_scale}"
        )

    ###########################################################################
    ### Apply domain transformation to the scene data
    ###########################################################################
    # bg_pcd = transform_world(bg_pcd, domain_center, target_domain_center, domain_rotate, domain_scale)
    # all_fq_aabb_points = transform_world(all_fq_aabb_points, domain_center, target_domain_center, domain_rotate, domain_scale)
    # domain_aabb_points = transform_world(domain_aabb_points, domain_center, target_domain_center, domain_rotate, domain_scale)
    # ground_aabb_points = transform_world(ground_aabb_points, domain_center, target_domain_center, domain_rotate, domain_scale)

    # for i in range(N_frames):
    #     video_data[i]["full_pcd"] = transform_world(video_data[i]["full_pcd"], domain_center, target_domain_center, domain_rotate, domain_scale)
    #     video_data[i]["ball_centroid"] = transform_world(video_data[i]["ball_centroid"], domain_center, target_domain_center, domain_rotate, domain_scale)
    #     video_data[i]["ball_radius"] = video_data[i]["ball_radius"] * domain_scale
    #     video_data[i]["ball_displacement"] = video_data[i]["ball_displacement"] * domain_scale

    bg_pcd = transform_points(bg_pcd, transform_dict["M"])
    if ubg_pcd is not None:
        ubg_pcd = transform_points(ubg_pcd, transform_dict["M"])
    ground_aabb_points = transform_points(ground_aabb_points, transform_dict["M"])
    all_fq_aabb_points = transform_points(all_fq_aabb_points, transform_dict["M"])
    domain_aabb_points = transform_points(domain_aabb_points, transform_dict["M"])
    all_cam_pos = transform_points(all_cam_pos, transform_dict["M"])

    for i in range(N_frames):
        video_data[i]["c2w"] = transform_pose_c2w_hom(
            video_data[i]["c2w"], transform_dict["M"], transform_dict["R"]
        )
        video_data[i]["full_pcd"] = transform_points(
            video_data[i]["full_pcd"], transform_dict["M"]
        )
        # video_data[i]["ball_centroid"] = transform_points(
        #     video_data[i]["ball_centroid"], transform_dict["M"]
        # )
        video_data[i]["primary_obj_centroid"] = transform_points(
            video_data[i]["primary_obj_centroid"], transform_dict["M"]
        )
        video_data[i]["secondary_obj_centroid"] = transform_points(
            video_data[i]["secondary_obj_centroid"], transform_dict["M"]
        )
        # video_data[i]["ball_radius"] = (
        #     video_data[i]["ball_radius"] / transform_dict["s"]
        # )
        video_data[i]["primary_obj_radius"] = (
            video_data[i]["primary_obj_radius"] / transform_dict["s"]
        )
        video_data[i]["secondary_obj_radius"] = (
            video_data[i]["secondary_obj_radius"] / transform_dict["s"]
        )
        # compute the displacement vector after transformation
        # all displacement vectors are relative to the first frame
        # video_data[i]["ball_displacement"] = (
        #     video_data[i]["ball_centroid"] - video_data[0]["ball_centroid"]
        # )
        video_data[i]["primary_obj_displacement"] = (
            video_data[i]["primary_obj_centroid"]
            - video_data[0]["primary_obj_centroid"]
        )

    # all_cam_ball_dist = [
    #     float(np.linalg.norm(all_cam_pos[i] - video_data[i]["ball_centroid"]))
    #     for i in range(N_frames)
    # ]
    all_cam_obj_dist = [
        float(np.linalg.norm(all_cam_pos[i] - video_data[i]["primary_obj_centroid"]))
        for i in range(N_frames)
    ]

    ###########################################################################
    ### Recalculate plane model in the new world frame
    ###########################################################################
    bg_pcd_o3d = o3d.geometry.PointCloud()
    bg_pcd_o3d.points = o3d.utility.Vector3dVector(bg_pcd)
    plane_model, inliers = bg_pcd_o3d.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    [a, b, c, d] = plane_model
    print(f"Updated Plane equation: [red]{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    plane_normal = np.array([a, b, c])
    _norm = np.linalg.norm(plane_normal)
    # we assume gravity is perpendicular downward to the plane
    gravity_direction = -plane_normal / _norm if _norm != 0 else np.array([0, -1, 0])
    # we assume gravity is 9.81 m/s^2
    # we scale it by the scale factor of the domain box
    gravity_vector = (9.81 / transform_dict["s"]) * gravity_direction

    def eval_for_y(plane_model, x, z):
        [a, b, c, d] = plane_model
        return -(a * x + c * z + d) / b

    plane_vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 2], [2, 0, 2]], dtype=float)
    for i in range(4):
        plane_vertices[i, 1] = eval_for_y(
            plane_model, x=plane_vertices[i, 0], z=plane_vertices[i, 2]
        )
    plane_faces = np.array([[2, 1, 0], [2, 3, 1]])

    first_primary_obj_radius = video_data[0].get("primary_obj_radius", 0.0)
    first_secondary_obj_radius = video_data[0].get("secondary_obj_radius", 0.0)

    all_c2w = []
    all_intrinsics = []
    all_primary_obj_centroid = []
    all_primary_obj_displacement = []
    for i in range(N_frames):
        all_c2w.append(video_data[i]["c2w"].astype(np.float32))
        all_intrinsics.append(video_data[i]["K"].astype(np.float32))
        all_primary_obj_centroid.append(
            video_data[i]["primary_obj_centroid"].astype(np.float32)
        )
        all_primary_obj_displacement.append(
            video_data[i]["primary_obj_displacement"].astype(np.float32)
        )

    all_c2w = np.stack(all_c2w, axis=0)  # (N_frames, 4, 4) float32
    all_intrinsics = np.stack(all_intrinsics, axis=0)  # (N_frames, 4) float32
    all_primary_obj_centroid = np.stack(
        all_primary_obj_centroid, axis=0
    )  # (N_frames, 3) float32
    all_primary_obj_displacement = np.stack(
        all_primary_obj_displacement, axis=0
    )  # (N_frames, 3) float32
    # arrgh(all_c2w, all_intrinsics, all_primary_obj_centroid, all_primary_obj_displacement)
    # exit()
    primary_obj_name = video_data[0]["primary_obj_name"]
    secondary_obj_name = video_data[0]["secondary_obj_name"]

    video_info = dict(
        sample_id=sample_id,
        N_frames=N_frames,
        plane_model=plane_model,  # [a, b, c, d]
        plane_normal=plane_normal,  # [a, b, c]
        plane_vertices=plane_vertices,  # 4 corner vertices on plane
        plane_faces=plane_faces,  # 2 simplest triangle to represent the plane
        all_cam_pos=all_cam_pos,
        all_cam_obj_dist=all_cam_obj_dist,
        bg_pcd=bg_pcd,
        bg_rgb=bg_rgb,
        ubg_pcd=ubg_pcd,
        ubg_rgb=ubg_rgb,
        ground_inliers_mask=ground_inliers_mask,
        ground_outlier_mask=ground_outlier_mask,
        ground_aabb_points=ground_aabb_points,
        all_fq_aabb_points=all_fq_aabb_points,
        domain_aabb_points=domain_aabb_points,
        first_primary_obj_radius=first_primary_obj_radius,
        first_secondary_obj_radius=first_secondary_obj_radius,
        primary_obj_name=primary_obj_name,
        secondary_obj_name=secondary_obj_name,
        domain_center=domain_center,  # (3,): original domain center before normalization
        domain_length=domain_side_length,  # float: domain side length before normalization
        domain_scale=domain_scale,  # float: scale factor to scale the domain box to length of 2
        gravity_direction=gravity_direction,  # unit vector pointing to the direction of gravity
        gravity_vector=gravity_vector,  # scaled gravity vector in the world frame for the simulator
        all_c2w=all_c2w,
        all_intrinsics=all_intrinsics,
        all_primary_obj_centroid=all_primary_obj_centroid,
        all_primary_obj_displacement=all_primary_obj_displacement,
    )

    video_export_dir = export_base_path / sample_id
    save_data_dict(
        video_info,
        "video_info",
        video_export_dir,
        overwrite=force_overwrite,
        use_grid_io=True,
        numpy_size_threshold=64,
    )
    # key_frames_idx = [x for x in [0, 5, 10, 15, 20] if x < N_frames]
    if export_all_frames:
        key_frames_idx = range(N_frames)
    else:
        key_frames_idx = [0, 5]

    frames_export_dir = video_export_dir / "frames_info"
    frames_export_dir.mkdir(parents=True, exist_ok=True)
    for idx in key_frames_idx:
        frame_dict = video_data[idx]
        save_data_dict(
            frame_dict,
            f"{idx:05d}",
            frames_export_dir,
            overwrite=force_overwrite,
            use_grid_io=True,
            numpy_size_threshold=64,
        )

    success_file = video_export_dir / "success.txt"
    success_file.touch()

    print("=" * 100)
    print(f"[green bold]{sample_id}[/green bold] Vipe result exported.")
    print("=" * 100)
    return video_info


###############################################################################
### Velocity Field Helpers
###############################################################################
def make_arrow_mesh(
    *,
    n_segments: int = 16,
    shaft_length: float = 0.7,
    head_length: float = 0.3,
    shaft_radius: float = 0.02,
    head_radius: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a unit arrow mesh aligned along +Z with base at z=0 and tip at z=1.
    Returns (vertices [V,3] float32, faces [F,3] int32).
    """
    # Cylinder (shaft): z in [0, shaft_length]
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    c = np.cos(theta)
    s = np.sin(theta)

    # bottom and top rings
    shaft_bottom = np.stack(
        [shaft_radius * c, shaft_radius * s, np.zeros_like(c)], axis=1
    )
    shaft_top = np.stack(
        [shaft_radius * c, shaft_radius * s, np.full_like(c, shaft_length)], axis=1
    )

    # side faces for cylinder (no caps)
    verts = np.concatenate([shaft_bottom, shaft_top], axis=0)  # [2*M,3]
    faces = []
    M = n_segments
    for i in range(M):
        j = (i + 1) % M
        # two triangles per quad
        faces.append([i, j, M + i])
        faces.append([M + i, j, M + j])

    # Cone (head): base at z=shaft_length, tip at z=shaft_length+head_length (==1)
    cone_base = np.stack(
        [head_radius * c, head_radius * s, np.full_like(c, shaft_length)], axis=1
    )
    tip = np.array([[0.0, 0.0, shaft_length + head_length]], dtype=np.float32)

    cone_base_idx_offset = verts.shape[0]
    verts = np.concatenate([verts, cone_base, tip], axis=0)
    tip_idx = verts.shape[0] - 1

    for i in range(M):
        j = (i + 1) % M
        faces.append([cone_base_idx_offset + i, cone_base_idx_offset + j, tip_idx])

    return verts.astype(np.float32), np.asarray(faces, dtype=np.int32)


# ---------- math helpers ----------
def _z_to_vec_quaternions(vectors: np.ndarray) -> np.ndarray:
    """
    Compute quaternions (wxyz) that rotate +Z to each vector direction.
    vectors: (N,3). Returns (N,4) float32.
    Handles the zero vector and the exact opposite-of-Z case robustly.
    """
    v = vectors.astype(np.float32)
    eps = 1e-9
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # normalized directions; for zero-length, keep 0
    dirs = np.divide(v, np.maximum(norms, eps), where=norms > 0, out=np.zeros_like(v))
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)[None, :]

    dots = (dirs * z).sum(axis=1)  # cos(angle)
    crosses = np.cross(z, dirs)  # rotation axes (unnormalized)
    cross_norms = np.linalg.norm(crosses, axis=1, keepdims=True)

    # If cross norm is tiny:
    # - if dot ~ +1 => same direction -> identity
    # - if dot ~ -1 => opposite => rotate pi around X (or any axis ⟂ z)
    axes = np.where(
        cross_norms > 1e-7,
        crosses / np.maximum(cross_norms, eps),
        np.array([1.0, 0.0, 0.0], dtype=np.float32)[None, :],  # fallback axis
    )

    angles = np.arctan2(cross_norms.squeeze(-1), np.clip(dots, -1.0, 1.0))  # shape (N,)
    half = 0.5 * angles
    sin_half = np.sin(half)[:, None]
    cos_half = np.cos(half)[:, None]

    # Identity for zero-magnitude vectors
    zero_mask = norms.squeeze(-1) <= eps
    wxyz = np.concatenate([cos_half, axes * sin_half], axis=1)  # (N,4)

    # Fix exact opposite direction (dot ~ -1 & cross ~ 0): set to 180° about X => [0,1,0,0]
    opp_mask = (np.abs(cross_norms.squeeze(-1)) <= 1e-7) & (dots < -0.999999)
    if np.any(opp_mask):
        wxyz[opp_mask] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    if np.any(zero_mask):
        wxyz[zero_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    return wxyz.astype(np.float32)


def _direction_to_rgb(dirs: np.ndarray) -> np.ndarray:
    """
    Map normalized directions in [-1,1] to RGB in [0,255]:
    R <- X, G <- Y, B <- Z (shifted/scaled to 0..1).
    For zero-norm vectors, use mid-grey.
    """
    eps = 1e-9
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    unit = np.divide(
        dirs, np.maximum(norms, eps), where=norms > 0, out=np.zeros_like(dirs)
    )
    rgb = unit * 0.5 + 0.5  # 0..1
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    zero_mask = norms.squeeze(-1) <= eps
    if np.any(zero_mask):
        rgb[zero_mask] = np.array([128, 128, 128], dtype=np.uint8)
    return rgb


def make_instances(pts, vels, k: int, gscale: float):
    idx = np.arange(0, pts.shape[0], max(1, k), dtype=np.int64)
    sel_pts = pts[idx]
    sel_vels = vels[idx]
    scales = np.linalg.norm(sel_vels, axis=1).astype(np.float32) * gscale
    # Quaternions orienting +Z to velocity
    wxyzs = _z_to_vec_quaternions(sel_vels)
    # Colors by direction
    colors = _direction_to_rgb(sel_vels)
    return sel_pts.astype(np.float32), wxyzs, scales.astype(np.float32), colors


###############################################################################
### Velocity Computation Helpers
###############################################################################
def velocities_from_rotation(points, center, axis, speed, units="rad/s"):
    """
    Compute instantaneous vertex velocities for a rigid body rotating about a center.

    Parameters
    ----------
    points : (N, 3) array_like
        World-space vertex positions.
    center : (3,) array_like
        World-space rotation center C (point on the axis).
    axis : (3,) array_like
        Rotation axis direction. Need not be unit; will be normalized.
        Right-hand rule defines positive rotation direction.
    speed : float
        Magnitude of rotational speed (>=0). Interpreted per `units`.
    units : {"rad/s", "deg/s", "rpm"}, optional
        Units of `speed`. Default "rad/s".

    Returns
    -------
    velocities : (N, 3) ndarray
        Instantaneous linear velocity vectors at each point.
    speeds : (N,) ndarray
        Speed magnitudes at each point.

    Notes
    -----
    - If you also have a body translational velocity v0, add it to `velocities`.
    - Points lying exactly on the axis have zero velocity.
    """
    P = np.asarray(points, dtype=float)
    C = np.asarray(center, dtype=float).reshape(3)
    a = np.asarray(axis, dtype=float).reshape(3)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("`points` must be an (N, 3) array.")
    if (
        not np.all(np.isfinite(P))
        or not np.all(np.isfinite(C))
        or not np.all(np.isfinite(a))
    ):
        raise ValueError("Inputs must be finite numbers.")
    na = np.linalg.norm(a)
    if na == 0:
        raise ValueError("Axis vector must be non-zero.")
    a_hat = a / na

    # Convert speed to radians per second
    if units == "rad/s":
        S = float(speed)
    elif units == "deg/s":
        S = np.deg2rad(float(speed))
    elif units == "rpm":
        S = float(speed) * 2.0 * np.pi / 60.0
    else:
        raise ValueError("units must be one of {'rad/s','deg/s','rpm'}")

    # Angular velocity vector
    omega = S * a_hat  # shape (3,)

    # Position vectors relative to center
    r = P - C  # shape (N, 3)

    # Velocity via ω × r
    velocities = np.cross(omega, r)  # shape (N, 3)

    # Speed magnitudes (||v||). Equivalent to S * distance-to-axis.
    # speeds = np.linalg.norm(velocities, axis=1)

    return velocities


###############################################################################
### Main Function
###############################################################################


def main(
    base_path: Path = VIPE_RAW_DIR,
    export_base_path: Path = VIPE_EXPORT_DIR,
    sample_id: str = None,
    launch_viser: bool = False,
    host: str = "0.0.0.0",
    port: int = 20541,
    dark_mode: bool = True,
) -> None:
    # Get list of artifacts.
    artifacts: list[ArtifactPath] = list(
        ArtifactPath.glob_artifacts(base_path, use_video=True)
    )
    if len(artifacts) == 0:
        print(
            f"No artifacts found in {base_path}. You need to run vipe infer first or provide the correct base path."
        )
        exit(1)

    artifact_names = [x.artifact_name for x in artifacts]
    if sample_id is None:
        sample_id = sorted(artifact_names)[0]

    if sample_id not in artifact_names:
        print(
            f"Sample ID {sample_id} not found in all available sample IDs: {artifact_names}."
        )
        exit(1)

    target_artifact = artifacts[artifact_names.index(sample_id)]

    temporal_subsample = spatial_subsample = 1

    video_data = export_video_data(
        target_artifact,
        temporal_subsample=temporal_subsample,
        spatial_subsample=spatial_subsample,
    )
    video_info = process_video_data(
        video_data=video_data,
        sample_id=sample_id,
        export_base_path=export_base_path,
        force_overwrite=not launch_viser,
        export_all_frames=False,
    )

    ###########################################################################
    ### Visualization
    ###########################################################################
    if launch_viser:
        print("\n\nStarting Viser server for visualization...")
        server = viser.ViserServer(host=host, port=port, verbose=False)
        # Enable dark mode
        server.gui.configure_theme(dark_mode=dark_mode)
        # Make world-frame origin and axes visible
        server.scene.world_axes.visible = True
        POINT_SIZE = 0.001

        ### Configure UP direction for 3D viewer
        # up_direction = -video_info["first_frame_y"]
        up_direction = np.array([0, 1, 0])
        server.scene.set_up_direction(up_direction)
        print(
            f"[green bold]Set Viser scene up direction to[/green bold] {up_direction}"
        )

        server.gui.add_text("sample_id", initial_value=sample_id, disabled=True)

        with server.gui.add_folder("Appearance"):
            dark_mode_toggle = server.gui.add_checkbox(
                "Dark Mode", initial_value=dark_mode
            )
            shadow_toggle = server.gui.add_checkbox("Cast Shadow", initial_value=False)

        with server.gui.add_folder("Ground Plane"):
            ground_plane_color = server.gui.add_rgb(
                "Color", initial_value=(220, 220, 220)
            )
            ground_plane_opacity = server.gui.add_slider(
                "Opacity", min=0.0, max=1.0, step=0.1, initial_value=0.5
            )

        @dark_mode_toggle.on_update
        def _(_: object) -> None:
            server.gui.configure_theme(dark_mode=dark_mode_toggle.value)

        ### Configure grid plane
        # Add grid
        # We draw this gird as the "floor" of the simulation domain [0, 2] in (x, y, z)
        server.scene.add_grid(
            "/grid",
            width=2.0,
            height=2.0,
            plane="xz",
            section_color=[240, 240, 240],
            position=np.array([1.0, 0.0, 1.0]),
            wxyz=tf.SO3.identity().wxyz,
            visible=True,
        )
        ### Add ground plane
        ground_plane_handle = server.scene.add_mesh_simple(
            "/ground_plane",
            vertices=video_info["plane_vertices"],
            faces=video_info["plane_faces"],
            wxyz=tf.SO3.identity().wxyz,
            position=(0.0, 0.0, 0.0),
            color=ground_plane_color.value,
            opacity=float(ground_plane_opacity.value),
            visible=True,
            side="double",
            cast_shadow=shadow_toggle.value,
        )

        @ground_plane_color.on_update
        def _(_: object) -> None:
            ground_plane_handle.color = ground_plane_color.value

        @ground_plane_opacity.on_update
        def _(_: object) -> None:
            ground_plane_handle.opacity = float(ground_plane_opacity.value)

        server.scene.add_point_cloud(
            name="/pcd/static_background/ground_plane",
            points=video_info["bg_pcd"][video_info["ground_inliers_mask"]],
            colors=video_info["bg_rgb"][video_info["ground_inliers_mask"]],
            point_size=POINT_SIZE,
            point_shape="rounded",
            visible=True,
        )
        server.scene.add_point_cloud(
            name="/pcd/static_background/outliers",
            points=video_info["bg_pcd"][video_info["ground_outlier_mask"]],
            colors=video_info["bg_rgb"][video_info["ground_outlier_mask"]],
            point_size=POINT_SIZE,
            point_shape="rounded",
            visible=True,
        )
        if "ubg_pcd" in video_info:
            server.scene.add_point_cloud(
                name="/pcd/static_background/unreliable",
                points=video_info["ubg_pcd"],
                colors=video_info["ubg_rgb"],
                point_size=POINT_SIZE,
                point_shape="rounded",
                visible=True,
            )

        rainbow_colors = get_rainbow_color(video_info["N_frames"])
        mean_ball_radius = video_info.get("first_primary_obj_radius")

        for idx, frame_dict in enumerate(video_data):
            # Draw per-frame camera frustum
            server.scene.add_camera_frustum(
                f"/cameras/{idx}",
                fov=frame_dict["fov"],
                aspect=frame_dict["frame_aspect_ratio"],
                scale=0.05,
                image=frame_dict["thumbnail_rgb"],
                wxyz=tf.SO3.from_matrix(frame_dict["c2w"][:3, :3]).wxyz,
                position=frame_dict["c2w"][:3, 3],
                color=rainbow_colors[idx],
                visible=True,
            )

            # # Draw per-frame foreground points
            # fg_pcd = frame_dict["full_pcd"][frame_dict["fg_mask"]]
            # fg_rgb = frame_dict["full_rgb"][frame_dict["fg_mask"]]
            # _fg_handle = server.scene.add_point_cloud(
            #     f"/pcd/dynamic_foreground/{idx}",
            #     points=fg_pcd,
            #     colors=fg_rgb,
            #     point_size=POINT_SIZE,
            #     point_shape="rounded",
            #     visible=False,
            # )

            # Draw per-frame primary object points
            primary_obj_name = frame_dict["primary_obj_name"]
            primary_obj_pcd = frame_dict["full_pcd"][frame_dict["primary_obj_mask"]]
            primary_obj_rgb = frame_dict["full_rgb"][frame_dict["primary_obj_mask"]]
            _primary_obj_handle = server.scene.add_point_cloud(
                f"/pcd/dynamic/{primary_obj_name}/{idx}",
                points=primary_obj_pcd,
                colors=primary_obj_rgb,
                point_size=POINT_SIZE,
                point_shape="rounded",
                visible=True,
            )
            # _primary_obj_handle.visible = True if idx in (0, 41) else False

            # Draw per-frame secondary object points
            if "multi" in sample_id:
                secondary_obj_name = frame_dict["secondary_obj_name"]
                secondary_obj_pcd = frame_dict["full_pcd"][
                    frame_dict["secondary_obj_mask"]
                ]
                secondary_obj_rgb = frame_dict["full_rgb"][
                    frame_dict["secondary_obj_mask"]
                ]
                _secondary_obj_handle = server.scene.add_point_cloud(
                    f"/pcd/dynamic/{secondary_obj_name}/{idx}",
                    points=secondary_obj_pcd,
                    colors=secondary_obj_rgb,
                    point_size=POINT_SIZE,
                    point_shape="rounded",
                    visible=True,
                )
            else:
                _secondary_obj_handle = None

            # Draw a line segment from camera position to ball centroid
            cam_pos = video_info["all_cam_pos"][idx]
            # ball_pos = frame_dict["ball_centroid"]
            ball_pos = frame_dict["primary_obj_centroid"]
            _line_handle = server.scene.add_line_segments(
                f"/lines/cam_to_ball/{idx}",
                points=[[cam_pos, ball_pos]],
                colors=rainbow_colors[idx],
                line_width=3.0,
                visible=True,
            )
            # _line_handle.visible = True if idx in (0, 41) else False

            # Draw a virtual sphere to represent the foreground ball
            if mean_ball_radius:
                _ball_handle = server.scene.add_icosphere(
                    f"/ball/{idx}",
                    radius=mean_ball_radius,
                    position=ball_pos,
                    color=rainbow_colors[idx],
                    opacity=0.5,
                    cast_shadow=False,
                    visible=True,
                )
                # _ball_handle.visible = True if idx in (0, 41) else False

        if sample_id == "2_basketball":
            ball_mesh_path = (
                OUT_PERCEPTION_DIR / sample_id / "00000" / "meshes" / "basketball.obj"
            )
        else:
            ball_mesh_path = Path("NA")

        if ball_mesh_path.exists():
            # ball_mesh = trimesh.load(ball_mesh_path)
            ball_mesh = normalize_mesh(ball_mesh_path, side=mean_ball_radius * 2.05)
            # ball_pos = video_data[0]["ball_centroid"]
            ball_pos = video_data[0]["primary_obj_centroid"]
            ball_rot = [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ]
            ball_rot = np.array(ball_rot, dtype=np.float32)
            ball_RT = np.eye(4, dtype=np.float32)
            ball_RT[:3, :3] = ball_rot
            ball_RT[:3, 3] = ball_pos

            ball_mesh.vertices = transform_pcd(ball_mesh.vertices, ball_RT)
            ball_mesh_out = ball_mesh_path.parent / "ball1_placed.obj"
            ball_mesh.export(ball_mesh_out)

            # _ball_handle = server.scene.add_mesh_trimesh(
            #     "/trimesh/00000",
            #     mesh=ball_mesh,
            #     wxyz=tf.SO3.from_matrix(ball_rot).wxyz,
            #     position=ball_pos,
            #     visible=True,
            # )
            _ball_handle = server.scene.add_mesh_trimesh(
                "/trimesh/00000",
                mesh=ball_mesh,
                wxyz=tf.SO3.identity().wxyz,
                position=np.zeros(3, dtype=np.float32),
                visible=True,
            )

        # Define local helper to draw bounding boxes
        def draw_bbox(
            server,
            name,
            bbox_lines12=None,
            bbox_points=None,
            color=[255, 255, 255],
            visible=True,
            long_label=False,
            show_label=False,
        ):
            if bbox_lines12 is None and bbox_points is not None:
                if isinstance(bbox_points, list):
                    bbox_points = np.array(bbox_points)
                assert isinstance(
                    bbox_points, np.ndarray
                ), "bbox_points must be a numpy array"
                assert bbox_points.shape == (8, 3), "bbox_points must have shape (8, 3)"
                bbox_lines12 = convert_pts8_to_line12(bbox_points)

            server.scene.add_line_segments(
                f"/bbox/{name}/lines",
                points=bbox_lines12,
                colors=np.array([[[color[0], color[1], color[2]]] * 2] * 12),
                line_width=3.0,
                visible=visible,
            )
            if bbox_points is not None:
                for i, pt in enumerate(bbox_points):
                    label = (
                        f"{i} ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})"
                        if long_label
                        else f"{i}"
                    )
                    server.scene.add_label(
                        f"/bbox/{name}/pts/{i}",
                        text=label,
                        position=pt,
                        wxyz=tf.SO3.identity().wxyz,
                        visible=show_label,
                    )

        # Draw bounding boxes
        draw_bbox(
            server,
            "ground_aabb",
            bbox_points=video_info["ground_aabb_points"],
            color=[255, 0, 0],  # red
            visible=True,
        )
        draw_bbox(
            server,
            "all_fq_aabb",
            bbox_points=video_info["all_fq_aabb_points"],
            color=[0, 255, 0],  # green
            visible=True,
        )
        draw_bbox(
            server,
            "domain_aabb",
            bbox_points=video_info["domain_aabb_points"],
            color=[0, 0, 255],  # blue
            visible=True,
            show_label=False,
            long_label=True,
        )

        time.sleep(3.0)
        print(f"Open your browser to http://localhost:{port}")
        print("Press Ctrl+C to exit\n")

        # Auto shutdown server after K minutes
        K = 30
        max_time_alive = 60.0 * K
        while True:
            time.sleep(10.0)
            max_time_alive -= 10.0
            if max_time_alive <= 0:
                print(f"Automatic server shutdown after {K} minutes. Exiting...")
                exit(0)

    return None


if __name__ == "__main__":
    ...
