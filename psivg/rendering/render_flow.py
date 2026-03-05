import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import cm
from PIL import Image
from psivg.constants import OUT_RENDERING_DIR, OUT_SIMULATION_DIR, VIPE_EXPORT_DIR
from rich import print

from .grid_io import load_grid
from .interpolate import interpolate_c2ws
from .make_video import make_video_from_frames_dir
from .particle_io import ParticleIO
from .projection import project


def backup_this_script(output_dir: Path, timestamp: str = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = Path(__file__)
    script_content = script_path.read_text()
    with open(output_dir / f"flow_{timestamp}.py", "w") as f:
        f.write(script_content)


def _read_json(path: Path) -> dict:
    """Read and parse a JSON file at the given path."""
    with open(path, "r") as f:
        return json.load(f)


def _load_video_info(exported_sample_dir: Path) -> dict:
    """Load the `video_info.json` for the given video directory."""
    video_info_dir = exported_sample_dir / "video_info"
    video_info_json = video_info_dir / "video_info.json"
    if not video_info_json.exists():
        raise FileNotFoundError(f"Missing video_info.json at {video_info_json}")
    return _read_json(video_info_json)


def _list_frame_dirs(exported_sample_dir: Path) -> List[Path]:
    """List and return all `frame_*` subdirectories under `video_dir`."""
    frame_dirs: List[Path] = []
    frames_info_dir = exported_sample_dir / "frames_info"
    for path in sorted(frames_info_dir.glob("*")):
        if path.is_dir():
            frame_dirs.append(path)
    frame_dirs = sorted(frame_dirs, key=lambda x: int(x.name))
    return frame_dirs


def _load_frame_infos(frame_dirs: List[Path]) -> list:
    frame_infos = []
    for frame_dir in frame_dirs:
        frame_meta = _read_json(frame_dir / f"{frame_dir.name}.json")
        frame_infos.append(frame_meta)
    return frame_infos


def load_cam_base_params(frame_meta: dict) -> dict:
    fov_deg = float(frame_meta["fov"]) * 180.0 / math.pi
    K = frame_meta["K"]
    fx, fy, cx, cy = K
    width, height = int(cx * 2), int(cy * 2)
    # Original data is RH, Mitsuba is also RH
    # But original Up is -Y, Mitsuba Up is +Y
    # So we need to rotate 180 degrees around Z axis
    C = np.eye(4, dtype=np.float32)
    C[0, 0] = -1.0
    C[1, 1] = -1.0
    return dict(fov_deg=fov_deg, width=width, height=height, C=C)


def build_intrinsics(K: list[float]) -> np.ndarray:
    fx, fy, cx, cy = K
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    return intrinsics


def get_rainbow_color(N: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(int(N)):
        color = cm.jet(i / N)[:3]  # r,g,b in [0,1]
        color = tuple((int(c * 255) for c in color))  # r,g,b in [0,255]
        colors.append(color)
    return colors


def get_unique_colors(N: int) -> list[tuple[int, int, int]]:
    """
    Generate N unique colors as long as N is less than 256^3 - 2
    starting from black (0, 0, 0) to white (255, 255, 254)
    """
    assert N <= 256**3 - 2, f"N must be less than 256^3 - 2, got {N}"

    colors = list(range(1, N + 1))
    colors = [(x // 256**2, (x // 256) % 256, x % 256) for x in colors]
    return colors


def _to_uint8_colors(arr: np.ndarray) -> np.ndarray:
    """
    Accept colors in [0,1] float or [0,255] float/int and return uint8.
    """
    arr = np.asarray(arr)
    if arr.dtype.kind == "f" and np.nanmax(arr) <= 1.0:
        arr = np.clip(arr * 255.0, 0, 255)
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def patchify(image: np.ndarray, patch_size: int = 5) -> np.ndarray:
    """
    Patchify an image into a grid of patches, handling arbitrary image sizes.
    Edge patches that are smaller than patch_size are padded with zeros.

    Args:
        image: np.ndarray
            Image to patchify, shape: (H, W) or (H, W, C).
        patch_size: int
            Size of the patches.
    Returns:
        patches: np.ndarray
            Patches, shape: (N_patches, patch_size, patch_size) or (N_patches, patch_size, patch_size, C).
            All patches have consistent size by zero-padding edge patches if needed.
    """
    H, W = image.shape[:2]
    patches = []

    # Determine if we have channels
    has_channels = len(image.shape) == 3
    C = image.shape[2] if has_channels else None

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            # Extract the patch
            patch = image[i : i + patch_size, j : j + patch_size]

            # Get actual patch dimensions
            patch_h, patch_w = patch.shape[:2]

            # If patch is smaller than patch_size, pad with zeros
            if patch_h < patch_size or patch_w < patch_size:
                if has_channels:
                    padded_patch = np.zeros(
                        (patch_size, patch_size, C), dtype=image.dtype
                    )
                    padded_patch[:patch_h, :patch_w, :] = patch
                else:
                    padded_patch = np.zeros((patch_size, patch_size), dtype=image.dtype)
                    padded_patch[:patch_h, :patch_w] = patch
                patches.append(padded_patch)
            else:
                patches.append(patch)

    return np.array(patches)


def unpatchify(patches: np.ndarray, H: int, W: int, patch_size: int = 5) -> np.ndarray:
    """
    Unpatchify a grid of patches into an image, handling arbitrary image sizes.
    Only extracts the valid portion of each patch, ignoring zero-padding.

    Args:
        patches: np.ndarray
            Patches, shape: (N_patches, patch_size, patch_size) or (N_patches, patch_size, patch_size, C).
        H: int
            Target image height.
        W: int
            Target image width.
        patch_size: int
            Size of the patches.
    Returns:
        image: np.ndarray
            Image, shape: (H, W) or (H, W, C).
    """
    # Determine the number of channels from the patches
    has_channels = len(patches.shape) == 4
    if has_channels:
        C = patches.shape[3]
        image = np.zeros((H, W, C), dtype=patches.dtype)
    else:
        # Handle grayscale case
        image = np.zeros((H, W), dtype=patches.dtype)

    # Calculate number of patches in each dimension
    n_patches_h = (H + patch_size - 1) // patch_size  # Ceiling division
    n_patches_w = (W + patch_size - 1) // patch_size  # Ceiling division

    patch_idx = 0
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            if patch_idx >= len(patches):
                break

            # Get the actual region dimensions in the target image
            target_h = min(patch_size, H - i)
            target_w = min(patch_size, W - j)

            patch = patches[patch_idx]

            # Copy only the valid portion of the patch (excluding any padding)
            if has_channels:
                image[i : i + target_h, j : j + target_w, :] = patch[
                    :target_h, :target_w, :
                ]
            else:
                image[i : i + target_h, j : j + target_w] = patch[:target_h, :target_w]

            patch_idx += 1

    return image


def dynamic_two_clusters(
    x,
    gap_ratio: float = 1.5,
    min_frac: float = 0.05,
    return_info: bool = False,
):
    """
    Split 1D data into two clusters using either a clear largest gap or the median.

    Parameters
    ----------
    x : array-like of shape (N,)
        1D data (e.g., in [0, 1]).
    gap_ratio : float, default=5.0
        How much larger the largest gap must be relative to the typical (median) gap
        to be considered a "clear" gap.
    min_frac : float, default=0.05
        Both clusters must have at least this fraction of points when using the gap rule.
        Otherwise we fall back to the median split.
    return_info : bool, default=True
        If True, also return a dict with details about the split.

    Returns
    -------
    labels : ndarray of shape (N,)
        0 = bigger cluster, 1 = smaller cluster (by number of points).
    threshold : float
        The threshold value used for splitting.
    info : dict (if return_info)
        Keys: rule ("max_gap" or "median"), gap, med_gap, left_size, right_size
    """
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        raise ValueError("Empty input after removing NaNs.")
    if n == 1:
        # Only one point: everything is one cluster; use that value as threshold.
        labels = np.array([0], dtype=int)
        return (
            (
                labels,
                float(x[0]),
                {
                    "rule": "degenerate",
                    "gap": 0.0,
                    "med_gap": 0.0,
                    "left_size": 1,
                    "right_size": 0,
                },
            )
            if return_info
            else (labels, float(x[0]))
        )

    s = np.sort(x)
    diffs = np.diff(s)  # gaps between consecutive sorted values

    if diffs.size == 0:
        # All values identical
        threshold = float(s[0])
        labels = np.zeros(n, dtype=int)
        return (
            (
                labels,
                threshold,
                {
                    "rule": "all_equal",
                    "gap": 0.0,
                    "med_gap": 0.0,
                    "left_size": n,
                    "right_size": 0,
                },
            )
            if return_info
            else (labels, threshold)
        )

    i_max = int(np.argmax(diffs))
    max_gap = float(diffs[i_max])

    pos_diffs = diffs[diffs > 0]
    med_gap = float(np.median(pos_diffs)) if pos_diffs.size > 0 else 0.0

    # Heuristic: clear gap if it's much larger than the typical gap
    clear_gap = max_gap > 0 and (
        (med_gap == 0.0 and max_gap > 0.0) or (max_gap >= gap_ratio * med_gap)
    )

    left_size = i_max + 1
    right_size = n - left_size
    min_size = max(1, int(np.ceil(min_frac * n)))
    size_ok = (left_size >= min_size) and (right_size >= min_size)

    if clear_gap and size_ok:
        threshold = 0.5 * (s[i_max] + s[i_max + 1])
        rule = "max_gap"
    else:
        threshold = float(np.median(s))
        rule = "median"

    # Assign: <= threshold -> 0, > threshold -> 1 (then we’ll relabel so 1 = smaller cluster)
    raw_labels = (x > threshold).astype(int)

    # Ensure label 1 is the smaller cluster
    counts = np.bincount(raw_labels, minlength=2)
    # counts[0] = num at/below threshold, counts[1] = num above threshold
    # If label 1 isn't the smaller one, swap labels
    if counts[1] > counts[0]:
        labels = 1 - raw_labels
    else:
        labels = raw_labels

    if return_info:
        info = {
            "rule": rule,
            "gap": max_gap,
            "med_gap": med_gap,
            "left_size": left_size,
            "right_size": right_size,
        }
        return labels, threshold, info
    else:
        return labels, threshold


def paint_image(
    W: int,
    H: int,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    bg_color: np.ndarray = np.array([0.0, 0.0, 0.0]),
    depth: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Paint points on a canvas of given size and background color.

    Args:
        W: int
            Width of the output image.
        H: int
            Height of the output image.
        points: np.ndarray
            2D Points in pixel coordinates to paint, shape: (N, 2). Pixel coordinates are (x, y), where (0, 0) is the top-left corner.
            2D points should be in the range of [0, W) and [0, H), if not, points will be discarded.
        colors: Optional[np.ndarray]
            Colors of the points, shape: (N, 3). Colors are in RGB format, range: [0, 255] or [0,1] if float.
            If not provided, rainbow colors will be used.
        bg_color: np.ndarray
            Background color of the canvas, shape: (3,). Colors are in RGB format, range: [0, 255] or [0,1] if float.
        depth: Optional[np.ndarray]
            Depth of the points, shape: (N,). If provided, for duplicate pixels the *smaller* depth is treated as closer (foreground).
        save_path: Optional[str | Path]
            If provided, the image will be saved to this path (PNG/JPG depending on extension).

    Returns:
        image: np.ndarray
            Output image array of shape (H, W, 3), dtype=uint8 (RGB in [0, 255]).
    """
    if W <= 0 or H <= 0:
        raise ValueError("W and H must be positive integers.")

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    N = pts.shape[0]

    # Colors
    if colors is None:
        col_arr = np.array(get_rainbow_color(N), dtype=np.uint8)
    else:
        col_arr = _to_uint8_colors(colors)
        if col_arr.shape != (N, 3):
            raise ValueError(f"colors must have shape (N,3); got {col_arr.shape}.")

    # Background color
    bg = _to_uint8_colors(np.asarray(bg_color).reshape(1, 3)).reshape(3)

    # Initialize image
    image = np.broadcast_to(bg, (H, W, 3)).copy()
    if depth is not None:
        image = np.concatenate([image, np.zeros((H, W, 1), dtype=np.uint8)], axis=-1)

    if N == 0:
        return None

    # Round pixel coordinates to nearest integer positions
    xy = np.rint(pts).astype(np.int64)
    x = xy[:, 0]
    y = xy[:, 1]

    # Keep only points within bounds
    if depth is not None:
        # make nan becomes zero
        depth = np.where(np.isnan(depth), 0.0, depth)
        # make negative depth becomes zero
        depth = np.where(depth < 0, 0.0, depth)
        # normalize depth to [0, 255]
        depth = (depth / depth.max()) * 255
        # convert depth to uint8
        depth = depth.astype(np.uint8)

        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (depth > 0)
    else:
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    if not np.any(inb):
        return None

    x = x[inb]
    y = y[inb]
    col_arr = col_arr[inb]

    if depth is not None:
        d = np.asarray(depth).reshape(-1)
        if d.shape[0] != points.shape[0]:
            raise ValueError("depth must have shape (N,).")
        d = d[inb]

        # Linearize pixel coordinates and z-buffer with "smaller is closer"
        lin = y * W + x
        order = np.argsort(d)  # ascending: closest first
        lin_sorted = lin[order]

        # Keep the first occurrence per pixel (closest depth)
        _, first_idx = np.unique(lin_sorted, return_index=True)
        keep = order[first_idx]

        image[y[keep], x[keep], :3] = col_arr[keep]
        image[y[keep], x[keep], 3] = d[keep]
    else:
        # Respect input order for overdraw: later points overwrite earlier ones
        # (use a loop to guarantee order with duplicate indices)
        for xi, yi, ci in zip(x, y, col_arr):
            image[yi, xi] = ci

    ### Perform patch-wise foreground/background separation
    PATCH_SIZE = 32
    patches = patchify(
        image, patch_size=PATCH_SIZE
    )  # (N_patches, patch_size, patch_size, C)
    N_patches, _, _, C = patches.shape
    patches = patches.reshape(N_patches, PATCH_SIZE * PATCH_SIZE, C)

    for i in range(N_patches):
        patch_depth = patches[i, :, 3]  # (patch_size*patch_size,)
        patch_depth = patch_depth[patch_depth > 0]
        if len(patch_depth) == 0:
            continue
        _max = patch_depth.max()
        # normalize depth to [0, 1]
        patch_depth = patch_depth.astype(np.float64) / _max
        # identify foreground/background separation threshold
        _, threshold = dynamic_two_clusters(patch_depth, return_info=False)
        background_mask = patches[i, :, 3] > threshold * _max
        patches[i, background_mask, 3] = 0
    patches = patches.reshape(N_patches, PATCH_SIZE, PATCH_SIZE, C)
    image = unpatchify(patches, H, W, patch_size=PATCH_SIZE)
    # remove background points
    bg_mask = image[..., 3] == 0
    image[bg_mask, :3] = bg

    if save_path is not None:
        try:
            Image.fromarray(image).save(str(save_path))
            print(f"Image saved: [green bold]{save_path}")
        except Exception:
            print(f"Error saving image: {save_path}")
            pass

    return image


###############################################################################
### Match Helpers
###############################################################################


def _color_keys(img):
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    H, W, C = img.shape
    return img.view(np.dtype((np.void, img.dtype.itemsize * C))).reshape(H * W), H, W


def _coords(H, W):
    lin = np.arange(H * W)
    y, x = np.divmod(lin, W)
    return x, y


def _pack_colors(colors, dtype, C):
    """Turn a list/array of color tuples into fixed-size keys that match _color_keys."""
    arr = np.asarray(colors, dtype=dtype).reshape(-1, C)
    return arr.view(np.dtype((np.void, dtype.itemsize * C))).ravel()


def match_pixels_by_color(
    img1,
    img2,
    *,
    ignore_colors=[(0, 0, 0), (255, 255, 255)],  # e.g., [(0,0,0), (255,255,255)]
    ignore_predicate=None,  # callable(img)->bool mask of shape (H,W) or (H,W,1)
    all_pairs=True,
    max_pairs_per_color=None,
):
    k1, H1, W1 = _color_keys(img1)
    k2, H2, W2 = _color_keys(img2)
    C = img1.shape[2]
    x1_all, y1_all = _coords(H1, W1)
    x2_all, y2_all = _coords(H2, W2)

    keep1 = np.ones(H1 * W1, dtype=bool)
    keep2 = np.ones(H2 * W2, dtype=bool)

    # Predicate-based ignoring (e.g., transparency, near-white, etc.)
    if ignore_predicate is not None:
        m1 = np.asarray(ignore_predicate(img1)).reshape(H1 * W1)
        m2 = np.asarray(ignore_predicate(img2)).reshape(H2 * W2)
        keep1 &= ~m1
        keep2 &= ~m2

    # Exact color ignoring (fast set membership on packed keys)
    if ignore_colors is not None:
        ign1 = _pack_colors(ignore_colors, img1.dtype, C)
        # same palette for both images; if you need different, pass via predicate or adapt below
        keep1 &= ~np.isin(k1, ign1)
        keep2 &= ~np.isin(k2, ign1)

    # Filter down to kept pixels
    idx1_keep = np.flatnonzero(keep1)
    idx2_keep = np.flatnonzero(keep2)
    k1f = k1[idx1_keep]
    k2f = k2[idx2_keep]

    # Group by color on the filtered keys
    u1, inv1, cnt1 = np.unique(k1f, return_inverse=True, return_counts=True)
    u2, inv2, cnt2 = np.unique(k2f, return_inverse=True, return_counts=True)
    shared, i1, i2 = np.intersect1d(u1, u2, assume_unique=True, return_indices=True)

    order1 = np.argsort(inv1)
    b1 = np.cumsum(cnt1)
    s1 = np.r_[0, b1[:-1]]
    order2 = np.argsort(inv2)
    b2 = np.cumsum(cnt2)
    s2 = np.r_[0, b2[:-1]]

    for g1, g2, color in zip(i1, i2, shared):
        # indexes into the FILTERED arrays…
        a_filt = order1[s1[g1] : b1[g1]]
        b_filt = order2[s2[g2] : b2[g2]]
        # …map back to ORIGINAL linear pixel indices
        a = idx1_keep[a_filt]
        b = idx2_keep[b_filt]

        if not all_pairs:
            k = min(len(a), len(b))
            a, b = a[:k], b[:k]
            yield (
                color,
                np.stack((x1_all[a], y1_all[a]), axis=1),
                np.stack((x2_all[b], y2_all[b]), axis=1),
            )
        else:
            if max_pairs_per_color is not None:
                import math

                s = int(math.sqrt(max_pairs_per_color))
                a = a[: min(len(a), s)]
                b = b[: min(len(b), s)]
            A = np.repeat(a, len(b))
            B = np.tile(b, len(a))
            yield (
                color,
                np.stack((x1_all[A], y1_all[A]), axis=1),
                np.stack((x2_all[B], y2_all[B]), axis=1),
            )


###############################################################################


def calculate_flow_from_simulation(
    exported_sample_dir: Path,
    output_dir: Path,
    simulation_id: str,
    interpolate_cameras: bool = True,
    # fps_factor: int = 1,
):
    ### load video info
    video_info = _load_video_info(exported_sample_dir)
    sample_id = video_info["sample_id"]
    print(
        f"Calculating flow for sample: {sample_id} (interpolate_cameras: {interpolate_cameras})"
    )
    video_n_frames = video_info["N_frames"]

    # get number of points
    # N_pts = get_mesh_vertices_count(sample_id)

    # metadata_file
    metadata_file = OUT_SIMULATION_DIR / sample_id / simulation_id / "metadata.json"
    assert metadata_file.exists(), f"Metadata file {metadata_file} does not exist"
    metadata = _read_json(metadata_file)
    primary_start_index = metadata.get("primary_start_index", None)
    primary_end_index = metadata.get("primary_end_index", None)
    secondary_start_index = metadata.get("secondary_start_index", None)
    secondary_end_index = metadata.get("secondary_end_index", None)

    if secondary_start_index is None:
        total_simulated_points = primary_end_index
        N_pts = primary_start_index
    else:
        total_simulated_points = secondary_end_index
        _first_obj_pts = primary_start_index
        _second_obj_pts = secondary_start_index - primary_end_index
        N_pts = _first_obj_pts + _second_obj_pts

    if "fps" in simulation_id:
        _loc = simulation_id.find("fps_x")
        if _loc == -1:
            raise ValueError(
                f"Failed to find fps factor in simulation_id: {simulation_id}"
            )
        _factor = int(simulation_id[_loc + 5 : _loc + 6])
        assert _factor in [1, 2, 3, 4], f"Invalid fps factor: {_factor}"
        fps_factor = _factor
    else:
        fps_factor = 1

    if interpolate_cameras:
        n_frames_to_render = video_n_frames * fps_factor
    else:
        n_frames_to_render = video_n_frames

    ### load all frames' infos
    frame_dirs = _list_frame_dirs(exported_sample_dir)  # already sorted
    frame_infos = _load_frame_infos(frame_dirs)

    ### load all camera data
    cam_base_params = load_cam_base_params(frame_infos[0])
    W, H = cam_base_params["width"], cam_base_params["height"]
    all_intrinsics = [build_intrinsics(frame_infos[0]["K"])] * n_frames_to_render
    if "static" in sample_id:
        all_c2w = [
            np.array(frame_infos[0]["c2w"], dtype=np.float32)
        ] * n_frames_to_render
    else:
        # Load all_c2w from video_info
        all_c2w_path = (
            exported_sample_dir / "video_info" / video_info.get("all_c2w", "")
        )
        assert all_c2w_path.exists(), f"all_c2w path '{all_c2w_path}' does not exist"
        all_c2w, _, _ = load_grid(str(all_c2w_path))
        assert all_c2w.shape == (
            video_n_frames,
            4,
            4,
        ), f"all_c2w shape {all_c2w.shape} does not match expected shape ({video_n_frames}, 4, 4)"
        all_c2w = [all_c2w[i] for i in range(video_n_frames)]

        ### Interpolate c2w
        if interpolate_cameras and fps_factor > 1:
            all_c2w = interpolate_c2ws(
                all_c2w, factor=fps_factor
            )  # N -> N * fps_factor - fps_factor + 1

            if len(all_c2w) < n_frames_to_render:
                less = n_frames_to_render - len(all_c2w)
                all_c2w = (
                    all_c2w + [all_c2w[-1]] * less
                )  # N * fps_factor - fps_factor + 1 -> N * fps_factor

    ### gather all simulation results
    particles_dir = OUT_SIMULATION_DIR / sample_id / simulation_id / "particles"
    assert particles_dir.exists(), f"Particles directory {particles_dir} does not exist"
    print(f"Searching for simulation results in {particles_dir}...")
    particles_files = sorted(particles_dir.glob("*.npz"))
    
    if len(particles_files) > n_frames_to_render:
        particles_files = particles_files[:n_frames_to_render]
    elif len(particles_files) < n_frames_to_render:
        short_num = n_frames_to_render - len(particles_files)
        particles_files = particles_files + [particles_files[-1]] * short_num
    else:
        print("OK: number of particles files matches number of frames")

    ### sanity check
    assert (
        len(particles_files)
        == len(all_c2w)
        == len(all_intrinsics)
        == n_frames_to_render
    )

    ### create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    ### build render output directories
    def make_sub_output_dir(output_dir: Path, sub_dir_name: str) -> Path:
        output_dir = Path(output_dir)
        sub_dir = output_dir / sub_dir_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        return sub_dir

    flow_vis_dir = make_sub_output_dir(output_dir, "flow_visual")
    pixel_corr_dir = make_sub_output_dir(output_dir, "pixel_correspondences")
    video_out_dir = make_sub_output_dir(output_dir, "video")
    rainbow_points_colors = get_rainbow_color(N_pts)  # [(r,g,b)] * N_pts
    unique_points_colors = get_unique_colors(N_pts)  # [(r,g,b)] * N_pts
    # sanity check
    _unique_count = len(set(tuple(c) for c in unique_points_colors))
    if _unique_count != N_pts:
        print(
            f"Number of unique colors ({_unique_count}) does not match number of points ({N_pts})"
        )
        exit(1)

    images = []
    for i in range(n_frames_to_render):
        print(f"-" * 100)
        print(f"Rendering frame {i}...")
        time_start = time.time()
        particle_file = particles_files[i]
        c2w = all_c2w[i]
        intrinsics = all_intrinsics[i]
        points_3d, _, _ = ParticleIO.read_particles_3d(particle_file)

        # if len(points_3d) >= N_pts:
        #     # NOTE: N_pts is the number of vertices of the mesh (surface)
        #     # points_3d contains extra points for its volume infill tailing after the mesh points
        #     # so we only need the first N_pts points
        #     points_3d = points_3d[:N_pts]
        # else:
        #     raise ValueError(
        #         f"Number of points ({len(points_3d)}) does not match number of points ({N_pts})"
        #     )

        # sanity check
        assert (
            len(points_3d) == total_simulated_points
        ), f"Number of points ({len(points_3d)}) does not match number of total simulated points ({total_simulated_points})"
        # get rid of the volume points
        if secondary_start_index is None:
            points_3d = points_3d[:N_pts]
        else:
            points_3d_primary = points_3d[:primary_start_index]
            points_3d_secondary = points_3d[primary_end_index:secondary_start_index]
            points_3d = np.concatenate([points_3d_primary, points_3d_secondary], axis=0)

        assert (
            len(points_3d) == N_pts
        ), f"Number of points ({len(points_3d)}) does not match number of points ({N_pts})"

        points_2d, z_depth = project(points_3d, c2w, intrinsics)

        paint_image(
            W=W,
            H=H,
            points=points_2d,
            colors=rainbow_points_colors,
            depth=z_depth,
            save_path=flow_vis_dir / f"{i:05d}.png",
        )
        image = paint_image(
            W=W,
            H=H,
            points=points_2d,
            colors=unique_points_colors,
            depth=z_depth,
        )
        images.append(image)

        time_elapsed = time.time() - time_start
        print(f"Flow for frame {i} calculated in {time_elapsed:.1f} seconds\n")

    # corr_dict = {}
    for i in range(n_frames_to_render - 1):
        if images[i] is None or images[i + 1] is None:
            continue
        image_i = images[i][..., :3]
        image_j = images[i + 1][..., :3]
        key_name = f"{i:05d}_{i+1:05d}"
        # find pixel correspondences between image_i and image_j
        matches = [
            (x[1][0].tolist(), x[2][0].tolist())
            for x in match_pixels_by_color(image_i, image_j)
        ]
        matches = [(x1, y1, x2, y2) for (x1, y1), (x2, y2) in matches]
        print(f"Frame {i:>2}->{i+1:<2}: {len(matches)} matches")
        # corr_dict[key_name] = matches
        with open(pixel_corr_dir / f"{key_name}.json", "w") as f:
            json.dump(matches, f)

    print(f"-" * 100)
    time.sleep(2)
    print("[green bold]Making video from rendered frames...")
    make_video_from_frames_dir(
        flow_vis_dir, video_out_dir / f"{sample_id}_flow_visual.mp4"
    )


def calculate_flow(sample_id: str, simulation_id: str, overwrite: bool = False):
    # Check if simulation results is complete
    sim_success_file = OUT_SIMULATION_DIR / sample_id / simulation_id / "success.txt"
    if not sim_success_file.exists():
        print(
            f"  [red]✗[/red] Simulation results not found for {sample_id} at {sim_success_file}"
        )
        exit(3)  # semantic: 404 failed exit

    root_success_file = (
        OUT_RENDERING_DIR / sample_id / simulation_id / "success_flow.txt"
    )
    if root_success_file.exists() and not overwrite:
        print(
            f"  [yellow]✔ Skipped[/yellow] {sample_id} ({simulation_id}) because flow result exists"
        )
        return  # semantic: success exit

    success_files = []
    for interpolate_cameras in [True, False]:
        _txt = "more_frames" if interpolate_cameras else "original_length"
        output_dir = OUT_RENDERING_DIR / sample_id / simulation_id / _txt
        success_file = output_dir / "success_flow.txt"
        success_files.append(success_file)
        if success_file.exists() and not overwrite:
            print(f"Flow already calculated for {sample_id} at {_txt}, skipping.")
            continue

        exported_sample_dir = VIPE_EXPORT_DIR / sample_id

        backup_this_script(output_dir)

        calculate_flow_from_simulation(
            exported_sample_dir=exported_sample_dir,
            output_dir=output_dir,
            simulation_id=simulation_id,
            interpolate_cameras=interpolate_cameras,
        )
        success_file.touch()

    if all([success_file.exists() for success_file in success_files]):
        root_success_file.touch()
