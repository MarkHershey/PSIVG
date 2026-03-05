import argparse
import glob
import json
import os

import cv2
import numpy as np
import rp


def _get_video_name_from_selected_path(
    selected_path: str, newload: bool = False
) -> str:
    if newload:
        digits = "".join(ch for ch in selected_path if ch.isdigit())
        return "0" + digits[:3]
    else:

        _normalized_path = selected_path.replace("\\", "/")
        _first_segment = _normalized_path.split("/", 1)[0]
        _segment_prefix = _first_segment.split("_", 1)[0]
        return _segment_prefix


def _load_first_frame(first_frame_folder: str, video_name: str) -> np.ndarray:
    first_frame_path = os.path.join(first_frame_folder, f"{video_name}.png")
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")
    img = rp.load_image(first_frame_path)

    # If PNG is RGBA, drop alpha channel and keep RGB. for rebuttal data where we have alpha channel as well
    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def _read_correspondence_file(json_path: str) -> np.ndarray:
    with open(json_path, "r") as f:
        data = json.load(f)
    arr = np.array(data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"Invalid correspondence format in {json_path}; expected Nx4 list."
        )
    return arr


def _load_mask(mask_firstframe_folder: str, video_name: str) -> np.ndarray:
    """Load the mask for the first frame."""
    mask_path = os.path.join(mask_firstframe_folder, f"{video_name}/mask_0000_00.npy")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = np.load(mask_path)
    return mask > 0.5  # Convert to boolean mask


def _filter_correspondences_by_background_points(
    correspondences: np.ndarray, background_points: set
) -> np.ndarray:
    """Filter correspondences to keep only those starting from non-background pixels."""
    if correspondences.size == 0:
        return correspondences

    # Get source coordinates (x1, y1)
    x1 = np.round(correspondences[:, 0]).astype(np.int32)
    y1 = np.round(correspondences[:, 1]).astype(np.int32)

    # Create set of source pixel coordinates
    source_coords = set(zip(x1, y1))

    # Find correspondences that start from non-background pixels
    foreground_indices = []
    for i, (x, y) in enumerate(zip(x1, y1)):
        if (x, y) not in background_points:
            foreground_indices.append(i)

    # Return only correspondences where source pixel is not in background
    return correspondences[foreground_indices]


def _track_background_points(
    correspondences: np.ndarray, current_background_points: set
) -> set:
    """Track where background points move to in the next frame."""
    if correspondences.size == 0:
        return current_background_points

    # Get source and destination coordinates
    x1 = np.round(correspondences[:, 0]).astype(np.int32)
    y1 = np.round(correspondences[:, 1]).astype(np.int32)
    x2 = np.round(correspondences[:, 2]).astype(np.int32)
    y2 = np.round(correspondences[:, 3]).astype(np.int32)

    # Find where current background points move to
    new_background_points = set()
    for i, (src_x, src_y, dst_x, dst_y) in enumerate(zip(x1, y1, x2, y2)):
        if (src_x, src_y) in current_background_points:
            new_background_points.add((dst_x, dst_y))

    return new_background_points


def _create_mask_from_points(points: set, height: int, width: int) -> np.ndarray:
    """Create a comprehensive binary mask from a set of (x, y) points."""
    if not points:
        return np.zeros((height, width), dtype=np.uint8)

    # First, create a sparse mask with just the points
    mask = np.zeros((height, width), dtype=np.uint8)
    valid_points = []
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = 255
            valid_points.append([x, y])

    if not valid_points:
        return mask

    # Convert to numpy array for processing
    points_array = np.array(valid_points, dtype=np.int32)

    # Method 1: Use convex hull to create a comprehensive area
    if len(points_array) >= 3:  # Need at least 3 points for convex hull
        try:
            # Create convex hull
            hull = cv2.convexHull(points_array)
            # Fill the convex hull
            cv2.fillPoly(mask, [hull], 255)
        except:
            # Fallback to dilation if convex hull fails
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        # For fewer than 3 points, use dilation to expand the area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Additional dilation to ensure we cover the background area well
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, kernel_large, iterations=1)

    return mask


def _load_simulator_mask(mask_path: str) -> np.ndarray:
    """Load a simulator mask PNG file."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")
    return mask


def _remove_small_connected_components(
    mask: np.ndarray, min_size: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Remove connected components smaller than min_size from a binary mask.

    Args:
        mask: Binary mask (0 for background, 255 for foreground)
        min_size: Minimum size of connected components to keep

    Returns:
        Tuple of (cleaned_mask, removed_components_mask)
        - cleaned_mask: Mask with small components removed
        - removed_components_mask: Mask showing only the removed components (0 if none removed)
    """
    # Convert to binary (0 or 1)
    # binary_mask = (mask > 127).astype(np.uint8)
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=4
    )

    # If there's only one component (background), return original mask
    if num_labels <= 1:
        print(f"\033[93mNo foreground components found\033[0m")
        return mask, np.zeros_like(mask)

    print(f"\033[91mFound {num_labels-1} connected components:\033[0m")

    # Print areas of all components
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        print(f"  Component {i}: area = {area}")

    # Create masks for kept and removed components
    cleaned_mask = np.zeros_like(binary_mask)
    removed_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned_mask[labels == i] = 1
        else:
            removed_mask[labels == i] = 1
            print(
                f"\033[91mRemoved small connected component {i} with area {area}\033[0m"
            )

    # Convert back to 0-255 range
    cleaned_mask = cleaned_mask * 255
    removed_mask = removed_mask * 255

    return cleaned_mask, removed_mask


def _extract_first_frame_visible_points(
    first_correspondences: np.ndarray, simulator_mask: np.ndarray
) -> set:
    """Extract the set of source points that are visible in the first frame based on simulator mask.

    Args:
        first_correspondences: Nx4 array of correspondences from first frame
        simulator_mask: Binary mask from simulator (foreground = True/255)

    Returns:
        Set of (x, y) tuples representing source points visible in first frame (within simulator mask)
    """
    if first_correspondences.size == 0:
        return set()

    # Get source coordinates (x1, y1) from first correspondence file
    x1 = np.round(first_correspondences[:, 0]).astype(np.int32)
    y1 = np.round(first_correspondences[:, 1]).astype(np.int32)

    # Filter points to only include those within the simulator mask
    visible_points = set()
    h, w = simulator_mask.shape

    for src_x, src_y in zip(x1, y1):
        # Check if point is within image bounds and within simulator mask
        if (
            0 <= src_x < w and 0 <= src_y < h and simulator_mask[src_y, src_x] > 0
        ):  # Simulator mask is foreground
            visible_points.add((src_x, src_y))

    return visible_points


def _find_destination_points(correspondences: np.ndarray, source_points: set) -> dict:
    """Find destination points for a given set of source points in correspondence data.

    Args:
        correspondences: Nx4 array of correspondences [x1, y1, x2, y2]
        source_points: Set of (x, y) tuples to find destinations for

    Returns:
        Dictionary with:
        - 'found': set of source points that were found in correspondences
        - 'destinations': dict mapping source points to their destination points
        - 'missing': set of source points that were not found in correspondences
    """
    if correspondences.size == 0:
        return {"found": set(), "destinations": {}, "missing": source_points.copy()}

    # Get source and destination coordinates
    x1 = np.round(correspondences[:, 0]).astype(np.int32)
    y1 = np.round(correspondences[:, 1]).astype(np.int32)
    x2 = np.round(correspondences[:, 2]).astype(np.int32)
    y2 = np.round(correspondences[:, 3]).astype(np.int32)

    # Find which source points are present in correspondences
    found_points = set()
    destinations = {}

    for i, (src_x, src_y, dst_x, dst_y) in enumerate(zip(x1, y1, x2, y2)):
        if (src_x, src_y) in source_points:
            found_points.add((src_x, src_y))
            destinations[(src_x, src_y)] = (dst_x, dst_y)

    # Points that were not found in correspondences
    missing_points = source_points - found_points

    return {
        "found": found_points,
        "destinations": destinations,
        "missing": missing_points,
    }


def _save_filtered_correspondences(
    correspondences: np.ndarray, output_path: str
) -> None:
    """Save filtered correspondences to a JSON file in the same format as loaded.

    Args:
        correspondences: Nx4 array of correspondences [x1, y1, x2, y2]
        output_path: Path to save the JSON file
    """
    if correspondences.size == 0:
        # Save empty list for empty correspondences
        json_data = []
    else:
        # Convert numpy array to list of lists for JSON serialization
        json_data = correspondences.tolist()

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)


def _create_convex_hull_mask_from_points(
    points: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Create convex hull mask from Nx2 points array. Falls back to dilation if <3 points."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if points is None or len(points) == 0:
        return mask
    pts = np.array(points, dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 255)
    else:
        for x, y in pts:
            mask[y, x] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _save_correspondence_plot(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    correspondences: np.ndarray,
    out_path: str,
    max_points: int = 4000,
    source_sim_mask: np.ndarray = None,
) -> None:
    """Save a side-by-side visualization of correspondences between src_img and dst_img.

    Left panel shows source with source points; right shows destination with destination points.
    Lines connect matching points across the split.
    """
    if src_img is None or dst_img is None:
        return
    h, w = src_img.shape[:2]
    if dst_img.shape[:2] != (h, w):
        dst_img = cv2.resize(dst_img, (w, h), interpolation=cv2.INTER_LINEAR)

    canvas = np.concatenate([src_img.copy(), dst_img.copy()], axis=1)

    # Optional overlay of simulator mask on left panel
    if source_sim_mask is not None and source_sim_mask.shape[:2] == (h, w):
        left = canvas[:, :w]
        mask_bin = (source_sim_mask > 127).astype(np.uint8)
        overlay = left.copy()
        overlay[mask_bin > 0] = (0, 0, 255)
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, left, 1 - alpha, 0, dst=left)
        try:
            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(left, contours, -1, (0, 0, 255), 1)
        except Exception:
            pass

    if correspondences is None or correspondences.size == 0:
        rp.save_image(canvas, out_path)
        return

    pts = correspondences.astype(np.float32)
    # Convex hull visualization of SOURCE points on left panel
    try:
        src_pts = pts[:, :2].copy()
        src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, w - 1)
        src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, h - 1)
        if src_pts.shape[0] >= 3:
            hull = cv2.convexHull(src_pts.astype(np.float32))
            cv2.polylines(
                canvas,
                [hull.astype(np.int32)],
                isClosed=True,
                color=(255, 0, 255),
                thickness=1,
            )
    except Exception:
        pass

    if len(pts) > max_points:
        idxs = np.linspace(0, len(pts) - 1, max_points).astype(np.int32)
        pts = pts[idxs]

    x1 = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
    y1 = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
    x2 = np.clip(np.round(pts[:, 2]).astype(np.int32), 0, w - 1)
    y2 = np.clip(np.round(pts[:, 3]).astype(np.int32), 0, h - 1)

    right_offset = w

    # Generate unique colors for each correspondence pair
    num_correspondences = len(pts)
    colors = []
    for i in range(num_correspondences):
        # Generate a unique color using HSV color space for better color distribution
        hue = int(
            180 * i / num_correspondences
        )  # Use full hue range (0-180 for OpenCV)
        saturation = 255
        value = 255
        hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(map(int, bgr_color)))

    # Plot corresponding points with the same unique color
    for i, (sx, sy, dx, dy) in enumerate(zip(x1, y1, x2, y2)):
        color = colors[i]
        # Draw source point on left panel
        cv2.circle(canvas, (int(sx), int(sy)), 2, color, thickness=-1)
        # Draw destination point on right panel
        cv2.circle(canvas, (int(dx + right_offset), int(dy)), 2, color, thickness=-1)

    rp.save_image(canvas, out_path)


def _apply_dense_warp_with_found_points(
    prev_img: np.ndarray,
    correspondences: np.ndarray,
    dst_simulator_mask: np.ndarray = None,
    found_points: set = None,
    k: int = 3,
) -> tuple:
    """
    Apply dense warping using found points with convex hull of destination points.

    Args:
        prev_img: Previous frame image
        correspondences: Nx4 array of correspondences [x1, y1, x2, y2]
        dst_simulator_mask: Simulator mask for destination frame (optional)
        found_points: Set of (x, y) tuples representing found points in source frame
        k: Number of nearest neighbors for interpolation (default: 4)

    Returns:
        tuple: (warped_points_img, warped_mask, uncovered_mask) where:
            - warped_points_img: The warped image
            - warped_mask: Mask showing areas covered by convex hull
            - uncovered_mask: Mask showing areas in simulator_mask not covered by convex hull
    """
    h, w = prev_img.shape[:2]
    warped_points_img = np.zeros_like(prev_img)
    warped_mask = np.zeros((h, w), dtype=np.uint8)

    if (
        correspondences is None
        or correspondences.size == 0
        or found_points is None
        or len(found_points) == 0
    ):
        # When no points are found, the entire simulator mask is "uncovered"
        uncovered_mask = np.zeros((h, w), dtype=np.uint8)
        if dst_simulator_mask is not None and dst_simulator_mask.shape[:2] == (h, w):
            uncovered_mask = (dst_simulator_mask > 127).astype(np.uint8) * 255
        return warped_points_img, warped_mask, uncovered_mask

    # Get source and destination points
    src_pts = correspondences[:, :2].astype(np.float32)
    dst_pts = correspondences[:, 2:].astype(np.float32)

    # Find destination points corresponding to the found points
    found_destinations = []
    for i, (src_x, src_y, dst_x, dst_y) in enumerate(
        zip(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0], dst_pts[:, 1])
    ):
        src_point = (int(round(src_x)), int(round(src_y)))
        if src_point in found_points:
            found_destinations.append([dst_x, dst_y])

    if len(found_destinations) == 0:
        # When no destination points are found, the entire simulator mask is "uncovered"
        uncovered_mask = np.zeros((h, w), dtype=np.uint8)
        if dst_simulator_mask is not None and dst_simulator_mask.shape[:2] == (h, w):
            uncovered_mask = (dst_simulator_mask > 127).astype(np.uint8) * 255
        return warped_points_img, warped_mask, uncovered_mask

    # Convert to numpy array and clip to image bounds
    found_dest_pts = np.array(found_destinations, dtype=np.float32)
    found_dest_pts_int = np.stack(
        [
            np.clip(np.round(found_dest_pts[:, 0]).astype(np.int32), 0, w - 1),
            np.clip(np.round(found_dest_pts[:, 1]).astype(np.int32), 0, h - 1),
        ],
        axis=1,
    )

    # Create convex hull mask from destination points
    dest_hull_mask = _create_convex_hull_mask_from_points(found_dest_pts_int, h, w)

    # Compute uncovered mask: areas in simulator_mask not covered by convex hull
    uncovered_mask = np.zeros((h, w), dtype=np.uint8)
    if dst_simulator_mask is not None and dst_simulator_mask.shape[:2] == (h, w):
        # Convert simulator mask to binary
        sim_bin = (dst_simulator_mask > 127).astype(np.uint8) * 255
        # Uncovered areas = simulator mask AND NOT convex hull mask
        uncovered_mask = cv2.bitwise_and(sim_bin, cv2.bitwise_not(dest_hull_mask))

    # Build a mask of known destination pixels so we don't re-interpolate them
    known_dst_mask = np.zeros((h, w), dtype=np.uint8)
    known_dst_mask[found_dest_pts_int[:, 1], found_dest_pts_int[:, 0]] = 255
    interpolation_mask = cv2.bitwise_and(
        dest_hull_mask, cv2.bitwise_not(known_dst_mask)
    )

    # Query coordinates only in interpolated area (convex hull minus known destination points)
    qy, qx = np.where(interpolation_mask > 0)
    if qx.size == 0:
        # No interpolation needed, just return the known points
        warped_points_img[found_dest_pts_int[:, 1], found_dest_pts_int[:, 0]] = (
            prev_img[found_dest_pts_int[:, 1], found_dest_pts_int[:, 0]]
        )
        warped_mask[found_dest_pts_int[:, 1], found_dest_pts_int[:, 0]] = 255
        return warped_points_img, warped_mask, uncovered_mask

    query_pts = np.stack([qx.astype(np.float32), qy.astype(np.float32)], axis=1)

    # Find k nearest neighbors in DESTINATION space using OpenCV FLANN
    indices = None
    dists = None
    found_knn = False

    try:
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=32)
        flann = cv2.flann_Index(found_dest_pts.astype(np.float32), index_params)

        idxs, dists_flann = flann.knnSearch(
            query_pts.astype(np.float32),
            min(k, len(found_dest_pts)),
            params=search_params,
        )
        indices = idxs
        dists = np.sqrt(dists_flann)
        found_knn = True
        print(f"Found kNN using OpenCV FLANN for {len(query_pts)} query points")
    except Exception as e:
        print(f"Warning: FLANN failed, using simple nearest neighbor: {str(e)}")
        found_knn = False

    # Ensure shapes are (m,k)
    if dists.ndim == 1:
        dists = dists[:, None]
    if indices.ndim == 1:
        indices = indices[:, None]

    k_eff = indices.shape[1]

    # Inverse-distance weighting (handle exact matches)
    eps = 1e-6
    exact_match = dists[:, 0] < eps
    weights = 1.0 / (dists + eps)
    weights_sum = np.sum(weights, axis=1, keepdims=True) + eps
    norm_weights = weights / weights_sum

    # Create a direct mapping from found destination points to their corresponding source points
    # This is much more efficient than nested loops
    found_src_pts = []
    for i, (src_x, src_y, dst_x, dst_y) in enumerate(
        zip(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0], dst_pts[:, 1])
    ):
        src_point = (int(round(src_x)), int(round(src_y)))
        if src_point in found_points:
            found_src_pts.append([src_x, src_y])

    found_src_pts = np.array(found_src_pts, dtype=np.float32)

    # Now we can directly use the indices to get the corresponding source points
    # This is similar to the approach in _apply_dense_warp
    src_neighbors = found_src_pts[indices]  # (m, k, 2)

    # Compute weighted average of source points
    mapped_src = (norm_weights[..., None] * src_neighbors).sum(axis=1)

    # For exact matches, use the exact corresponding source point
    if np.any(exact_match):
        mapped_src[exact_match] = found_src_pts[indices[exact_match, 0]]

    # Ensure mapped source points are within image bounds
    sx = np.clip(np.round(mapped_src[:, 0]).astype(np.int32), 0, w - 1)
    sy = np.clip(np.round(mapped_src[:, 1]).astype(np.int32), 0, h - 1)

    # Sample source image at mapped source points and write into query positions
    warped_points_img[qy, qx] = prev_img[sy, sx]
    warped_mask[qy, qx] = 255

    # Also scatter the original found correspondences exactly
    for i, (src_x, src_y, dst_x, dst_y) in enumerate(
        zip(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0], dst_pts[:, 1])
    ):
        src_point = (int(round(src_x)), int(round(src_y)))
        if src_point in found_points:
            dst_x_int = int(round(dst_x))
            dst_y_int = int(round(dst_y))
            if 0 <= dst_x_int < w and 0 <= dst_y_int < h:
                warped_points_img[dst_y_int, dst_x_int] = prev_img[
                    int(round(src_y)), int(round(src_x))
                ]
                warped_mask[dst_y_int, dst_x_int] = 255

    return warped_points_img, warped_mask, uncovered_mask


def process_sequence(
    base_folder: str,
    selected_rel_path: str,
    first_frame_folder: str,
    output_base_folder: str,
    mask_firstframe_folder: str,
    newload: bool = False,
) -> None:
    sequence_folder = os.path.join(base_folder, selected_rel_path)
    if not os.path.exists(sequence_folder):
        print(f"Warning: Sequence folder not found at {sequence_folder}, skipping...")
        return

    corr_folder = os.path.join(sequence_folder, "pixel_correspondences")
    if not os.path.exists(corr_folder):
        print(f"Warning: pixel_correspondences not found at {corr_folder}, skipping...")
        return

    corr_files = sorted(glob.glob(os.path.join(corr_folder, "*.json")))
    if not corr_files:
        print(f"Warning: No correspondence json files in {corr_folder}, skipping...")
        return

    #### getting the first 49 frames for now due to gwtf. Can change this logic (and logic elsewhere) if need something more complicated
    corr_files = corr_files[:48]

    video_name = _get_video_name_from_selected_path(selected_rel_path, newload)
    first_img = _load_first_frame(first_frame_folder, video_name)

    # Load mask for initial background point identification
    mask = _load_mask(mask_firstframe_folder, video_name)
    print(f"Loaded mask for {video_name}, foreground pixels: {np.sum(mask)}")

    # Initialize background points from the first frame mask (background = False in mask)
    h, w = mask.shape
    background_points = set()
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:  # Background pixel
                background_points.add((x, y))

    print(f"Initial background points: {len(background_points)}")

    simulator_masks_folder = os.path.join(sequence_folder, "obj_mask")
    obj_only_folder = os.path.join(sequence_folder, "obj_only")

    simulator_masks_files = sorted(
        glob.glob(os.path.join(simulator_masks_folder, "*.png"))
    )
    if not simulator_masks_files:
        print(f"Warning: No simulator masks in {simulator_masks_folder}, skipping...")
        return

    obj_only_files = sorted(glob.glob(os.path.join(obj_only_folder, "*.png")))
    if not obj_only_files:
        print(f"Warning: No simulator RGB frames in {obj_only_folder}, skipping...")
        return

    # Initialize tracking system
    first_frame_points = None
    tracking_results = []

    out_seq_folder = os.path.join(
        output_base_folder, selected_rel_path, "warped_from_first"
    )
    os.makedirs(out_seq_folder, exist_ok=True)

    # Create folder for updated masks
    # updated_masks_folder = os.path.join(output_base_folder, selected_rel_path, "updated_masks")
    updated_masks_folder = os.path.join(
        output_base_folder, selected_rel_path, "convex_hull_masks"
    )
    os.makedirs(updated_masks_folder, exist_ok=True)
    # Create folder for simulator-only regions (simulator minus updated convex-hull)
    simulator_only_masks_folder = os.path.join(
        output_base_folder, selected_rel_path, "simulator_only_masks"
    )
    os.makedirs(simulator_only_masks_folder, exist_ok=True)

    # Create folder for updated simulator masks (after removing small components)
    obj_masks_updated_folder = os.path.join(
        output_base_folder, selected_rel_path, "obj_masks_updated"
    )
    os.makedirs(obj_masks_updated_folder, exist_ok=True)

    # Create folder for filtered correspondence files
    corr_files_folder = os.path.join(
        output_base_folder, selected_rel_path, "corr_files"
    )
    os.makedirs(corr_files_folder, exist_ok=True)

    # Debug outputs for warped results
    warped_points_out_folder = os.path.join(
        output_base_folder, selected_rel_path, "warped_points_img"
    )
    warped_masks_out_folder = os.path.join(
        output_base_folder, selected_rel_path, "warped_masks"
    )
    os.makedirs(warped_points_out_folder, exist_ok=True)
    os.makedirs(warped_masks_out_folder, exist_ok=True)

    # Debug outputs for correspondence visualization
    corr_vis_folder = os.path.join(
        output_base_folder, selected_rel_path, "correspondence_plots"
    )
    os.makedirs(corr_vis_folder, exist_ok=True)

    # Tracking outputs
    tracking_vis_folder = os.path.join(
        output_base_folder, selected_rel_path, "tracking_visualizations"
    )
    os.makedirs(tracking_vis_folder, exist_ok=True)

    # Frame 0 convex hull from sources of first correspondence (after filtering)
    frame0_path = os.path.join(out_seq_folder, "00000.png")
    # We'll save after compositing
    saved_paths = []

    # Build background-filtered points from first correspondence for frame-0 mask
    try:
        first_corr = _read_correspondence_file(corr_files[0])
        print("first_corr", first_corr.shape)  ## (13273, 4)
        first_corr_filtered = _filter_correspondences_by_background_points(
            first_corr, background_points
        )
        print("first_corr_filtered", first_corr_filtered.shape)  ## (12250, 4)

        if first_corr_filtered.shape[0] == 0:
            raise Exception(
                f"Warning: No foreground correspondences found for frame 0; stopping sequence early. Likely because first frame is all background, due to segmentation errors"
            )

        # Initialize tracking with first frame points (using simulator mask)
        try:
            first_simulator_mask = _load_simulator_mask(simulator_masks_files[0])
            first_frame_points = _extract_first_frame_visible_points(
                first_corr_filtered, first_simulator_mask
            )
            print(
                f"Initialized tracking with {len(first_frame_points)} first-frame points (within simulator mask)"
            )
        except Exception as e:
            print(
                f"Warning: could not initialize tracking with simulator mask: {str(e)}"
            )
            first_frame_points = set()

        if first_corr_filtered.size > 0:
            src_x0 = np.clip(
                np.round(first_corr_filtered[:, 0]).astype(np.int32),
                0,
                first_img.shape[1] - 1,
            )
            src_y0 = np.clip(
                np.round(first_corr_filtered[:, 1]).astype(np.int32),
                0,
                first_img.shape[0] - 1,
            )
            frame0_points = np.stack([src_x0, src_y0], axis=1)
        else:
            frame0_points = np.empty((0, 2), dtype=np.int32)
    except Exception as e:
        print(
            f"Warning: could not build frame-0 convex hull from correspondences: {str(e)}"
        )
        frame0_points = np.empty((0, 2), dtype=np.int32)
        first_frame_points = set()

    # print("frame0_points", frame0_points.shape) ## (12250, 2)

    frame0_updated_mask = _create_convex_hull_mask_from_points(
        frame0_points,
        first_img.shape[0],
        first_img.shape[1],
    )
    frame0_mask_path = os.path.join(updated_masks_folder, "00000.png")
    cv2.imwrite(frame0_mask_path, frame0_updated_mask)
    print(f"Saved updated mask for frame 0: {frame0_mask_path}")

    # Simulator-only for frame 0 and composite
    sim_only0 = None
    try:
        sim0 = _load_simulator_mask(simulator_masks_files[0])

        # Erode the simulator mask to shrink foreground boundary by ~2 pixels. Otherwise, there will be a black boundary around the object.
        ### This is because the simulator mask is slightly larger than the object rgb
        _kernel_2px = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sim0 = cv2.erode(sim0, _kernel_2px, iterations=2)

        # Also save the first simulator mask into obj_masks_updated_folder
        first_updated_obj_mask_path = os.path.join(
            obj_masks_updated_folder, "00000.png"
        )
        cv2.imwrite(first_updated_obj_mask_path, sim0)
        print(
            f"Saved updated simulator mask for frame 0: {first_updated_obj_mask_path}"
        )

        sim0_bin = (sim0 > 127).astype(np.uint8) * 255
        sim_only0 = cv2.bitwise_and(sim0_bin, cv2.bitwise_not(frame0_updated_mask))
        sim_only0_path = os.path.join(simulator_only_masks_folder, "00000.png")
        cv2.imwrite(sim_only0_path, sim_only0)
        print(f"Saved simulator-only mask for frame 0: {sim_only0_path}")
    except Exception as e:
        print(f"Warning: could not compute simulator-only mask for frame 0: {str(e)}")

    # Load fixed background used for all frames
    background_img = None
    try:
        background_path = os.path.join(
            output_base_folder, selected_rel_path, "background.png"
        )
        if os.path.exists(background_path):
            background_img = rp.load_image(background_path)
            # Resize to match sequence resolution if needed
            if background_img.shape[:2] != first_img.shape[:2]:
                background_img = cv2.resize(
                    background_img,
                    (first_img.shape[1], first_img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
        else:
            print(
                f"Warning: background.png not found at {background_path}; falling back to previous-frame base for compositing."
            )
    except Exception as e:
        print(f"Warning: failed to load background image: {str(e)}")
        background_img = None

    frame0_composited = first_img

    rp.save_image(frame0_composited, frame0_path)
    saved_paths.append(frame0_path)

    prev_img = frame0_composited

    all_first_frame_points = [first_frame_points]
    # Iteratively warp using 00000_00001.json, 00001_00002.json, ...
    for idx, jf in enumerate(corr_files, start=1):
        # print("idx", idx) ## starts from 1
        try:
            correspondences = _read_correspondence_file(jf)
            print(
                f"Loaded {len(correspondences)} correspondences from {os.path.basename(jf)}"
            )
        except Exception as e:
            print(f"Warning: Failed reading {jf}: {str(e)}; stopping sequence early.")
            break

        # Filter correspondences to remove those starting from tracked background points
        filtered_correspondences = _filter_correspondences_by_background_points(
            correspondences, background_points
        )
        print(
            f"Filtered to {len(filtered_correspondences)} foreground correspondences (removed {len(correspondences) - len(filtered_correspondences)} background)"
        )

        print("DEBUG: filtered_correspondences", filtered_correspondences.shape)
        if filtered_correspondences.shape[0] == 0:
            raise Exception(
                f"Warning: No foreground correspondences found for frame {idx}; stopping sequence early."
            )

        # Track where current background points move to for next frame
        background_points = _track_background_points(correspondences, background_points)
        print(f"Tracked background points for next frame: {len(background_points)}")

        # Compute simulator-only region for this frame
        sim_only = None
        removed_components_mask = None
        try:
            print("DEBUG: Loading simulator mask for frame ", idx)
            simulator_mask = _load_simulator_mask(simulator_masks_files[idx])

            # Remove small connected components from simulator mask
            simulator_mask, removed_components_mask = (
                _remove_small_connected_components(simulator_mask, min_size=30)
            )

            # Erode the simulator mask to shrink foreground boundary by ~2 pixels. Otherwise, there will be a black boundary around the object.
            _kernel_2px = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            simulator_mask = cv2.erode(simulator_mask, _kernel_2px, iterations=2)

            # Save the updated simulator mask after removing small components
            updated_obj_mask_path = os.path.join(
                obj_masks_updated_folder, f"{idx:05d}.png"
            )
            cv2.imwrite(updated_obj_mask_path, simulator_mask)
            print(
                f"Saved updated simulator mask for frame {idx}: {updated_obj_mask_path}"
            )

            # Save removed components mask if there are any removed components
            if np.any(removed_components_mask > 0):
                removed_path = os.path.join(
                    simulator_only_masks_folder, f"{idx:05d}_removed.png"
                )
                cv2.imwrite(removed_path, removed_components_mask)
                print(f"Saved removed components mask for frame {idx}: {removed_path}")
        except Exception as e:
            print(
                f"Warning: could not compute simulator-only mask for frame {idx}: {str(e)}"
            )

        ###### To save an updated mask
        ### Filter correspondences using the removed_components_mask (drop points that land in removed comps)
        try:
            print("DEBUG: Filtering correspondences by removed_components_mask")
            print("removed_components_mask", removed_components_mask.shape)
            print("filtered_correspondences", filtered_correspondences.shape)

            if (
                removed_components_mask is not None
                and filtered_correspondences is not None
                and filtered_correspondences.size > 0
            ):
                h_rcm, w_rcm = removed_components_mask.shape
                dst_x = np.clip(
                    np.round(filtered_correspondences[:, 2]).astype(np.int32),
                    0,
                    w_rcm - 1,
                )
                dst_y = np.clip(
                    np.round(filtered_correspondences[:, 3]).astype(np.int32),
                    0,
                    h_rcm - 1,
                )
                print("dst_x", dst_x.shape)
                print("dst_y", dst_y.shape)
                print("removed_components_mask", removed_components_mask.shape)
                print(
                    "removed_components_mask", np.sum(removed_components_mask) / 255.0
                )
                keep_mask = removed_components_mask[dst_y, dst_x] == 0
                if not np.all(keep_mask):
                    num_removed = int(np.sum(~keep_mask))
                    print(
                        f"\033[91mFiltered out {num_removed} correspondences for updated masks due to removed small components\033[0m"
                    )
                filtered_correspondences_for_updated_mask = filtered_correspondences[
                    keep_mask
                ]
        except Exception as e:
            print(
                f"Warning: failed to filter correspondences by removed_components_mask for frame {idx}: {str(e)}"
            )

        ### Create and save convex hull mask from current frame's filtered correspondence sources
        if filtered_correspondences_for_updated_mask.size > 0:
            src_x = np.round(filtered_correspondences_for_updated_mask[:, 2]).astype(
                np.int32
            )
            src_y = np.round(filtered_correspondences_for_updated_mask[:, 3]).astype(
                np.int32
            )
            points_xy = np.stack([src_x, src_y], axis=1)
        else:
            points_xy = np.empty((0, 2), dtype=np.int32)
        updated_mask = _create_convex_hull_mask_from_points(
            points_xy, prev_img.shape[0], prev_img.shape[1]
        )
        mask_path = os.path.join(updated_masks_folder, f"{idx:05d}.png")
        cv2.imwrite(mask_path, updated_mask)
        print(f"Saved updated mask for frame {idx}: {mask_path}")

        ###### find the set of points that are present
        first_frame_corr_dict = _find_destination_points(
            filtered_correspondences, all_first_frame_points[idx - 1]
        )
        all_first_frame_points.append(
            set(first_frame_corr_dict["destinations"].values())
        )

        print(
            f"Frame {idx}: {len(first_frame_corr_dict['found'])} points corresponding to first frame"
        )
        print(
            f"Frame {idx}: {len(first_frame_corr_dict['missing'])} points missing from previous frame"
        )

        first_frame_found_points = first_frame_corr_dict["found"]

        # # Load simulator mask for SOURCE (previous) frame to constrain mapped source positions
        source_sim_mask = None
        try:
            prev_idx = max(0, idx - 1)
            source_sim_mask = _load_simulator_mask(simulator_masks_files[prev_idx])
        except Exception:
            source_sim_mask = None

        # Load the previous frame's updated mask to constrain source positions
        source_updated_mask = None
        try:
            if idx > 1:  # For frames after the first
                prev_mask_path = os.path.join(updated_masks_folder, f"{idx-1:05d}.png")
                if os.path.exists(prev_mask_path):
                    source_updated_mask = cv2.imread(
                        prev_mask_path, cv2.IMREAD_GRAYSCALE
                    )
            elif idx == 1:  # For the first frame, use frame 0's mask
                prev_mask_path = os.path.join(updated_masks_folder, "00000.png")
                if os.path.exists(prev_mask_path):
                    source_updated_mask = cv2.imread(
                        prev_mask_path, cv2.IMREAD_GRAYSCALE
                    )
        except Exception as e:
            print(f"Warning: could not load previous updated mask: {str(e)}")
            source_updated_mask = None

        # warped_points_img, warped_mask, uncovered_mask = _apply_dense_warp_with_found_points(prev_img, filtered_correspondences, dst_simulator_mask=simulator_mask, found_points=first_frame_found_points)
        warped_points_img, warped_mask, uncovered_mask = (
            _apply_dense_warp_with_found_points(
                prev_img,
                filtered_correspondences,
                dst_simulator_mask=simulator_mask,
                found_points=first_frame_found_points,
            )
        )

        # Save debug warped outputs
        try:
            rp.save_image(
                warped_points_img,
                os.path.join(warped_points_out_folder, f"{idx:05d}.png"),
            )
            cv2.imwrite(
                os.path.join(warped_masks_out_folder, f"{idx:05d}.png"), warped_mask
            )
        except Exception as e:
            print(
                f"Warning: failed to save warped debug outputs for frame {idx}: {str(e)}"
            )

        # Save correspondence visualization using prev_img (left) and warped_points_img (right). To visualize the current bug
        try:
            vis_out_path = os.path.join(corr_vis_folder, f"{idx:05d}.png")
            _save_correspondence_plot(
                prev_img,
                warped_points_img,
                filtered_correspondences_for_updated_mask,
                vis_out_path,
                max_points=200,
                source_sim_mask=source_updated_mask,
            )

        except Exception as e:
            print(
                f"Warning: failed to save correspondence plot for frame {idx}: {str(e)}"
            )

        # Save filtered correspondences to corr_files folder
        try:
            corr_output_path = os.path.join(corr_files_folder, f"{idx:05d}.json")
            # _save_filtered_correspondences(filtered_correspondences, corr_output_path)
            _save_filtered_correspondences(
                filtered_correspondences_for_updated_mask, corr_output_path
            )

            print(f"Saved filtered correspondences: {corr_output_path}")
        except Exception as e:
            print(
                f"Warning: failed to save filtered correspondences for frame {idx}: {str(e)}"
            )

        # Composite warped points onto the fixed background (or previous frame if background missing)
        base_img = background_img if background_img is not None else prev_img
        if warped_mask is not None and warped_mask.size > 0:
            next_img = np.where(
                (warped_mask > 0)[..., None], warped_points_img, base_img
            )
        else:
            next_img = base_img.copy()

        # Composite simulator-only RGB onto next_img if available
        try:
            sim_rgb = rp.load_image(obj_only_files[idx])
            if uncovered_mask is not None:
                next_img = np.where((uncovered_mask > 0)[..., None], sim_rgb, next_img)
        except Exception as e:
            next_img = first_img  ### nonsense to get it to run
            print(
                f"Warning: could not composite simulator-only RGB for frame {idx}: {str(e)}"
            )

        out_path = os.path.join(out_seq_folder, f"{idx:05d}.png")
        rp.save_image(next_img, out_path)
        saved_paths.append(out_path)
        prev_img = next_img

    #### here, we want to account for the case where there were not enough corr_files.
    #### this means the object is not in the foreground for some frames!
    num_corr_files = len(corr_files)
    if num_corr_files < 48:
        print(f"Warning: there are only {num_corr_files} corr_files")

        for idx in range(num_corr_files, 49):
            print(f"Creating empty masks and plain background for missing frame {idx}")

            # Create empty mask (all zeros) for obj_masks_updated
            empty_mask = np.zeros(
                (first_img.shape[0], first_img.shape[1]), dtype=np.uint8
            )
            obj_mask_path = os.path.join(obj_masks_updated_folder, f"{idx:05d}.png")
            cv2.imwrite(obj_mask_path, empty_mask)
            print(f"Saved empty obj mask for frame {idx}: {obj_mask_path}")

            # Create empty mask (all zeros) for convex_hull_masks
            convex_hull_mask_path = os.path.join(updated_masks_folder, f"{idx:05d}.png")
            cv2.imwrite(convex_hull_mask_path, empty_mask)
            print(
                f"Saved empty convex hull mask for frame {idx}: {convex_hull_mask_path}"
            )

            # Save plain background for warped_from_first
            if background_img is not None:
                warped_frame = background_img.copy()
            else:
                # If no background available, use the first frame as fallback
                warped_frame = first_img.copy()

            warped_path = os.path.join(out_seq_folder, f"{idx:05d}.png")
            rp.save_image(warped_frame, warped_path)
            saved_paths.append(warped_path)
            print(f"Saved plain background for frame {idx}: {warped_path}")

    # Compile video
    out_video_path = os.path.join(
        os.path.dirname(out_seq_folder), "warped_from_first.mp4"
    )
    try:
        rp.save_video_mp4(
            saved_paths, out_video_path, framerate=12, video_bitrate="max"
        )
        print(f"Saved warped video: {out_video_path}")
    except Exception as e:
        print(f"Warning: failed to save video at {out_video_path}: {str(e)}")

    # Compile updated object masks video
    try:
        updated_mask_frames = sorted(
            glob.glob(os.path.join(obj_masks_updated_folder, "*.png"))
        )
        if len(updated_mask_frames) > 0:
            updated_masks_video_path = os.path.join(
                os.path.dirname(out_seq_folder), "obj_masks_updated.mp4"
            )
            rp.save_video_mp4(
                updated_mask_frames,
                updated_masks_video_path,
                framerate=12,
                video_bitrate="max",
            )
            print(f"Saved updated masks video: {updated_masks_video_path}")
        else:
            print(
                f"Warning: no frames found in {obj_masks_updated_folder} to save updated masks video"
            )
    except Exception as e:
        print(
            f"Warning: failed to save updated masks video from {obj_masks_updated_folder}: {str(e)}"
        )

    try:
        updated_masks_folder_frames = sorted(
            glob.glob(os.path.join(updated_masks_folder, "*.png"))
        )
        if len(updated_masks_folder_frames) > 0:
            updated_masks_folder_video_path = os.path.join(
                os.path.dirname(out_seq_folder), "convex_hull_masks.mp4"
            )
            rp.save_video_mp4(
                updated_masks_folder_frames,
                updated_masks_folder_video_path,
                framerate=12,
                video_bitrate="max",
            )
            print(
                f"Saved updated masks folder video: {updated_masks_folder_video_path}"
            )
        else:
            print(
                f"Warning: no frames found in {updated_masks_folder} to save updated masks folder video"
            )
    except Exception as e:
        print(
            f"Warning: failed to save updated masks folder video from {updated_masks_folder}: {str(e)}"
        )


def main(
    selected_vids_file: str,
    input_folder: str,
    output_folder: str,
    first_frame_folder: str,
    mask_firstframe_folder: str,
    newload: bool = False,
) -> None:
    if not os.path.exists(selected_vids_file):
        raise FileNotFoundError(f"Selected vids file not found: {selected_vids_file}")

    with open(selected_vids_file, "r") as f:
        selected_paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(selected_paths)} video paths to process")

    for rel_path in selected_paths:
        process_sequence(
            base_folder=input_folder,
            selected_rel_path=rel_path,
            first_frame_folder=first_frame_folder,
            output_base_folder=output_folder,
            mask_firstframe_folder=mask_firstframe_folder,
            newload=newload,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Warp first frame across sequence using pixel correspondences."
    )
    parser.add_argument(
        "--selected_vids_file",
        type=str,
        required=True,
        help="Selected vids file with relative paths",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Base input folder containing sequences",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Base output folder"
    )
    parser.add_argument(
        "--first_frame_folder",
        type=str,
        required=True,
        help="Folder with first-frame PNGs",
    )
    parser.add_argument(
        "--mask_firstframe_folder",
        type=str,
        required=False,
        default="",
        help="Unused here; reserved for parity",
    )
    parser.add_argument(
        "--newload",
        action="store_true",
        default=False,
        help="Whether to load the paths the new way",
    )
    args = parser.parse_args()

    main(
        selected_vids_file=args.selected_vids_file,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        first_frame_folder=args.first_frame_folder,
        mask_firstframe_folder=args.mask_firstframe_folder,
        newload=args.newload,
    )
