import numpy as np


def homogenize_points(points: np.ndarray) -> np.ndarray:
    """Convert batched points (..., dim) to homogeneous (..., dim+1) by appending 1."""
    ones = np.ones_like(points[..., :1])
    return np.concatenate([points, ones], axis=-1)


def homogenize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Convert batched vectors (..., dim) to homogeneous (..., dim+1) by appending 0."""
    zeros = np.zeros_like(vectors[..., :1])
    return np.concatenate([vectors, zeros], axis=-1)


def transform_rigid(
    homogeneous_coordinates: np.ndarray, transformation: np.ndarray
) -> np.ndarray:
    """
    Applies a rigid-body (affine) transformation (rotation and translation)
    to points or vectors in homogeneous coordinates.
    Shapes:
      homogeneous_coordinates: (..., D)
      transformation:          (..., D, D)
    Returns:                   (..., D)
    """
    return np.einsum("...ij,...j->...i", transformation, homogeneous_coordinates)


def transform_cam2world(
    homogeneous_coordinates: np.ndarray, extrinsics: np.ndarray
) -> np.ndarray:
    """
    Transforms points from 3D camera coordinates to 3D world coordinates.

    NOTE: `extrinsics` should be the camera-to-world transformation matrix.
    """
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: np.ndarray, extrinsics: np.ndarray
) -> np.ndarray:
    """
    Transforms points from 3D world coordinates to 3D camera coordinates.

    NOTE: `extrinsics` should be the camera-to-world transformation matrix.
    """
    inv_extrinsics = np.linalg.inv(extrinsics)
    return transform_rigid(homogeneous_coordinates, inv_extrinsics)


def project_camera_space(
    points: np.ndarray,
    intrinsics: np.ndarray,
    epsilon: float = np.finfo(np.float32).eps,
    infinity: float = 1e8,
) -> np.ndarray:
    """
    Projects points from 3D camera space onto the 2D image plane using the intrinsic matrix.

    Args:
      points:     (..., 3) camera-space points (x, y, z)
      intrinsics: (..., 3, 3) intrinsic matrices
    Returns:
      (..., 2) image-plane coordinates
    """
    # Normalize by z (avoid division by zero)
    points = points / (
        points[
            ...,
            -1:,
        ]
        + epsilon
    )
    # Replace NaN/Inf that may arise from extremely small/negative z
    points = np.nan_to_num(points, posinf=infinity, neginf=-infinity)
    # Apply intrinsics
    proj = np.einsum("...ij,...j->...i", intrinsics, points)
    return proj[..., :-1]  # drop homogeneous 1


def project(
    points: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    epsilon: float = np.finfo(np.float32).eps,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts 3D points in world space to 2D image coordinates and also indicates
    if each point is in front of the camera.

    Args:
      points:     (..., 3) world-space points
      extrinsics: (..., 4, 4) camera-to-world matrices
      intrinsics: (..., 3, 3) intrinsic matrices
    Returns:
      xy:                  (..., 2) image coordinates
      in_front_of_camera:  (...,)   boolean mask (z_cam >= 0)
    """
    # World -> homogeneous
    pts_h = homogenize_points(points)  # (..., 4)
    # World -> Camera (drop homogeneous w to get xyz in camera space)
    cam_xyz = transform_world2cam(pts_h, extrinsics)[..., :-1]  # (..., 3)
    # Visibility: z >= 0 (in front of camera)
    z_depth = cam_xyz[..., -1]
    # Camera -> Image
    xy = project_camera_space(cam_xyz, intrinsics, epsilon=epsilon)
    return xy, z_depth
