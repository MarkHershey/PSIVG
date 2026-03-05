#!/usr/bin/env python3
"""
Velocity field visualization in Viser.

Inputs:
- points:    array shaped (N, 3) or (..., 3)
- velocities:array shaped (N, 3) or (..., 3)
"""
import time

import numpy as np
import viser


# ---------- geometry helpers ----------
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


def visualize_velocity_field(
    points: np.ndarray,
    velocities: np.ndarray,
    *,
    initial_stride: int = 5,
    initial_scale: float = 0.2,
    arrow_geometry_kwargs: dict | None = None,
) -> None:
    """Launch a Viser server and render arrows anchored at points and aligned to velocities."""
    pts = np.asarray(points, dtype=np.float32)
    vels = np.asarray(velocities, dtype=np.float32)
    assert pts.shape[-1] == 3 and vels.shape[-1] == 3, "Last dim must be 3."
    pts = pts.reshape(-1, 3)
    vels = vels.reshape(-1, 3)
    assert pts.shape[0] == vels.shape[0], "points and velocities must have same length."

    # Build the arrow mesh once; instance it N times with per-instance transform/color.
    verts, faces = make_arrow_mesh(**(arrow_geometry_kwargs or {}))

    server = viser.ViserServer(host="0.0.0.0", port=8989)
    server.gui.configure_theme(dark_mode=True)
    server.scene.configure_default_lights()
    server.scene.add_grid(
        "grid", width=10, height=10, width_segments=10, height_segments=10
    )

    # GUI
    with server.gui.add_folder("Velocity Field"):
        stride_slider = server.gui.add_slider(
            "Spatial Stride", min=1, max=50, step=1, initial_value=int(initial_stride)
        )
        scale_slider = server.gui.add_slider(
            "Arrow Scale", min=0.01, max=5.0, step=0.01, initial_value=float(initial_scale)
        )
        show_points = server.gui.add_checkbox("Show Anchors", initial_value=False)
        point_size = server.gui.add_slider(
            "Point Size", min=0.001, max=0.5, step=0.01, initial_value=0.05
        )

    # Create initial downsample
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

    positions, wxyzs, scales, colors = make_instances(
        stride_slider.value, scale_slider.value
    )

    # Add optional point cloud of anchors
    anchors_handle = server.scene.add_point_cloud(
        "anchors",
        points=positions,
        colors=colors,
        point_size=float(point_size.value),
        point_shape="circle",
    )
    anchors_handle.visible = bool(show_points.value)

    # Create batched arrow instances
    arrow_handle = server.scene.add_batched_meshes_simple(
        name="velocity_arrows",
        vertices=verts,
        faces=faces,
        batched_positions=positions,
        batched_wxyzs=wxyzs,
        batched_scales=scales,  # uniform scale per instance
        batched_colors=colors,  # per-instance color
        lod="auto",
    )

    def _upsert_anchors(scene, points, colors, point_size, visible):
        handle = scene.add_point_cloud(
            "anchors",
            points=points,
            colors=colors,
            point_size=float(point_size),
            point_shape="circle",
        )
        handle.visible = bool(visible)
        return handle

    def _update_arrows(arrow_handle, positions, wxyzs, scales, colors) -> None:
        arrow_handle.batched_positions = positions
        arrow_handle.batched_wxyzs = wxyzs
        arrow_handle.batched_scales = scales
        arrow_handle.batched_colors = colors

    # Event-driven GUI: update handlers
    def _recompute_and_update() -> None:
        nonlocal anchors_handle
        stride = int(stride_slider.value)
        scale = float(scale_slider.value)
        positions_i, wxyzs_i, scales_i, colors_i = make_instances(stride, scale)
        with server.atomic():
            _update_arrows(arrow_handle, positions_i, wxyzs_i, scales_i, colors_i)
            anchors_handle = _upsert_anchors(
                server.scene,
                points=positions_i,
                colors=colors_i,
                point_size=point_size.value,
                visible=show_points.value,
            )

    @stride_slider.on_update
    def _(_: object) -> None:
        _recompute_and_update()

    @scale_slider.on_update
    def _(_: object) -> None:
        _recompute_and_update()

    @show_points.on_update
    def _(_: object) -> None:
        anchors_handle.visible = bool(show_points.value)

    @point_size.on_update
    def _(_: object) -> None:
        anchors_handle.point_size = float(point_size.value)

    # Idle loop to keep server alive
    while True:
        time.sleep(10.0)


def demo():
    N = 6000
    pts_demo = np.random.uniform(-2.0, 2.0, size=(N, 3)).astype(np.float32)
    v_demo = np.stack(
        [
            -pts_demo[:, 1],
            pts_demo[:, 0],
            np.ones(N) * 0.5 - 0.2 * pts_demo[:, 2],
        ],
        axis=1,
    ).astype(np.float32)
    visualize_velocity_field(pts_demo, v_demo, initial_stride=6, initial_scale=0.25)


if __name__ == "__main__":
    demo()
