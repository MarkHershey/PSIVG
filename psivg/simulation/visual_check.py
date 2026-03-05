import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import viser
from psivg.constants import OUT_SIMULATION_DIR


def _load_ply_as_points_colors(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a .ply point cloud and return:
      points: (N, 3) float32
      colors: (N, 3) uint8 in [0, 255] or None

    Uses trimesh first; optionally falls back to open3d if trimesh can't parse.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # --- Try trimesh ---
    try:
        geom = trimesh.load(
            path, process=False
        )  # don't merge/repair, keep raw vertices/colors
        # trimesh may return a Scene, PointCloud, Trimesh, or other geometry.
        if isinstance(geom, trimesh.Scene):
            # Grab first geometry in the scene.
            if len(geom.geometry) == 0:
                raise ValueError(f"No geometry found in scene for {path}")
            geom = next(iter(geom.geometry.values()))

        points = None
        colors = None

        if isinstance(geom, trimesh.points.PointCloud):
            points = np.asarray(geom.vertices)
            if geom.colors is not None and len(geom.colors) == len(points):
                # trimesh stores colors as RGBA uint8 typically
                c = np.asarray(geom.colors)
                colors = c[:, :3]  # RGB
        elif isinstance(geom, trimesh.Trimesh):
            # Some exporters store point clouds as vertices in a mesh
            points = np.asarray(geom.vertices)
            if geom.visual is not None and hasattr(geom.visual, "vertex_colors"):
                c = np.asarray(geom.visual.vertex_colors)
                if c is not None and len(c) == len(points):
                    colors = c[:, :3]
        else:
            # Try generic vertices attr if present
            if hasattr(geom, "vertices"):
                points = np.asarray(getattr(geom, "vertices"))
                if hasattr(geom, "colors"):
                    c = np.asarray(getattr(geom, "colors"))
                    if c is not None and len(c) == len(points):
                        colors = c[:, :3]

        if points is None or points.size == 0:
            raise ValueError(
                f"Loaded geometry has no vertices: {type(geom)} for {path}"
            )

        points = points.astype(np.float32)
        if colors is not None:
            colors = colors.astype(np.uint8)

        return points, colors

    except Exception as e_trimesh:
        # --- Optional fallback: Open3D ---
        try:
            import open3d as o3d  # type: ignore

            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points, dtype=np.float32)
            if pts.size == 0:
                raise ValueError("Open3D read empty point cloud")

            cols = None
            if len(pcd.colors) == len(pts):
                c = np.asarray(pcd.colors, dtype=np.float32)  # 0..1
                cols = np.clip(c * 255.0, 0, 255).astype(np.uint8)

            return pts, cols
        except Exception:
            raise RuntimeError(
                f"Failed to load PLY with trimesh ({e_trimesh}) and Open3D fallback."
            ) from e_trimesh


def run_viewer(
    ply_paths, host="0.0.0.0", port=8080, default_fps=60.0, point_size=0.0001
):
    # (sorting + initial load omitted for brevity — keep what you had)
    # Assume you already created:
    #   server = viser.ViserServer(...)
    #   pcd_handle = server.scene.add_point_cloud(...)

    server = viser.ViserServer(host=host, port=port)

    server.scene.set_up_direction(np.array([0, 1, 0]))
    # server.scene.world_axes.visible = True
    server.scene.dark_mode = True

    # Preload first frame
    pts0, cols0 = _load_ply_as_points_colors(ply_paths[0])
    colors0_f = (cols0.astype(np.float32) / 255.0) if cols0 is not None else None
    pcd_handle = server.scene.add_point_cloud(
        name="/pointcloud",
        points=pts0,
        colors=colors0_f,
        point_size=point_size,
    )

    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Play", initial_value=True)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        fps = server.gui.add_slider(
            "FPS", min=0.5, max=60.0, step=0.5, initial_value=default_fps
        )
        frame = server.gui.add_slider(
            "Frame", min=0, max=len(ply_paths) - 1, step=1, initial_value=0
        )
        step_back = server.gui.add_button("⏮ Prev")
        step_fwd = server.gui.add_button("Next ⏭")

    status = server.gui.add_text("Status", initial_value="Ready")

    current_idx = 0
    last_update_t = time.perf_counter()

    # --- Guard to prevent recursion when we set frame.value programmatically ---
    suppress_frame_cb = False

    def _set_frame(i: int, *, update_slider: bool = True) -> None:
        nonlocal current_idx, suppress_frame_cb

        i = int(np.clip(i, 0, len(ply_paths) - 1))
        path = ply_paths[i]

        pts, cols = _load_ply_as_points_colors(path)
        cols_f = (cols.astype(np.float32) / 255.0) if cols is not None else None

        # Update point cloud
        pcd_handle.points = pts
        pcd_handle.colors = cols_f  # None is fine

        current_idx = i

        # Update UI slider (WITHOUT re-triggering on_update)
        if update_slider and frame.value != i:
            suppress_frame_cb = True
            try:
                frame.value = i
            finally:
                suppress_frame_cb = False

        status.value = f"{os.path.basename(path)}  (N={len(pts)})"

    @frame.on_update
    def _(_evt):
        nonlocal suppress_frame_cb
        if suppress_frame_cb:
            return
        playing.value = False
        # Don't write frame.value here; user already set it.
        _set_frame(frame.value, update_slider=False)

    @step_back.on_click
    def _(_evt):
        playing.value = False
        _set_frame(current_idx - 1, update_slider=True)

    @step_fwd.on_click
    def _(_evt):
        playing.value = False
        _set_frame(current_idx + 1, update_slider=True)

    _set_frame(0, update_slider=False)

    while True:
        time.sleep(0.001)
        if not playing.value:
            continue

        now = time.perf_counter()
        target_dt = 1.0 / float(fps.value)
        if (now - last_update_t) < target_dt:
            continue
        last_update_t = now

        next_idx = current_idx + 1
        if next_idx >= len(ply_paths):
            if loop.value:
                next_idx = 0
            else:
                playing.value = False
                continue

        _set_frame(next_idx, update_slider=True)


def ply_files_from_dir(dir_path: str) -> List[str]:
    dir_path = Path(dir_path)
    ply_paths = []
    for ply_file in dir_path.glob("*.ply"):
        ply_paths.append(str(ply_file.resolve()))
    return sorted(ply_paths)


if __name__ == "__main__":
    sample_id = "xxxx"
    sim_id = "yyyyyyy"
    ply_paths = ply_files_from_dir(
        OUT_SIMULATION_DIR / sample_id / sim_id / "point_cloud"
    )
    run_viewer(ply_paths=ply_paths)
