import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import taichi as ti
import trimesh
from rich import print
from tqdm import tqdm

from .taichi_mpm import MPMSolver, load_mesh_obj
from .physics_query import get_physics_data
from ..constants import (
    OUT_PERCEPTION_DIR,
    OUT_SIMULATION_DIR,
    VIPE_EXPORT_DIR,
    INPUT_META_DIR,
)


def get_gpu_memory():
    """Check GPU memory usage."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TORCH: GPU not available")
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1024**3
    total_gb = total / 1024**3
    print(f"TORCH: GPU memory usage: {free_gb:.2f} GB / {total_gb:.2f} GB")
    if free_gb / total_gb < 0.8:
        print(
            f"TORCH: GPU memory usage pre-simulation is too high: {free_gb:.2f} GB / {total_gb:.2f} GB"
        )
        raise RuntimeError("TORCH: GPU memory usage pre-simulation is too high")

    return int(free_gb - 0.8)


@dataclass
class SimulationConfig:
    """Configuration class for MPM simulation parameters."""

    # Basic simulation parameters
    device: str = "gpu"  # "gpu" or "cpu"
    config_file: str = None
    sample_id: str = None
    perception_dir: str = None
    output_dir: str = None

    filling: bool = True
    filling_density: int = 8
    pull_to_ground: bool = False
    save_video: bool = False
    with_gui: bool = False
    with_ggui: bool = False

    # Physics parameters
    damping: float = 2.0
    friction: float = 0.5
    R: int = 256
    size: int = 2
    scale: float = 1.0
    max_num_particles: int = 2**20

    # Young's modulus (Pa)
    E_default: float = 1.0 * 1e6
    E_factor_filling: float = 0.5
    # Density (kg/m^3)
    density_default: float = 1e3
    # rebound from surface
    extra_rebound: float = 1.0

    # Output settings
    num_frames: int = 50
    save_point_cloud: bool = True

    # Spinning parameters
    obj_rotation_axis: str = "-x"
    rot_speed: float = 2.0
    FPS: int = 25
    fps_factor: int = 4

    velocities: np.ndarray = None  # not in use
    angular_velocity: np.ndarray = None  # not in use
    linear_velocity_scale: float = 1.0  # not in use

    def __post_init__(self):
        if self.config_file is not None:
            assert Path(
                self.config_file
            ).exists(), f"Config file does not exist: {self.config_file}"
            _configs = json.load(open(self.config_file))
            # use configs to override self
            for key, value in _configs.items():
                if key == "output_dir":
                    continue
                setattr(self, key, value)

        if self.sample_id is None:
            assert self.perception_dir is not None, "Input path must be provided"
            assert self.output_dir is not None, "Output directory must be provided"
        else:
            if not self.perception_dir:
                self.perception_dir = OUT_PERCEPTION_DIR / self.sample_id
            if not self.output_dir:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.output_dir = (
                    OUT_SIMULATION_DIR / self.sample_id / f"v3_debug_{timestamp}"
                )
                if self.output_dir.exists():
                    raise RuntimeError(
                        f"Output directory already exists: {self.output_dir}"
                    )
                self.output_dir.mkdir(parents=True, exist_ok=True)

            # Validate input path
            assert Path(
                self.perception_dir
            ).exists(), f"Input path does not exist: {self.perception_dir}"
            self.perception_dir = str(self.perception_dir)
            self.output_dir = str(self.output_dir)

        backup_dir = Path(self.output_dir) / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        self.freeze_config(backup_dir / "config.json")
        self.freeze_simulation_script(backup_dir / "simulation.py")

    def print_to_console(self):
        """Print the configuration to console."""
        params = self.__dict__
        print("=" * 100)
        for key, value in params.items():
            print(f"{key}: {value}")
        print("=" * 100)

    def freeze_config(self, save_path: Path):

        def default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(save_path, "w") as f:
            json.dump(self.__dict__, f, indent=4, default=default)

    def freeze_simulation_script(self, save_path: Path):
        script_path = Path(__file__)
        script_content = script_path.read_text()
        with open(save_path, "w") as f:
            f.write(script_content)


class SimulationRenderer:
    """Handles visualization and rendering for the simulation."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.gui = None
        self.window = None
        self.canvas = None
        self.scene = None
        self.camera = None
        self.video_writer = None

        # Material colors for visualization
        self.material_colors = np.array(
            [
                [0.1, 0.1, 1.0, 0.8],
                [236.0 / 255.0, 84.0 / 255.0, 59.0 / 255.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        )

        self._setup_gui()
        self._setup_video_writer()

    def _setup_gui(self):
        """Initialize GUI components."""
        if self.config.with_gui:
            self.gui = ti.GUI(
                "MLS-MPM", res=512, background_color=0x112F41, show_gui=True
            )

        if self.config.with_ggui:
            res = (512, 512)
            self.window = ti.ui.Window("Real MPM 3D", res, vsync=True)
            self.canvas = self.window.get_canvas()
            self.scene = ti.ui.Scene()
            self.camera = ti.ui.make_camera()
            self._setup_camera()

    def _setup_camera(self):
        """Configure camera settings for GGUI."""
        if self.camera:
            self.camera.position(5, 5, 5)
            self.camera.lookat(5, 5, 10)
            self.camera.up(0, -1, 0)
            self.camera.fov(100)

    def _setup_video_writer(self):
        """Initialize video writer if saving video."""
        if self.config.save_video:
            video_files = glob.glob(os.path.join("sim_result", "*.mp4"))
            next_index = len(video_files) + 1
            video_output = os.path.join("sim_result", f"{next_index}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 24
            self.video_writer = cv2.VideoWriter(video_output, fourcc, fps, (512, 512))

    @ti.kernel
    def _set_color(
        self,
        ti_color: ti.template(),
        material_color: ti.types.ndarray(),
        ti_material: ti.template(),
    ):
        """Kernel to set particle colors based on material type."""
        for I in ti.grouped(ti_material):
            material_id = ti_material[I]
            color_4d = ti.Vector([0.0, 0.0, 0.0, 1.0])
            for d in ti.static(range(3)):
                color_4d[d] = material_color[material_id, d]
            ti_color[I] = color_4d

    def render_ggui(self, mpm):
        """Render using GGUI (3D rendering)."""
        if not self.config.with_ggui or not self.window:
            return

        self.camera.track_user_inputs(
            self.window, movement_speed=0.03, hold_key=ti.ui.RMB
        )
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0, 0, 0))

        self._set_color(mpm.color_with_alpha, self.material_colors, mpm.material)

        self.scene.particles(mpm.x, per_vertex_color=mpm.color_with_alpha, radius=0.02)

        # Add lighting
        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

        self.canvas.scene(self.scene)
        self.window.show()

    def render_gui(self, particles: Dict) -> Optional[np.ndarray]:
        """Render using traditional GUI (2D projection)."""
        if not self.config.with_gui or not self.gui:
            return None

        np_x = particles["position"] / self.config.size
        screen_x = np_x[:, 0]
        screen_y = 1 - np_x[:, 2]
        screen_pos = np.stack([screen_x, screen_y], axis=-1)

        self.gui.circles(screen_pos, radius=1.5, color=particles["color"])

        if self.config.save_video:
            img = self._capture_frame()
            self.gui.clear()
            return img
        else:
            self.gui.clear()
            return None

    def _capture_frame(self) -> np.ndarray:
        """Capture current frame as image."""
        img = self.gui.get_image()
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        img = img[:, :, :3]  # Remove alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def write_video_frame(self, frame: np.ndarray):
        """Write frame to video file."""
        if self.video_writer and frame is not None:
            self.video_writer.write(frame)

    def cleanup(self):
        """Release resources."""
        if self.video_writer:
            self.video_writer.release()


class ParticleManager:
    """Manages particle operations and mesh loading."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.center = np.array([config.size / 2, config.size / 2, config.size / 2])

    def transform_pcd(self, pcd: np.ndarray, c2w: np.ndarray) -> np.ndarray:
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

    def normalize_mesh(
        self,
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

    def load_scene_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:

        sample_id = self.config.sample_id
        vipe_exported_dir = VIPE_EXPORT_DIR / sample_id
        video_info_dir = vipe_exported_dir / "video_info"
        video_info_path = video_info_dir / "video_info.json"
        assert (
            video_info_path.exists()
        ), f"Video info path does not exist: {video_info_path}"

        # e.g. data_root/OUT_ViPE_Export/2_basketball/frames_info/00000/00000.json
        frame0_info_path = vipe_exported_dir / "frames_info" / "00000" / "00000.json"
        assert (
            frame0_info_path.exists()
        ), f"Frame 0 info path does not exist: {frame0_info_path}"

        video_info = json.load(open(video_info_path))
        frame0_info = json.load(open(frame0_info_path))

        ### Load ground plane
        plane_model = video_info["plane_model"]
        plane_normal = np.array(video_info["plane_normal"])
        plane_point = np.array([0, 0, -plane_model[3] / plane_model[2]])
        gravity_vector = np.array(video_info["gravity_vector"])
        domain_scale = video_info["domain_scale"]
        N_frames = video_info["N_frames"]

        ### Load primary object
        primary_obj_name = frame0_info.get("primary_obj_name")
        assert (
            primary_obj_name is not None
        ), "Primary object is not found in frame 0 info"
        primary_obj_centroid = np.array(frame0_info["primary_obj_centroid"])

        ### Load secondary object (if exists)
        secondary_obj_name = frame0_info.get("secondary_obj_name")
        secondary_obj_centroid = (
            np.array(frame0_info["secondary_obj_centroid"])
            if secondary_obj_name
            else None
        )

        ### Default physics
        physics_data = get_physics_data(sample_id)
        default_physics = physics_data.get("default")

        ### Load primary object mesh
        _name = primary_obj_name.replace(" ", "_")
        primary_obj_mesh_path = video_info_dir / "size_estimate_0" / f"{_name}.obj"
        assert (
            primary_obj_mesh_path.exists()
        ), f"Primary object mesh path does not exist: {primary_obj_mesh_path}"
        primary_obj_mesh = trimesh.load(primary_obj_mesh_path)
        primary_obj_physics = physics_data.get(primary_obj_name, default_physics)
        primary_obj_physics = primary_obj_physics.to_mpm()
        self.extra_rebound = primary_obj_physics["extra_rebound"]

        ### Load secondary object mesh (if exists)
        if secondary_obj_name:
            _name = secondary_obj_name.replace(" ", "_")
            secondary_obj_mesh_path = (
                video_info_dir / "size_estimate_1" / f"{_name}.obj"
            )
            assert (
                secondary_obj_mesh_path.exists()
            ), f"Secondary object mesh path does not exist: {secondary_obj_mesh_path}"
            # secondary_obj_mesh = trimesh.load(secondary_obj_mesh_path)
            secondary_obj_physics = physics_data.get(
                secondary_obj_name, default_physics
            )
            secondary_obj_physics = secondary_obj_physics.to_mpm()
        else:
            secondary_obj_mesh_path = None
            secondary_obj_physics = None

        ###############################################################################

        # compute instantaneous linear velocity vector for object
        key_frame_idx = 5
        FPS = int(self.config.FPS)
        key_frame_info_path = (
            vipe_exported_dir
            / "frames_info"
            / f"{key_frame_idx:05d}"
            / f"{key_frame_idx:05d}.json"
        )
        key_frame_info = json.load(open(key_frame_info_path))

        delta_d = np.array(key_frame_info["primary_obj_displacement"])
        delta_t = 1.0 / FPS * key_frame_idx
        linear_velocity = delta_d / delta_t
        if "multi" in sample_id:
            linear_velocity *= 2.0

        def build_rot_axis(obj_rotation_axis: str) -> np.ndarray | None:
            obj_rotation_axis = str(obj_rotation_axis).lower()
            if len(obj_rotation_axis) < 1 or len(obj_rotation_axis) > 2:
                return None
            _axis_val = -1.0 if "-" in obj_rotation_axis else 1.0
            if "x" in obj_rotation_axis:
                return np.array([_axis_val, 0.0, 0.0])
            elif "y" in obj_rotation_axis:
                return np.array([0.0, _axis_val, 0.0])
            elif "z" in obj_rotation_axis:
                return np.array([0.0, 0.0, _axis_val])
            else:
                return None

        rot_axis_fallback = np.array([-1.0, 0.0, 0.0])
        rot_axis = build_rot_axis(self.config.obj_rotation_axis)
        if rot_axis is None:
            rot_axis = rot_axis_fallback
            rot_speed = 0.0

        rot_speed = self.config.rot_speed

        return dict(
            primary_obj_name=primary_obj_name,
            primary_obj_mesh_path=primary_obj_mesh_path,
            primary_obj_centroid=primary_obj_centroid,
            primary_obj_physics=primary_obj_physics,
            secondary_obj_name=secondary_obj_name,
            secondary_obj_mesh_path=secondary_obj_mesh_path,
            secondary_obj_centroid=secondary_obj_centroid,
            secondary_obj_physics=secondary_obj_physics,
            plane_normal=plane_normal,
            plane_point=plane_point,
            gravity=gravity_vector,
            domain_scale=domain_scale,
            n_surface_particles=len(primary_obj_mesh.vertices),
            linear_velocity=linear_velocity,
            rot_points=primary_obj_mesh.vertices,
            rot_center=primary_obj_centroid,
            rot_axis=rot_axis,
            rot_speed=rot_speed,  # rad/s
            N_frames=N_frames,
        )

    def load_mesh_data(
        self, mesh_file: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and transform mesh data."""
        # Setup path and import here to avoid circular imports
        # project_dir = os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), os.pardir)
        # )
        # sys.path.append(project_dir)
        # from engine.mesh_io import load_mesh_obj

        vertices, triangles, colors = load_mesh_obj(str(mesh_file), offset=[0, 0, 0])

        return vertices, triangles, colors

    def velocities_from_rotation(
        self,
        rot_points,
        rot_center,
        rot_axis,
        rot_speed,
        units="rad/s",
        **kwargs,
    ):
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
        P = np.asarray(rot_points, dtype=float)
        C = np.asarray(rot_center, dtype=float).reshape(3)
        a = np.asarray(rot_axis, dtype=float).reshape(3)

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
            S = float(rot_speed)
        elif units == "deg/s":
            S = np.deg2rad(float(rot_speed))
        elif units == "rpm":
            S = float(rot_speed) * 2.0 * np.pi / 60.0
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

    @staticmethod
    def convert_colors(colors: np.ndarray) -> np.ndarray:
        """Convert colors to decimal format for rendering."""
        return np.zeros_like(colors)[:, 0]
        dec_colors = (colors * 255).astype(np.uint8)
        return dec_colors[:, 0] * 65536 + dec_colors[:, 1] * 256 + dec_colors[:, 2]


class MPMSimulation:
    """Main simulation class that orchestrates the MPM simulation."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.renderer = SimulationRenderer(config)
        self.particle_manager = ParticleManager(config)
        self.mpm = None
        self.output_dir = None
        self.extra_rebound = self.config.extra_rebound
        self.metadata = {}

        # self.config.print_to_console()

        self._create_output_directory()
        self._initialize_taichi()

    def _initialize_taichi(self):
        """Initialize Taichi with appropriate settings."""
        if self.config.device == "cpu":
            ti.init(arch=ti.cpu)
            print("[green bold]Initialized Taichi with CPU\n\n")
        else:
            device_memory_GB = get_gpu_memory()
            print(f"device_memory_GB: {device_memory_GB}")
            ti.init(
                arch=ti.gpu,
                device_memory_GB=device_memory_GB,
                debug=False,
                # random_seed=42,
            )
            print(
                f"[green bold]Initialized Taichi with {device_memory_GB} GB of GPU memory\n\n"
            )

    def _create_output_directory(self):
        """Create output directory for simulation results."""
        if self.config.output_dir is None:
            raise ValueError("Output directory not configured")
        else:
            self.output_dir = Path(self.config.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_point_cloud:
            self.point_cloud_dir = self.output_dir / "point_cloud"
            self.point_cloud_dir.mkdir(parents=True, exist_ok=True)

        self.log_file_path = self.output_dir / "log.txt"
        self.log_file = open(self.log_file_path, "w")

    def log(self, message: str):
        """Log a message to the log file."""
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()
        else:
            print(
                "[yellow]Warn: Log file not initialized, printing to console instead."
            )
            print(message)

    def _generate_unique_colors(self, num_points: int) -> np.ndarray:
        """Generate unique colors for each point based on its index using a spectrum.

        Args:
            num_points: Number of points to generate colors for

        Returns:
            Array of shape (num_points, 3) with RGB colors in range [0, 1]
        """
        colors = np.zeros((num_points, 3))

        for i in range(num_points):
            # Use a spectrum-based approach with HSV color space
            # Map index to hue value (0 to 1) for a full spectrum
            hue = i / max(1, num_points - 1)  # Normalize to [0, 1]

            # Convert HSV to RGB
            # Using a rainbow spectrum: red -> orange -> yellow -> green -> blue -> purple
            # This creates a smooth color transition across all points

            # HSV to RGB conversion
            h = hue * 6  # Scale to [0, 6]
            c = 1.0  # Chroma (saturation)
            x = c * (1 - abs(h % 2 - 1))
            m = 0.0  # Value offset

            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors[i] = [r, g, b]

        return colors

    def _setup_mpm_solver(self):
        """Initialize the MPM solver with configuration."""
        self.mpm = MPMSolver(
            res=(self.config.R, self.config.R, self.config.R),
            quant=False,  # by default it was True
            size=self.config.size,
            unbounded=False,
            dt_scale=1,
            E_scale=1,
            use_diff_E=True,
            use_adaptive_dt=True,
            support_plasticity=True,
            use_ggui=self.config.with_ggui,
            max_num_particles=self.config.max_num_particles,
            drag_damping=self.config.damping,
        )

    def _setup_physics_environment(
        self,
        gravity: np.ndarray,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
        **kwargs,
    ):
        """Setup gravity and colliders."""
        self.log(f"Setting gravity: {gravity}")
        self.mpm.set_gravity(gravity.tolist())

        self.mpm.add_surface_collider(
            point=plane_point,
            normal=plane_normal,
            surface=self.mpm.surface_separate,
            friction=self.config.friction,
            rebound=self.extra_rebound,
        )

    def add_surface_particles(
        self,
        obj_mesh_path: str,
        E: float,
        rho: float,
        velocities: np.ndarray = None,
        triangles_list: list = [],
        material=MPMSolver.material_elastic,
        **kwargs,
    ) -> List[np.ndarray]:
        """Add surface particles for all objects."""
        # Load and process mesh
        vertices, triangles, colors = self.particle_manager.load_mesh_data(
            obj_mesh_path
        )
        if velocities is None:
            velocities = np.zeros((len(vertices), 3))
        if len(velocities) > len(vertices):
            # NOTE: this is a temporary fix to avoid unmatched number of particles and velocities
            # the velocities points are from trimesh vertices
            # the vertices are from load_mesh_data
            # it is wierd that only some times, it has a few points difference
            velocities = velocities[: len(vertices)]

        self.mpm.add_particles_with_velocities(
            particles=np.ascontiguousarray(vertices),
            velocities=np.ascontiguousarray(velocities),
            colors=self.particle_manager.convert_colors(colors),
            material=material,
            E=E,
            nu=0.2,
            rho=rho,
        )
        triangles_list.append(triangles)
        return triangles_list

    def add_volume_particles(
        self,
        triangles_list: List[np.ndarray],
        E: float,
        sample_density: int,
        material=MPMSolver.material_elastic,
    ):
        dummy_velocity = np.array([0.0, 0.0, 0.0])
        particles = self.mpm.particle_info()
        start_index = len(particles["position"])
        self.mpm.add_mesh(
            triangles=triangles_list[-1],
            material=material,
            sample_density=sample_density,
            velocity=dummy_velocity,
            E=E,
            nu=0.2,
            rho=1000,
        )
        particles = self.mpm.particle_info()
        end_index = len(particles["position"])
        filled_xyzs = particles["position"][start_index:end_index]

        return filled_xyzs, start_index, end_index

    def _save_metadata(self):
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

    def _run_simulation_loop(self, num_frames: int, fps: int):
        """Execute the main simulation loop."""
        start_time = time.time()
        SAVE_POINTS = True
        _idx_colors = None
        particles_dir = self.output_dir / "particles"
        particles_dir.mkdir(parents=True, exist_ok=True)
        dt = 1.0 / fps

        for frame in tqdm(range(num_frames), desc="Simulation frames"):
            frame_start = time.time()

            # Render and save if needed
            if self.config.with_gui and frame % 1 == 0:
                particles = self.mpm.particle_info()
                image = self.renderer.render_gui(particles)

                if image is not None:
                    self.renderer.write_video_frame(image)
                    # Save frame and point cloud periodically
                    if frame % 5 == 0:
                        cv2.imwrite(f"sim_result/frames/frame_{frame:04d}.png", image)

            if SAVE_POINTS:
                info = self.mpm.particle_info()
                particles3D = info["position"]
                if _idx_colors is None:
                    _N_points = len(particles3D)
                    _idx_colors = self._generate_unique_colors(_N_points)
                point_cloud = trimesh.points.PointCloud(particles3D, colors=_idx_colors)
                point_cloud_save_path = self.point_cloud_dir / f"{frame:05d}.ply"
                point_cloud.export(point_cloud_save_path)

            # Save particle data
            particle_save_path = particles_dir / f"{frame:05d}.npz"
            self.mpm.write_particles(particle_save_path)

            # Render GGUI
            if self.config.with_ggui:
                self.renderer.render_ggui(self.mpm)

            # Step simulation
            # self.mpm.step(1e-2, print_stat=False)
            self.mpm.step(dt, print_stat=False)

            frame_time = time.time() - frame_start
            if frame % 20 == 0:
                total_time = time.time() - start_time
                self.log(f"Frame {frame}: {frame_time:.3f}s, Total: {total_time:.3f}s")

    def run(self):
        """Execute the complete simulation."""
        self.log("Starting MPM simulation...")
        self.log(
            f"Configuration: filling={self.config.filling}, R={self.config.R}, size={self.config.size}"
        )

        # Load scene data
        scene_data = self.particle_manager.load_scene_data()

        # Compute vertices-wise velocities
        linear_velocity = scene_data["linear_velocity"]  # (3, )
        n_surface_particles = scene_data["n_surface_particles"]  # int
        linear_vels = np.tile(linear_velocity, (n_surface_particles, 1))  # (N, 3)
        if scene_data["rot_speed"] > 0:
            # compute rotational velocities
            rot_vels = self.particle_manager.velocities_from_rotation(
                rot_points=scene_data["rot_points"],
                rot_center=scene_data["rot_center"],
                rot_axis=scene_data["rot_axis"],
                rot_speed=scene_data["rot_speed"],
            )
            assert linear_vels.shape == rot_vels.shape, "Unexpected shape error"
            # combines linear and rotational velocities
            velocities = linear_vels + rot_vels
        else:
            velocities = linear_vels

        # Initialize MPM solver class
        self._setup_mpm_solver()
        # Setup gravity and set ground plane as a surface collider
        self._setup_physics_environment(**scene_data)

        primary_obj_physics = scene_data["primary_obj_physics"]
        triangles_list = []
        try:
            # Add particles for primary object
            triangles_list = self.add_surface_particles(
                obj_mesh_path=scene_data["primary_obj_mesh_path"],
                velocities=velocities,
                E=primary_obj_physics["E"] * 1_000_000,  # convert MPa to Pa
                rho=primary_obj_physics["rho"],
                triangles_list=triangles_list,
                material=MPMSolver.material_elastic,
            )
        except Exception as e:
            print(f"Error in self.add_surface_particles: {e}")
            print(f"n_surface_particles: {n_surface_particles}")
            print(f"velocities.shape: {velocities.shape}")
            raise e

        if self.config.filling:
            print("Adding volume filling particles for primary object...")
            filled_xyzs, start_index, end_index = self.add_volume_particles(
                sample_density=self.config.filling_density,
                E=primary_obj_physics["E"] * self.config.E_factor_filling * 1_000_000,
                triangles_list=triangles_list,
                material=MPMSolver.material_elastic,
            )
            n_volume_particles = end_index - start_index
            self.metadata["primary_n_surface_particles"] = n_surface_particles
            self.metadata["primary_n_volume_particles"] = n_volume_particles
            self.metadata["primary_start_index"] = start_index
            self.metadata["primary_end_index"] = end_index

            print(f"Number of Surface Particles: {n_surface_particles}")
            print(f"Number of Volume Particles : {n_volume_particles}")
            vol_linear_vels = np.tile(linear_velocity, (n_volume_particles, 1))
            vol_rot_vels = self.particle_manager.velocities_from_rotation(
                rot_points=filled_xyzs,
                rot_center=scene_data["rot_center"],
                rot_axis=scene_data["rot_axis"],
                rot_speed=scene_data["rot_speed"],
            )
            vol_vels = vol_linear_vels + vol_rot_vels
            # overwrite velocities
            self.mpm.overwrite_velocities(
                start_index=start_index,
                end_index=end_index,
                velocities=vol_vels,
            )
        else:
            print("No volume filling is performed.")

        #######################################################################
        ### Add secondary object if exists

        secondary_obj_mesh_path = scene_data["secondary_obj_mesh_path"]
        secondary_obj_physics = scene_data["secondary_obj_physics"]
        if None not in (secondary_obj_mesh_path, secondary_obj_physics):
            try:
                # Add particles for secondary object
                triangles_list = self.add_surface_particles(
                    obj_mesh_path=secondary_obj_mesh_path,
                    E=secondary_obj_physics["E"] * 1_000_000,
                    rho=secondary_obj_physics["rho"],
                )
            except Exception as e:
                print(f"Error when adding particles for secondary object: {e}")
                raise e

            if self.config.filling:
                print("Adding volume filling particles for secondary object...")
                filled_xyzs, start_index, end_index = self.add_volume_particles(
                    sample_density=self.config.filling_density,
                    E=secondary_obj_physics["E"]
                    * self.config.E_factor_filling
                    * 1_000_000,
                    triangles_list=triangles_list,
                    material=MPMSolver.material_elastic,
                )
                n_volume_particles = end_index - start_index
                _n_surface_particles = start_index - self.metadata["primary_end_index"]
                self.metadata["secondary_n_surface_particles"] = _n_surface_particles
                self.metadata["secondary_n_volume_particles"] = n_volume_particles
                self.metadata["secondary_start_index"] = start_index
                self.metadata["secondary_end_index"] = end_index

                print(f"Number of Surface Particles: {_n_surface_particles}")
                print(f"Number of Volume Particles : {n_volume_particles}")

        # Run simulation
        num_frames = scene_data.get("N_frames", int(self.config.num_frames))
        if self.config.fps_factor is not None:
            fps_factor = int(self.config.fps_factor)
            if fps_factor not in [1, 2, 3, 4, 8]:
                fps_factor = 1

            num_frames = num_frames * fps_factor
            fps = int(self.config.FPS) * fps_factor

        self._run_simulation_loop(num_frames=num_frames, fps=fps)

        self._save_metadata()

        # Cleanup
        self.renderer.cleanup()

        self.log(f"Simulation completed. Results saved to: {self.output_dir}")

        success_file = self.output_dir / "success.txt"
        success_file.touch()


def run_mpm_simulation(overwrite: bool = False, **kwargs):
    sample_id = kwargs["sample_id"]
    meta_json_path = INPUT_META_DIR / f"{sample_id}.json"
    metadata = json.load(open(meta_json_path)) if meta_json_path.exists() else {}
    rot_axis = metadata.get("primary_obj_rot_axis")

    config = SimulationConfig(**kwargs)
    if rot_axis is not None:
        config.obj_rotation_axis = rot_axis
    success_file = Path(config.output_dir) / "success.txt"
    if success_file.exists() and not overwrite:
        print(
            f"  [yellow]✔ Skipped[/yellow] {config.sample_id} because simulation result exists"
        )
        return

    simulation = MPMSimulation(config)
    simulation.run()


def main():
    parser = argparse.ArgumentParser(description="Run MPM simulation with config file")
    parser.add_argument("--config_file", type=str, help="Path to configuration file")
    parser.add_argument("--sample_id", type=str, help="Sample ID")
    parser.add_argument("--perception_dir", type=str, help="Perception directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--device", type=str, help="Device", default="cpu", choices=["cpu", "gpu"]
    )

    parser.add_argument("--rot_speed", type=float, help="Rotation speed", default=5.0)
    parser.add_argument(
        "--obj_rotation_axis",
        type=str,
        help="Rotation axis choice (x, y, z, x-, y-, z-)",
        default="-x",
    )
    parser.add_argument("--damping", type=float, help="Damping", default=2.5)
    parser.add_argument("--friction", type=float, help="Friction", default=2.5)
    parser.add_argument(
        "--num_frames", type=int, help="Number of steps to simulate", default=50
    )
    parser.add_argument(
        "--e_value", type=float, help="Young's modulus (MPa)", default=0.5
    )
    parser.add_argument("--fps_factor", type=int, help="FPS factor", default=1)
    args = parser.parse_args()

    E_default = args.e_value * 1e6  # convert MPa to Pa

    # Create config and run simulation
    config = SimulationConfig(
        device=args.device,
        config_file=args.config_file,
        sample_id=args.sample_id,
        perception_dir=args.perception_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        E_default=E_default,
        rot_speed=args.rot_speed,
        damping=args.damping,
        friction=args.friction,
        obj_rotation_axis=args.obj_rotation_axis,
        fps_factor=args.fps_factor,
    )
    simulation = MPMSimulation(config)
    simulation.run()


if __name__ == "__main__":
    main()
