"""Render a sequence of Mitsuba images from per-frame point data.

This module constructs Mitsuba scenes using foreground point clouds (as tiny
colored spheres) and an optional ground plane mesh, rendering each video frame
from the provided camera pose.
"""

import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from rich import print

try:
    import mitsuba as mi

    mi.set_variant("cuda_ad_rgb")
except Exception:
    print("Mitsuba 3 is not installed. Please install mitsuba and try again.")
    mi = None

from ..constants import (
    OUT_PERCEPTION_DIR,
    OUT_RENDERING_DIR,
    OUT_SIMULATION_DIR,
    VIPE_EXPORT_DIR,
)
from .grid_io import load_grid
from .interpolate import interpolate_c2ws
from .make_video import make_video_from_frames_dir
from .particle_io import ParticleIO


def backup_this_script(output_dir: Path, timestamp: str = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = Path(__file__)
    script_content = script_path.read_text()
    with open(output_dir / f"rendering_{timestamp}.py", "w") as f:
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


def build_ground_mesh(video_info: dict, output_dir: Path) -> dict:
    ply_path = build_ground_plane_ply(video_info, output_dir)
    # refl = {"type": "rgb", "value": [0.85, 0.85, 0.85]}
    refl = {"type": "rgb", "value": [0.3, 0.3, 0.2]}
    twosided = {"type": "twosided", "bsdf": {"type": "diffuse", "reflectance": refl}}
    return {"type": "ply", "filename": str(ply_path), "bsdf": twosided}


def build_ground_plane_ply(video_info: dict, output_dir: Path) -> str:
    ply_path = Path(output_dir) / "plane.ply"
    if ply_path.exists():
        return str(ply_path)
    verts = np.asarray(video_info["plane_vertices"], dtype=np.float32)
    faces = np.asarray(video_info["plane_faces"], dtype=np.int32)
    assert verts.shape == (4, 3) and faces.shape == (2, 3)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_colors=False)
    return str(ply_path)


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


def _load_foreground_points_for_frame(
    frame_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """Load foreground points, colors, and camera info for a single frame.

    Returns a tuple: (fg_xyz, fg_col, c2w, fov_deg, width, height)
    """
    frame_meta = _read_json(frame_dir / f"{frame_dir.name}.json")
    c2w = np.array(frame_meta["c2w"], dtype=np.float32)
    fov_deg = float(frame_meta["fov"]) * 180.0 / math.pi

    rgb_npz = frame_dir / str(frame_meta["full_rgb"])  # (H,W,3) float32 or uint8
    pcd_npz = frame_dir / str(frame_meta["full_pcd"])  # (H,W,3)
    fg_npz = frame_dir / str(frame_meta["fg_mask"])  # (H,W)

    rgb, _, _ = load_grid(str(rgb_npz))
    xyz, _, _ = load_grid(str(pcd_npz))
    fg_mask, _, _ = load_grid(str(fg_npz))

    assert xyz.ndim == 3 and xyz.shape[-1] == 3
    height, width, _ = xyz.shape
    if rgb.ndim == 3 and rgb.shape[:2] == (height, width):
        cols = rgb.astype(np.float32)
    else:
        # Fallback: create white colors
        cols = np.ones((height, width, 3), dtype=np.float32)
    if cols.max() > 1.0:
        cols = cols / 255.0

    fg = (
        fg_mask.astype(bool)
        if fg_mask is not None
        else np.zeros((height, width), dtype=bool)
    )
    if fg.shape != (height, width):
        fg = fg.reshape(height, width)

    pts = xyz.reshape(-1, 3)[fg.reshape(-1)]
    col = cols.reshape(-1, 3)[fg.reshape(-1)]
    return pts.astype(np.float32), col.astype(np.float32), c2w, fov_deg, width, height


def _build_sphere_shapegroup(
    group_id: str, positions: np.ndarray, colors: np.ndarray, radius: float
) -> Dict:
    """Build a Mitsuba shapegroup containing many small colored spheres."""
    positions_array = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    colors_array = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    num_spheres = min(positions_array.shape[0], colors_array.shape[0])
    positions_array = positions_array[:num_spheres]
    colors_array = colors_array[:num_spheres]
    # Build a shapegroup containing many small spheres at the given centers with per-sphere diffuse color
    children: Dict[str, Dict] = {"type": "shapegroup", "id": group_id}
    for i in range(num_spheres):
        color_tuple = tuple(float(x) for x in colors_array[i])
        children[f"s{i}"] = {
            "type": "sphere",
            "center": tuple(float(x) for x in positions_array[i]),
            "radius": float(radius),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": color_tuple},
            },
        }
    return children


def get_mesh_path(sample_id: str, obj_name: str, is_primary: bool) -> dict:
    _suffix = "_0" if is_primary else "_1"
    obj_name = obj_name.strip()
    if " " in obj_name:
        obj_name = obj_name.replace(" ", "_")
    model_path = (
        VIPE_EXPORT_DIR
        / sample_id
        / "video_info"
        / f"size_estimate{_suffix}"
        / f"{obj_name}.obj"
    )
    assert model_path.exists(), f"Mesh path {model_path} does not exist"
    # texture_path = model_path.parent / "ball1.png"
    # NOTE: !IMPORTANT! Use material_0.png for *_transformed.obj due to trimesh export
    texture_path = model_path.parent / "material_0.png"
    assert texture_path.exists(), f"Texture path {texture_path} does not exist"
    return dict(model_path=model_path, texture_path=texture_path)


def load_input_model(
    model_path,
    with_metallic=True,
    with_roughness=True,
    transform=None,
    texture_path=None,
    tex_dim=2048,
    # starting_values=(0.5, 1, 0),
    starting_values=(0, 0.9, 0.5),
):
    """Load a mesh with optional texture maps and build a Mitsuba shape.

    Args:
        model_path (str): Path to the triangle mesh file (e.g., OBJ).
        with_metallic (bool): If True, allocate a metallic texture map.
        with_roughness (bool): If True, allocate a roughness texture map.
        transform (np.ndarray|None): Optional 4x4 transform applied before
            centering/axis adjustments.
        texture_path (str|None): Optional path to an albedo texture map image.
        tex_dim (int): Fallback texture resolution when generating defaults.
        starting_values (tuple): Initial values for (albedo, roughness, metallic)
            textures when generated procedurally.

    Returns:
        mi.Shape: Mitsuba mesh with a Principled BSDF assembled from texture maps.
    """
    print(f"Loading {model_path}...")
    albedo_start, roughness_start, metallic_start = starting_values

    # Load the mesh via Open3D (legacy -> tensor) for consistent material handling
    mesh_old = o3d.io.read_triangle_mesh(str(model_path), False)
    print(f"Number of vertices in mesh_old: {len(mesh_old.vertices)}")
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_old)
    print(f"Number of vertices in mesh: {mesh.vertex['positions'].shape[0]}")

    mesh.material.material_name = (
        "defaultLit"  # note: ignored by Mitsuba, just used to visualize in Open3D
    )

    # Assemble texture maps: albedo (required), optional roughness/metallic.
    # If a map is not provided/available, create a constant texture image.
    if texture_path is not None:
        # albedo_image = np.array(Image.open(tex_map)) / 255.0
        mesh.material.texture_maps["albedo"] = o3d.t.io.read_image(texture_path)
        tex_dim = len(np.array(mesh.material.texture_maps["albedo"]))
        print("Given albedo texture", mesh.material.texture_maps["albedo"])
    elif "albedo" in mesh.material.texture_maps:
        tex_dim = len(np.array(mesh.material.texture_maps["albedo"]))
        print("Loaded albedo texture", mesh.material.texture_maps["albedo"])
    else:
        mesh.material.texture_maps["albedo"] = o3d.t.geometry.Image(
            albedo_start + np.zeros((tex_dim, tex_dim, 3), dtype=np.float32)
        )
        print("Default albedo texture", mesh.material.texture_maps["albedo"])

    if with_roughness:
        mesh.material.texture_maps["roughness"] = o3d.t.geometry.Image(
            roughness_start + np.zeros((tex_dim, tex_dim, 1), dtype=np.float32)
        )

    if with_metallic:
        mesh.material.texture_maps["metallic"] = o3d.t.geometry.Image(
            metallic_start + np.zeros((tex_dim, tex_dim, 1), dtype=np.float32)
        )

    # NOTE POTENTIALLY IMPORTANT:
    # Apply transform to place the mesh at scene center and flip XY axes to
    # match Mitsuba's coordinate expectations in this setup.
    if transform is None:
        transform = np.eye(4)
    trafo = np.array(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    mesh.transform(trafo @ transform)
    # # Persist a transformed mesh copy next to the source (primarily for debugging)
    # o3d.t.io.write_triangle_mesh(f'{model_path.split(".")[0]}_1.obj', mesh)

    # Build a Principled BSDF in Mitsuba, wiring in the texture maps from Open3D.
    bsdf = mi.load_dict(
        {
            "type": "principled",
            "base_color": {
                "type": "bitmap",
                "bitmap": mi.Bitmap(
                    mesh.material.texture_maps["albedo"].as_tensor().numpy()
                ),
                "wrap_mode": "mirror",
            },
            "roughness": {
                "type": "bitmap",
                "bitmap": mi.Bitmap(
                    mesh.material.texture_maps["roughness"].as_tensor().numpy()
                ),
            },
            "metallic": {
                "type": "bitmap",
                "bitmap": mi.Bitmap(
                    mesh.material.texture_maps["metallic"].as_tensor().numpy()
                ),
            },
        }
    )
    mesh_mitsuba = mesh.to_mitsuba("mesh", bsdf=bsdf)
    return mesh_mitsuba


def _build_sensor(
    c2w: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
    C: np.ndarray | None = None,
) -> Dict:
    """Construct a Mitsuba perspective sensor dict.

    If `C` is provided, the final transform is `c2w @ C`; otherwise `c2w`.
    """
    to_world = c2w.astype(np.float32) if C is None else (c2w @ C).astype(np.float32)
    return {
        "type": "perspective",
        "to_world": mi.ScalarTransform4f(to_world),
        "fov_axis": "y",
        "fov": fov_deg,
        "near_clip": 1e-3,
        "far_clip": 100.0,
        "film": {
            "type": "hdrfilm",
            "width": int(width),
            "height": int(height),
            # "rfilter": {"type": "box"},
            "pixel_format": "rgba",
        },
        "sampler": {"type": "multijitter", "sample_count": 64},
    }


def _update_mitsuba_mesh_positions(mesh: Dict, points: np.ndarray) -> Dict:
    mesh_params = mi.traverse(mesh)
    vertices = np.array(mesh_params["vertex_positions"]).reshape(-1, 3)
    vertex_num = len(vertices)
    points = points.reshape(-1, 3)
    if points.shape[0] < vertex_num:
        print(
            f"Warning: number of points ({points.shape[0]}) is less than number of vertices ({vertex_num})"
        )
        exit(1)
    points = points[:vertex_num, :]
    mesh_params["vertex_positions"] = points.flatten()
    mesh_params.update()
    return mesh


def add_cam_light_to_scene(scene: dict, position: np.ndarray, intensity: float = 5.0):
    scene["light_cam"] = {
        "type": "point",
        "to_world": mi.ScalarTransform4f.translate(position),
        "intensity": {"type": "rgb", "value": intensity},
    }


def _build_scene(
    sensor: dict = None,
    bg_group: dict = None,
    fg_group: dict = None,
    bg_mesh: dict = None,
    obj_mesh_primary: dict = None,
    obj_mesh_secondary: dict = None,
) -> dict:
    """Assemble the Mitsuba scene with sensor, environment, lights, and objects."""
    scene: Dict = {
        "type": "scene",
        # "integrator": {"type": "direct"},
        "integrator": {
            "type": "prb",
            "hide_emitters": True,
        },
    }
    scene["env"] = {
        "type": "constant",
        "radiance": {"type": "rgb", "value": [0.2, 0.2, 0.2]},
    }

    # Default point lights
    scene["light_top"] = {
        "type": "point",
        "to_world": mi.ScalarTransform4f.translate([1.0, 8.0, 1.0]),
        "intensity": {"type": "rgb", "value": 35.0},
    }
    # scene["light_back"] = {
    #     "type": "point",
    #     "to_world": mi.ScalarTransform4f.translate([1.0, 1.0, -2.0]),
    #     "intensity": {"type": "rgb", "value": 35.0},
    # }

    if sensor is not None:
        scene["sensor"] = sensor
    if obj_mesh_primary is not None:
        scene["obj_mesh_primary"] = obj_mesh_primary
    if obj_mesh_secondary is not None:
        scene["obj_mesh_secondary"] = obj_mesh_secondary
    if bg_mesh is not None:
        scene["bg_mesh"] = bg_mesh
    if bg_group is not None:
        scene["bg_group"] = bg_group
        scene["bg"] = {
            "type": "instance",
            "child": {"type": "ref", "id": bg_group.get("id", "bg_group")},
        }
    if fg_group is not None:
        scene["fg_group"] = fg_group
        scene["fg"] = {
            "type": "instance",
            "child": {"type": "ref", "id": fg_group.get("id", "fg_group")},
        }
    return scene


def update_scene_dict_and_render(scene_dict: dict, **kwargs):
    scene_dict.update(kwargs)
    for k in list(scene_dict.keys()):
        if scene_dict[k] is None:
            del scene_dict[k]

    scene = mi.load_dict(scene_dict)
    image = mi.render(scene)
    W, H = scene_dict["sensor"]["film"]["width"], scene_dict["sensor"]["film"]["height"]
    denoiser = mi.OptixDenoiser(
        input_size=(W, H), albedo=False, normals=False, temporal=False
    )
    image = denoiser(image)
    return image


def check_type_and_shape(target, name: str = ""):
    if isinstance(target, np.ndarray):
        print(
            f"{name:<20} >>> Object type: {type(target)}, shape: {target.shape}, range: {target.min():.2f} to {target.max():.2f}"
        )
    else:
        print(f"{name:<20} >>> Object type: {type(target)}")


def save_image(img, save_path: str | Path, verbose: bool = True) -> Path:
    if isinstance(save_path, Path):
        save_path = str(save_path)

    if isinstance(img, np.ndarray):
        arr = np.asarray(img)
        # Squeeze trailing single-channel dimension for masks (H,W,1) -> (H,W)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        # Normalize dtype and value range to uint8 [0, 255]
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8) * 255
        else:
            max_value = float(arr.max()) if arr.size else 0.0
            min_value = float(arr.min()) if arr.size else 0.0

            if arr.dtype.kind in {"f", "c"}:  # float/complex (complex unlikely)
                if 0.0 <= min_value and max_value <= 1.0:
                    arr = (arr * 255.0).round().astype(np.uint8)
                elif 0.0 <= min_value and max_value <= 255.01:
                    arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
                else:
                    raise ValueError(
                        f"Image range is not supported for float array: {min_value} to {max_value}"
                    )
            elif arr.dtype.kind in {"u", "i"}:  # unsigned/signed int
                if min_value < 0 or max_value > 255:
                    raise ValueError(
                        f"Integer image values must be within 0..255, got {min_value} to {max_value}"
                    )
                arr = arr.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported numpy dtype for image: {arr.dtype}")

        # Determine mode and save
        if arr.ndim == 2:
            # Grayscale mask or single-channel image
            Image.fromarray(arr, mode="L").save(save_path)
        elif arr.ndim == 3:
            channels = arr.shape[-1]
            if channels == 3:
                Image.fromarray(arr, mode="RGB").save(save_path)
            elif channels == 4:
                Image.fromarray(arr, mode="RGBA").save(save_path)
            else:
                raise ValueError(
                    f"Unsupported channel count: expected 1, 3, or 4 but got {channels}"
                )
        else:
            raise ValueError(
                f"Unsupported array shape for image: {arr.shape}. Expected (H,W), (H,W,3) or (H,W,4)."
            )

    elif isinstance(img, Image.Image):
        img.save(save_path)
    elif mi is not None and isinstance(img, mi.TensorXf):
        mi.util.write_bitmap(save_path, img)
    else:
        raise ValueError(f"Image type is not supported: {type(img)}")

    if verbose:
        print(f"Image saved: [green bold]{save_path}")
    return Path(save_path).resolve()


def post_process_images(
    img_inpainted_bg, img_ground_only, img_obj_only, img_obj_ground, **kwargs
) -> np.ndarray:
    img_inpainted_bg = np.asarray(img_inpainted_bg)
    img_ground_only = np.asarray(img_ground_only)
    img_obj_only = np.asarray(img_obj_only)
    img_obj_ground = np.asarray(img_obj_ground)
    # Compute per-pixel shadow by comparing composited object-only
    # result with the object+plane render.
    # NOTE: important step for shadow mapping
    object_alpha = img_obj_only[..., 3:4]  # (H,W,1) [0, 1]
    # check_type_and_shape(object_alpha, "object_alpha")
    save_image(object_alpha, kwargs["obj_mask_dir"] / f"{kwargs['i']:05d}.png")

    # compose object-only image with ground-only image using object_alpha
    image_no_shade = img_obj_only[..., :3] * object_alpha + img_ground_only[..., :3] * (
        1 - object_alpha
    )  # (H,W,3) [0, 1]
    # check_type_and_shape(image_no_shade, "image_no_shade")
    # mi.util.write_bitmap(f"{output_folder}/{f:05d}_2.png", image_no_shade)

    # get image residual between image_all (with shadow) and image_no_shade (without shadow)
    # the larger the difference, the darker the shadow
    shadow_difference = image_no_shade - img_obj_ground[..., :3]  # (H,W,3) [-1, 1]
    # check_type_and_shape(shadow_difference, "shadow_difference")
    # mi.util.write_bitmap(f"{output_folder}/{f:05d}_3.png", shadow_difference)

    # only to avoid division by zero
    image_no_shade = np.maximum(image_no_shade, 0.1)
    # check_type_and_shape(image_no_shade, "image_no_shade")

    shadow_ratio = shadow_difference / image_no_shade
    # check_type_and_shape(shadow_ratio, "shadow_ratio after division")
    shadow_ratio = np.clip(shadow_ratio, 0, 1)
    # check_type_and_shape(shadow_ratio, "shadow_ratio after clipping")

    # Reinsert shadows into the original photographic background and
    # alpha-composite the rendered object over it.
    shadow_ratio = cv2.cvtColor(np.array(shadow_ratio), cv2.COLOR_BGR2RGB)
    # check_type_and_shape(shadow_ratio, "shadow_ratio after color conversion")

    image_shadowed = img_inpainted_bg * (1 - shadow_ratio)
    # image_shadowed = image_origin - shadow_difference * 255
    # save_image(image_shadowed, kwargs["misc_dir"] / f"shadowed_{kwargs['i']:05d}.png")

    time.sleep(1)  # wait for the image to be saved
    img_obj_ground_3 = np.asarray(
        Image.open(kwargs["obj_ground_dir"] / f"{kwargs['i']:05d}.png")
    )  # (H,W,3) [0, 255]
    # check_type_and_shape(img_obj_ground_3, "img_obj_ground_3")

    final_img = img_obj_ground_3 * object_alpha + image_shadowed * (1 - object_alpha)
    # check_type_and_shape(final_img, "final_img")
    # print("\n\n")
    return final_img


def check_image_object(img, name: str = ""):
    if isinstance(img, np.ndarray):
        _target = img
        dtype = _target.dtype
        shape = _target.shape
        max_value = _target.max()
        min_value = _target.min()
        if 0 <= min_value <= max_value <= 1:
            image_range = "0-1"
        elif 0 <= min_value <= max_value <= 255 and max_value > 3:
            # check max_value > 3 just in case
            image_range = "0-255"
        else:
            image_range = f"{min_value}-{max_value}"

        print(f"{name} >>> Image dtype: {dtype}, shape: {shape}, range: {image_range}")
        return

    elif isinstance(img, Image.Image):
        print(f"{name} >>> Image type: {type(img)}")
        return
    else:
        print(f"{name} >>> Unknown image type: {type(img)}")
        return


def render_simulation_result(
    exported_sample_dir: Path,
    output_dir: Path,
    simulation_id: str,
    interpolate_cameras: bool = True,
):
    """Render each frame directory under `video_dir` into `output_dir` using Mitsuba."""
    assert (
        mi is not None
    ), "Mitsuba 3 is not installed. Please install mitsuba and try again."
    # mi.set_variant("cuda_rgb") if mi.has_backend("cuda_ad_rgb") else mi.set_variant("llvm_rgb")

    ### load video info
    video_info = _load_video_info(exported_sample_dir)
    sample_id = video_info["sample_id"]
    print(
        f"Rendering RGB frames for sample: {sample_id} (interpolate_cameras: {interpolate_cameras})"
    )
    video_n_frames = video_info["N_frames"]

    # check objects
    primary_obj_name = video_info.get("primary_obj_name")
    secondary_obj_name = video_info.get("secondary_obj_name")
    if not primary_obj_name:
        raise ValueError(f"Primary object name not found in video info for {sample_id}")
    primary_obj_path_dict = get_mesh_path(sample_id, primary_obj_name, is_primary=True)
    if secondary_obj_name:
        secondary_obj_path_dict = get_mesh_path(
            sample_id, secondary_obj_name, is_primary=False
        )

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

    ### gather all background images from inpainting results
    perception_sample_dir = OUT_PERCEPTION_DIR / sample_id
    _frames_dirs = sorted(
        [
            x
            for x in perception_sample_dir.glob("*")
            if x.is_dir() and x.stem.startswith("0")
        ]
    )

    if "static" in sample_id:
        background_frames = [
            _frames_dirs[0] / "inpaint" / "inpainted_all.jpg"
        ] * n_frames_to_render
    else:
        background_frames = [
            _frames_dirs[0] / "inpaint" / "inpainted_all.jpg"
        ] * n_frames_to_render

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
        # print(f"Warning: number of particles files ({len(particles_files)}) does not match number of frames ({n_frames_to_render})")
        # n_frames_to_render = len(particles_files)
        # background_frames = background_frames[:n_frames_to_render]
    else:
        print("OK: number of particles files matches number of frames")

    ### sanity check
    assert (
        len(particles_files)
        == len(all_c2w)
        == len(background_frames)
        == n_frames_to_render
    )

    ### create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    ### build ground plane mesh
    ground_mesh = build_ground_mesh(video_info, output_dir)

    ### build object mesh
    # mesh_mitsuba = load_input_model(**get_mesh_path(sample_id))
    mesh_mitsuba_primary = load_input_model(**primary_obj_path_dict)
    mesh_mitsuba_secondary = (
        load_input_model(**secondary_obj_path_dict) if secondary_obj_name else None
    )

    ### build scene
    scene_dict = _build_scene(
        bg_mesh=ground_mesh,
        obj_mesh_primary=mesh_mitsuba_primary,
        obj_mesh_secondary=mesh_mitsuba_secondary,
    )

    ### build render output directories
    def make_sub_output_dir(output_dir: Path, sub_dir_name: str) -> Path:
        output_dir = Path(output_dir)
        sub_dir = output_dir / sub_dir_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        return sub_dir

    inpainted_bg_dir = make_sub_output_dir(output_dir, "inpainted_bg")
    ground_only_dir = make_sub_output_dir(output_dir, "ground_only")
    obj_only_dir = make_sub_output_dir(output_dir, "obj_only")
    obj_mask_dir = make_sub_output_dir(output_dir, "obj_mask")
    obj_ground_dir = make_sub_output_dir(output_dir, "obj_ground")
    # misc_dir = make_sub_output_dir(output_dir, "misc")
    final_dir = make_sub_output_dir(output_dir, "final")
    video_out_dir = make_sub_output_dir(output_dir, "video")

    for i in range(n_frames_to_render):
        print(f"Rendering frame {i}...")
        print(f"-" * 100)
        time_start = time.time()
        # build new sensor for this frame
        sensor = _build_sensor(c2w=all_c2w[i], **cam_base_params)
        cam_pos = all_c2w[i][:3, 3].copy().tolist()
        add_cam_light_to_scene(scene_dict, cam_pos)

        # DEBUG ONLY: render zero image, check mesh loading
        if i == 0:
            img_zero = update_scene_dict_and_render(scene_dict, sensor=sensor)
            save_image(img_zero, output_dir / f"zero.png")
        # DEBUG end here

        # update mesh vertices position from physics simulation output
        # mesh_mitsuba = _update_mitsuba_mesh_positions_from_npz(mesh_mitsuba, particles_files[i])
        # load simulated particles
        np_x, _, _ = ParticleIO.read_particles_3d(particles_files[i])
        np_x_primary = np_x[:primary_start_index]
        mesh_mitsuba_primary = _update_mitsuba_mesh_positions(
            mesh_mitsuba_primary, np_x_primary
        )
        if secondary_obj_name:
            np_x_secondary = np_x[primary_end_index:secondary_start_index]
            mesh_mitsuba_secondary = _update_mitsuba_mesh_positions(
                mesh_mitsuba_secondary, np_x_secondary
            )

        # get inpainted background image
        img_inpainted_bg = Image.open(background_frames[i])
        img_inpainted_bg = np.array(img_inpainted_bg).astype(np.float32)
        # save_image(img_inpainted_bg, inpainted_bg_dir / f"{i:05d}.png")

        # render new image for this frame
        img_ground_only = update_scene_dict_and_render(
            scene_dict, sensor=sensor, obj_mesh_primary=None
        )
        img_obj_only = update_scene_dict_and_render(
            scene_dict,
            sensor=sensor,
            obj_mesh_primary=mesh_mitsuba_primary,
            obj_mesh_secondary=mesh_mitsuba_secondary,
            bg_mesh=None,
        )
        img_obj_ground = update_scene_dict_and_render(
            scene_dict,
            sensor=sensor,
            obj_mesh_primary=mesh_mitsuba_primary,
            obj_mesh_secondary=mesh_mitsuba_secondary,
            bg_mesh=ground_mesh,
        )
        # save_image(img_ground_only, ground_only_dir / f"{i:05d}.png")
        save_image(img_obj_only, obj_only_dir / f"{i:05d}.png")
        save_image(img_obj_ground, obj_ground_dir / f"{i:05d}.png")

        # compose images
        img_final = post_process_images(
            img_inpainted_bg,
            img_ground_only,
            img_obj_only,
            img_obj_ground,
            i=i,
            obj_ground_dir=obj_ground_dir,
            obj_mask_dir=obj_mask_dir,
        )
        save_image(img_final, final_dir / f"{i:05d}.png")

        time_elapsed = time.time() - time_start
        print(f"Frame {i} rendered in {time_elapsed:.1f} seconds")

    print(f"-" * 100)
    time.sleep(2)
    print("[green bold]Making video from rendered frames...")
    make_video_from_frames_dir(final_dir, video_out_dir / f"{sample_id}_final.mp4")
    make_video_from_frames_dir(
        obj_ground_dir, video_out_dir / f"{sample_id}_obj_ground.mp4"
    )
    make_video_from_frames_dir(
        obj_only_dir, video_out_dir / f"{sample_id}_obj_only.mp4"
    )


def render_RGB_video(sample_id: str, simulation_id: str, overwrite: bool = False):
    sim_success_file = OUT_SIMULATION_DIR / sample_id / simulation_id / "success.txt"
    if not sim_success_file.exists():
        print(f"Simulation results not found for {sample_id} at {sim_success_file}")
        exit(3)

    root_success_file = (
        OUT_RENDERING_DIR / sample_id / simulation_id / "success_render.txt"
    )
    if root_success_file.exists() and not overwrite:
        print(
            f"  [yellow]✔ Skipped[/yellow] {sample_id} ({simulation_id}) because rendering result exists"
        )
        return

    success_files = []
    for interpolate_cameras in [False, True]:
        _txt = "more_frames" if interpolate_cameras else "original_length"
        output_dir = OUT_RENDERING_DIR / sample_id / simulation_id / _txt
        out_video_path = output_dir / "video" / f"{sample_id}_obj_ground.mp4"
        success_file = output_dir / "success_render.txt"
        success_files.append(success_file)
        if success_file.exists() and out_video_path.exists() and not overwrite:
            print(
                f"RGB frames are already rendered for {sample_id} at {_txt}, skipping."
            )
            continue

        exported_sample_dir = VIPE_EXPORT_DIR / sample_id

        backup_this_script(output_dir)

        render_simulation_result(
            exported_sample_dir=exported_sample_dir,
            output_dir=output_dir,
            simulation_id=simulation_id,
            interpolate_cameras=interpolate_cameras,
        )

        success_file.touch()

        if not interpolate_cameras:
            out_video_copy = (
                OUT_RENDERING_DIR
                / "all_videos"
                / f"{simulation_id}"
                / out_video_path.name
            )
            out_video_copy.parent.mkdir(parents=True, exist_ok=True)
            # copy to all_videos
            if not out_video_copy.exists():
                shutil.copy(out_video_path, out_video_copy)

    if all([success_file.exists() for success_file in success_files]):
        root_success_file.touch()


if __name__ == "__main__":
    ...
