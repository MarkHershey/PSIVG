import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Dict, List

import numpy as np
import open3d as o3d
import trimesh
from arrgh import arrgh
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from psivg.constants import INPUT_FRAMES_DIR, OUT_PERCEPTION_DIR, VIPE_EXPORT_DIR
from rich import print

try:
    import mitsuba as mi

    mi.set_variant("cuda_ad_rgb")
except Exception:
    print("Mitsuba 3 is not installed. Please install mitsuba and try again.")
    mi = None


def backup_this_script(output_dir: Path, timestamp: str = None):
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
    vertices = np.array(mesh_old.vertices)
    # print(f"Number of vertices in mesh_old: {len(mesh_old.vertices)}")
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_old)
    # print(f"Number of vertices in mesh: {mesh.vertex['positions'].shape[0]}")

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
    # NOTE: somehow the following line is necessary for the correct vertex positions
    mesh_mitsuba = _update_mitsuba_mesh_positions(mesh_mitsuba, vertices)
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


def _build_scene(
    sensor: dict = None,
    bg_group: dict = None,
    fg_group: dict = None,
    bg_mesh: dict = None,
    obj_mesh: dict = None,
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
    scene["light_back"] = {
        "type": "point",
        "to_world": mi.ScalarTransform4f.translate([1.0, 1.0, -2.0]),
        "intensity": {"type": "rgb", "value": 35.0},
    }

    if sensor is not None:
        scene["sensor"] = sensor
    if obj_mesh is not None:
        scene["obj_mesh"] = obj_mesh
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
                elif 0.0 <= min_value and max_value <= 255.0:
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

    return (int(y_min), int(x_min), int(y_max), int(x_max))


def draw_and_save_bbox(image_path: str, bbox: tuple, color=(255, 0, 0), width=2):
    """
    Draw a bounding box on an image and overwrite the original file.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    bbox : tuple
        (y_min, x_min, y_max, x_max) bounding box coordinates.
    color : tuple, optional
        RGB color of the box. Default is red (255, 0, 0).
    width : int, optional
        Line width of the rectangle. Default is 2.
    """
    # Open the image and remember its mode
    img = Image.open(image_path)
    original_mode = img.mode

    # Convert to RGB for drawing (RGBA also works)
    if original_mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # Draw bounding box
    draw = ImageDraw.Draw(img)
    y_min, x_min, y_max, x_max = bbox
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

    # Save back in original mode (if possible)
    img.save(image_path)


def load_mask_image(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return np.asarray(mask)


def box_iou(box1: tuple, box2: tuple) -> float:
    y_min1, x_min1, y_max1, x_max1 = box1
    y_min2, x_min2, y_max2, x_max2 = box2
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    intersection = max(0, min(x_max1, x_max2) - max(x_min1, x_min2)) * max(
        0, min(y_max1, y_max2) - max(y_min1, y_min2)
    )
    return intersection / (area1 + area2 - intersection)


def object_scale_refinement(
    exported_sample_dir: Path,
    num_proposals: int = 30,
    debug: bool = False,
):
    """Render each frame directory under `exported_sample_dir` into `output_dir` using Mitsuba."""
    assert (
        mi is not None
    ), "Mitsuba 3 is not installed. Please install mitsuba and try again."
    # mi.set_variant("cuda_rgb") if mi.has_backend("cuda_ad_rgb") else mi.set_variant("llvm_rgb")

    video_info_dir = exported_sample_dir / "video_info"
    assert (
        video_info_dir.exists()
    ), f"Video info directory {video_info_dir} does not exist"

    sample_id = exported_sample_dir.name
    # perception_mask = OUT_PERCEPTION_DIR / sample_id / "00000" / "mask" / "mask.jpg"
    # perception_mask = load_mask_image(perception_mask)
    # # arrgh(perception_mask)
    # ref_box = get_box_from_mask(perception_mask, threshold=0.5)

    ### load video info
    video_info = _load_video_info(exported_sample_dir)
    sample_id = video_info["sample_id"]

    ### load all frames' infos
    frame_dirs = _list_frame_dirs(exported_sample_dir)  # already sorted
    frame_0_info = _load_frame_infos(frame_dirs[0:1])[0]

    primary_obj_name = frame_0_info.get("primary_obj_name")
    secondary_obj_name = frame_0_info.get("secondary_obj_name")
    if primary_obj_name and secondary_obj_name:
        object_names = [primary_obj_name, secondary_obj_name]
    elif primary_obj_name:
        object_names = [primary_obj_name]
    else:
        raise ValueError(f"No object names found in {frame_0_info}")

    for idx, object_name in enumerate(object_names):
        is_primary = idx == 0
        if object_name == "yo-yo":
            object_name = "yoyo"

        obj_file_name = object_name.replace(" ", "_")
        obj_mesh_path = (
            OUT_PERCEPTION_DIR / sample_id / "00000" / "meshes" / f"{obj_file_name}.obj"
        )
        if not obj_mesh_path.exists():
            print(f"Object mesh path {obj_mesh_path} does not exist")
            print(
                f"Listing out all files in {OUT_PERCEPTION_DIR / sample_id / '00000' / 'meshes'}"
            )
            for file in (OUT_PERCEPTION_DIR / sample_id / "00000" / "meshes").glob("*"):
                print(file)
            raise ValueError(f"Object mesh path {obj_mesh_path} does not exist")

        perception_mask_path = (
            OUT_PERCEPTION_DIR
            / sample_id
            / "00000"
            / "mask"
            / f"mask_{obj_file_name}.jpg"
        )
        assert (
            perception_mask_path.exists()
        ), f"Perception mask path {perception_mask_path} does not exist"
        perception_mask = load_mask_image(perception_mask_path)
        ref_box = get_box_from_mask(perception_mask, threshold=0.5)

        if is_primary:
            obj_radius = float(frame_0_info["primary_obj_radius"])
        else:
            obj_radius = float(frame_0_info["secondary_obj_radius"])

        lower = obj_radius * 0.5
        upper = obj_radius * 3.5
        side_length_proposals = np.linspace(lower, upper, num_proposals).tolist()
        if debug:
            print(f"lower: {lower}, upper: {upper}, num_proposals: {num_proposals}")
            print(f"Side length proposals: {side_length_proposals}")

        ### build render output directories
        def make_sub_output_dir(output_dir: Path, sub_dir_name: str) -> Path:
            output_dir = Path(output_dir)
            sub_dir = output_dir / sub_dir_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            return sub_dir

        size_estimate_dir = make_sub_output_dir(video_info_dir, f"size_estimate_{idx}")
        tmp_models_dir = make_sub_output_dir(size_estimate_dir, f"tmp_models_{idx}")

        ### build fixed object transform
        if is_primary:
            obj_pos = np.array(frame_0_info["primary_obj_centroid"])
        else:
            obj_pos = np.array(frame_0_info["secondary_obj_centroid"])

        obj_rot = [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
        obj_rot = np.array(obj_rot, dtype=np.float32)
        obj_RT = np.eye(4, dtype=np.float32)
        obj_RT[:3, :3] = obj_rot
        obj_RT[:3, 3] = obj_pos

        candidate_paths = []
        for i in range(num_proposals):
            side_length = float(side_length_proposals[i])
            object_mesh = normalize_mesh(obj_mesh_path, side=side_length)
            object_mesh.vertices = transform_pcd(object_mesh.vertices, obj_RT)
            obj_mesh_transformed_path = (
                tmp_models_dir / f"{obj_mesh_path.stem}_{i:05d}.obj"
            )
            object_mesh.export(obj_mesh_transformed_path)
            candidate_paths.append(obj_mesh_transformed_path)

        texture_path = tmp_models_dir / "material_0.png"
        assert texture_path.exists(), f"Texture path {texture_path} does not exist"

        ### load all camera data
        cam_base_params = load_cam_base_params(frame_0_info)
        first_c2w = np.array(frame_0_info["c2w"], dtype=np.float32)

        ### sensor
        sensor = _build_sensor(c2w=first_c2w, **cam_base_params)

        ### build scene
        scene_dict = _build_scene(sensor=sensor)

        box_ious = []

        for i in range(num_proposals):
            ### build object mesh
            mesh_mitsuba = load_input_model(
                model_path=candidate_paths[i], texture_path=texture_path
            )

            ### render
            img_obj_only = update_scene_dict_and_render(
                scene_dict, obj_mesh=mesh_mitsuba, bg_mesh=None
            )
            if debug:
                img_obj_path = size_estimate_dir / f"{i:05d}.png"
                save_image(img_obj_only, img_obj_path)
            img_obj_only = np.asarray(img_obj_only)

            object_alpha = img_obj_only[..., 3:4]  # (H,W,1) [0, 1] float32
            box = get_box_from_mask(
                object_alpha, threshold=0.0
            )  # !Important: threshold=0.0
            box_ious.append(box_iou(ref_box, box))

            if debug:
                print(f"Box: {box}")
                sleep(0.2)
                draw_and_save_bbox(img_obj_path, box)

        if debug:
            print(f"Box ious: {box_ious}")
            largest_iou = max(box_ious)
            largest_iou_index = box_ious.index(largest_iou)
            print(f"Largest iou: {largest_iou}, index: {largest_iou_index}")

        ###########################################################################
        ### Find the optimal side length
        X = side_length_proposals
        Y = box_ious
        # fit a polynomial
        # and return the side length that gives the largest iou
        coeffs = np.polyfit(X, Y, int(num_proposals / 2))
        eval_x = np.linspace(lower, upper, 1000)
        eval_y = np.polyval(coeffs, eval_x)
        optimal_x = eval_x[np.argmax(eval_y)]
        # graph the polynomial
        plt.plot(eval_x, eval_y)
        # scatter the original data points as red dots
        plt.scatter(X, Y, color="red")
        # draw a vertical line at the optimal x
        plt.axvline(x=optimal_x, color="green", linestyle="--")
        # label the optimal x
        plt.annotate(f"{optimal_x:.3f}", (optimal_x, 0.5), color="green")
        # label points
        for i in range(len(X)):
            plt.annotate(f"{X[i]:.4f}", (X[i], Y[i]))
        # x label
        plt.xlabel("Object Box Side Length")
        # y label
        plt.ylabel("IOU")
        # title
        plt.title("Object Size Estimation")

        plt.savefig(size_estimate_dir / "box_ious.png")
        plt.close()
        print(f"Optimal side length: {optimal_x}")

        ###########################################################################
        ### Remove tmp models
        shutil.rmtree(tmp_models_dir)

        ###########################################################################
        ### Scale the object mesh properly using optimal side length
        optimized_out = size_estimate_dir / f"{obj_mesh_path.stem}.obj"
        object_mesh = normalize_mesh(obj_mesh_path, side=optimal_x)
        object_mesh.vertices = transform_pcd(object_mesh.vertices, obj_RT)
        object_mesh.export(optimized_out)
        print(f"Optimized object mesh saved to {optimized_out}")

        ###########################################################################
        ### Render the optimized object mesh
        mesh_mitsuba = load_input_model(
            model_path=optimized_out, texture_path=size_estimate_dir / "material_0.png"
        )
        img_obj_only = update_scene_dict_and_render(
            scene_dict, obj_mesh=mesh_mitsuba, bg_mesh=None
        )
        img_obj_path = size_estimate_dir / f"optimized.png"
        save_image(img_obj_only, img_obj_path)
        render_box = get_box_from_mask(
            np.asarray(img_obj_only)[..., 3:4], threshold=0.0
        )  # !Important: threshold=0.0
        # draw side-to-side comparison of the original and optimized object rendering
        ref_rgb_path = INPUT_FRAMES_DIR / sample_id / "00000.jpg"
        assert (
            ref_rgb_path.exists()
        ), f"Reference RGB path {ref_rgb_path} does not exist"
        ref_rgb = Image.open(ref_rgb_path).convert("RGB")
        # draw ref_box on ref_rgb
        draw = ImageDraw.Draw(ref_rgb)
        y_min, x_min, y_max, x_max = ref_box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
        # label coordinates
        font = ImageFont.load_default(size=20)
        draw.text((x_min, y_min), f"({x_min}, {y_min})", fill="green", font=font)
        draw.text((x_max, y_max), f"({x_max}, {y_max})", fill="green", font=font)
        draw.text((5, 5), f"Original Video Frame", fill="green", font=font)

        # load saved optimized image
        sleep(0.5)
        render_rgb = Image.open(img_obj_path).convert("RGB")
        draw = ImageDraw.Draw(render_rgb)
        y_min, x_min, y_max, x_max = render_box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        # label coordinates
        draw.text((x_min, y_min), f"({x_min}, {y_min})", fill="red", font=font)
        draw.text((x_max, y_max), f"({x_max}, {y_max})", fill="red", font=font)
        draw.text(
            (5, 5), f"Scale-Optimized Object Mesh Rendering", fill="red", font=font
        )

        # convert to numpy array
        ref_rgb = np.asarray(ref_rgb)
        render_rgb = np.asarray(render_rgb)

        if debug:
            arrgh(ref_rgb, render_rgb)
        assert (
            ref_rgb.shape == render_rgb.shape
        ), f"Reference RGB and render RGB have different shapes: {ref_rgb.shape} != {render_rgb.shape}"
        # draw side-to-side comparison
        comp_rgb = np.concatenate([ref_rgb, render_rgb], axis=1)
        comp_rgb_path = size_estimate_dir / f"RGB_side_by_side.png"
        Image.fromarray((comp_rgb).astype(np.uint8)).save(comp_rgb_path)


def run_object_scale_refinement(sample_id: str, overwrite: bool = True):
    exported_sample_dir = VIPE_EXPORT_DIR / sample_id

    estimate_success_file = exported_sample_dir / "estimate_success.txt"
    if estimate_success_file.exists() and not overwrite:
        print(
            f"  [yellow]✔ Skipped[/yellow] {sample_id} because refinement results already exist"
        )
        return
    estimate_lock_file = exported_sample_dir / "estimate_lock.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(estimate_lock_file, "w") as f:
        f.write(timestamp)

    object_scale_refinement(
        exported_sample_dir=exported_sample_dir,
        num_proposals=30,
        debug=False,
    )

    # release the lock
    estimate_lock_file.unlink()
    # set the success file
    estimate_success_file.touch()
