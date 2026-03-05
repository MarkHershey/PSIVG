# grid_npz_io.py — NPZ-only grids: (H,W) or (H,W,C), NumPy 1.x/2.x compatible
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["GridMeta", "save_grid", "load_grid", "save_depth", "load_depth"]

MAGIC = "GRID_NPZ"
VERSION = "grid-npz/1.0"

# Use a cross-version Unicode dtype (works in NumPy 1.x and 2.x)
_STR_DTYPE = "U"


@dataclass
class GridMeta:
    # Global stats over all channels
    units: str = "meter"  # generic description (or main channel)
    min: float = float("nan")
    max: float = float("nan")
    nan_count: int = 0

    # Per-channel info (only present if C>1)
    channels: int = 1
    units_c: Optional[List[str]] = None
    min_c: Optional[List[float]] = None
    max_c: Optional[List[float]] = None
    nan_count_c: Optional[List[int]] = None
    channel_names: Optional[List[str]] = None

    # Misc
    version: str = VERSION
    notes: Optional[str] = None
    data_sha256: Optional[str] = None  # integrity check for 'data' bytes


# ---------- internals ----------


def _ensure_hw_or_hwc(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        # FIXME: temporary allow 1D arrays (convert to 2D)
        a = a.reshape(-1, 1)
    if a.ndim not in (2, 3):
        raise ValueError(f"Array must be (H,W) or (H,W,C); got {a.shape}")
    return a


def _as_c_le_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32, order="C")
    # Force little-endian on disk for stability across platforms
    if x.dtype.byteorder not in ("<", "="):
        x = x.byteswap().newbyteorder("<")
    return np.ascontiguousarray(x)


def _normalize_units(
    units: Optional[Sequence[str] | str], C: int
) -> Tuple[str, Optional[List[str]]]:
    if units is None:
        base = "meter" if C == 1 else "unit"
        return base, ([base] * C if C > 1 else None)
    if isinstance(units, str):
        return units, ([units] * C if C > 1 else None)
    lst = [str(u) for u in units]
    if len(lst) != C:
        raise ValueError(f"units list length {len(lst)} must match channel count {C}")
    return ("mixed" if C > 1 else lst[0]), lst


def _compute_stats(
    arr: np.ndarray,
) -> Tuple[float, float, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Return (gmin, gmax, gnan, C, min_c, max_c, nan_c). Per-channel arrays are (C,)."""
    A = arr if arr.ndim == 3 else arr[..., None]
    H, W, C = A.shape
    finite = np.isfinite(A)
    min_c = np.full(C, np.nan, dtype=np.float32)
    max_c = np.full(C, np.nan, dtype=np.float32)
    nan_c = np.zeros(C, dtype=np.int64)
    for c in range(C):
        f = finite[..., c]
        nan_c[c] = int(np.count_nonzero(~f))
        if np.any(f):
            v = A[..., c][f]
            min_c[c] = float(np.min(v))
            max_c[c] = float(np.max(v))
    if np.any(finite):
        gmin = float(np.min(A[finite]))
        gmax = float(np.max(A[finite]))
    else:
        gmin = gmax = float("nan")
    gnan = int(np.count_nonzero(~finite))
    return gmin, gmax, gnan, C, min_c, max_c, nan_c


def _validate_mask(
    mask: Optional[np.ndarray], data_shape: Tuple[int, ...]
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask).astype(bool, copy=False)
    if m.shape == data_shape or (m.ndim == 2 and m.shape == data_shape[:2]):
        return np.ascontiguousarray(m)
    raise ValueError(
        f"mask shape {m.shape} must be (H,W) or match data shape {data_shape}"
    )


def _put_str(d: Dict[str, Any], key: str, val: Optional[str]) -> None:
    if val is not None:
        d[key] = np.array(val, dtype=_STR_DTYPE)  # cross-version Unicode


def _get_str(z, key: str, default: Optional[str] = None) -> Optional[str]:
    if key not in z.files:
        return default
    v = z[key]
    try:
        return v.item()
    except Exception:
        # Robust fallbacks if shape/dtype is unexpected
        try:
            return str(v[...])
        except Exception:
            return default


# ---------- public API ----------


def save_grid(
    path: str,
    data: np.ndarray,  # (H,W) or (H,W,C) float-like
    *,
    units: Optional[Sequence[str] | str] = "meter",
    mask: Optional[np.ndarray] = None,  # (H,W) or (H,W,C) boolean
    channel_names: Optional[Sequence[str]] = None,
    notes: Optional[str] = None,
    write_checksum: bool = True,
) -> None:
    """
    Save a grid to .npz with explicit, pickle-free metadata.
    - Data is stored as C-contiguous, little-endian float32.
    - Mask (if provided) is stored as uint8 with the same shape or (H,W).
    """
    a = _ensure_hw_or_hwc(data)
    path = str(path)
    if not path.lower().endswith(".npz"):
        path = path + ".npz"

    m = _validate_mask(mask, a.shape)
    gmin, gmax, gnan, C, min_c, max_c, nan_c = _compute_stats(
        np.asarray(a, dtype=np.float32)
    )
    scalar_units, units_c = _normalize_units(units, C)

    if channel_names is not None:
        names = [str(x) for x in channel_names]
        if len(names) != C:
            raise ValueError(
                f"channel_names length {len(names)} must match channel count {C}"
            )
    else:
        names = None

    data_le = _as_c_le_f32(a)
    checksum = hashlib.sha256(data_le.tobytes()).hexdigest() if write_checksum else None

    meta = GridMeta(
        units=scalar_units,
        min=gmin,
        max=gmax,
        nan_count=gnan,
        channels=C,
        units_c=(units_c if C > 1 else None),
        min_c=min_c.tolist() if C > 1 else None,
        max_c=max_c.tolist() if C > 1 else None,
        nan_count_c=nan_c.tolist() if C > 1 else None,
        channel_names=names,
        version=VERSION,
        notes=notes,
        data_sha256=checksum,
    )

    payload: Dict[str, Any] = {
        "magic": np.array(MAGIC, dtype=_STR_DTYPE),
        "version": np.array(VERSION, dtype=_STR_DTYPE),
        "shape": np.asarray(a.shape, dtype=np.int64),
        "ndim": np.array(a.ndim, dtype=np.int64),
        "channels": np.array(C, dtype=np.int64),
        "data": data_le,
        "min": np.array(meta.min, dtype=np.float32),
        "max": np.array(meta.max, dtype=np.float32),
        "nan_count": np.array(meta.nan_count, dtype=np.int64),
    }
    if C > 1:
        payload["min_c"] = min_c.astype(np.float32)
        payload["max_c"] = max_c.astype(np.float32)
        payload["nan_count_c"] = nan_c.astype(np.int64)
    if m is not None:
        payload["mask"] = m.astype(np.uint8)  # 1=valid

    # Meta JSON + convenience fields
    _put_str(payload, "meta_json", json.dumps(asdict(meta), ensure_ascii=False))
    _put_str(payload, "units", meta.units)
    if notes:
        _put_str(payload, "notes", notes)
    if checksum:
        _put_str(payload, "data_sha256", checksum)

    np.savez_compressed(path, **payload)


def load_grid(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], GridMeta]:
    """
    Load a grid saved by save_grid().
    Returns (data, mask_or_none, meta). Data is float32, C-contiguous.
    """
    with np.load(path, allow_pickle=False) as z:
        # Required fields
        if "data" not in z.files or "shape" not in z.files:
            raise ValueError("Invalid file: expected 'data' and 'shape'.")
        if "magic" in z.files and _get_str(z, "magic") != MAGIC:
            raise ValueError("Unrecognized magic; not a GRID_NPZ file.")

        shp = tuple(int(x) for x in z["shape"].tolist())
        arr = z["data"]
        if tuple(arr.shape) != shp:
            raise ValueError(
                f"Shape stamp mismatch: file {shp} vs array {tuple(arr.shape)}"
            )
        if arr.ndim not in (2, 3):
            raise ValueError(f"Data must be 2D or 3D; got {arr.ndim}D")

        data = _as_c_le_f32(arr)

        mask = None
        if "mask" in z.files:
            m = z["mask"]
            ok = (m.shape == data.shape) or (m.ndim == 2 and m.shape == data.shape[:2])
            if not ok:
                raise ValueError(
                    f"Mask shape {m.shape} incompatible with data shape {data.shape}"
                )
            mask = m.astype(np.uint8, copy=False) > 0

        # Metadata
        mj = _get_str(z, "meta_json", None)
        if not mj:
            raise ValueError("Missing 'meta_json' in NPZ.")
        d = json.loads(mj)
        meta = GridMeta(
            units=d.get("units", "meter"),
            min=float(d.get("min", float("nan"))),
            max=float(d.get("max", float("nan"))),
            nan_count=int(d.get("nan_count", 0)),
            channels=int(d.get("channels", data.shape[2] if data.ndim == 3 else 1)),
            units_c=d.get("units_c"),
            min_c=d.get("min_c"),
            max_c=d.get("max_c"),
            nan_count_c=d.get("nan_count_c"),
            channel_names=d.get("channel_names"),
            version=d.get("version", VERSION),
            notes=d.get("notes"),
            data_sha256=d.get("data_sha256"),
        )

        # Optional integrity check
        if meta.data_sha256:
            chk = hashlib.sha256(data.tobytes()).hexdigest()
            if chk != meta.data_sha256:
                raise ValueError("Data checksum mismatch; file may be corrupted.")

    return data, mask, meta


# Friendly aliases
save_depth = save_grid
load_depth = load_grid

if __name__ == "__main__":
    # quick smoke test (works on NumPy 1.x and 2.x)
    H, W = 8, 10
    d = np.linspace(0.1, 3.1, H * W, dtype=np.float32).reshape(H, W)
    m2 = np.isfinite(d)
    save_grid("demo_2d.npz", d, units="meter", mask=m2, notes="2D depth")
    d2, m2r, meta2 = load_grid("demo_2d.npz")
    assert d2.shape == (H, W) and (m2r is None or m2r.shape == (H, W))

    xyz = np.zeros((H, W, 3), dtype=np.float32)
    xyz[..., 2] = np.linspace(0, 1, H * W, dtype=np.float32).reshape(H, W)
    save_grid(
        "demo_xyz.npz",
        xyz,
        units=["meter", "meter", "meter"],
        channel_names=["x", "y", "z"],
    )
    g3, m3, meta3 = load_grid("demo_xyz.npz")
    assert g3.shape == (H, W, 3) and m3 is None
    print("OK:", meta2, meta3)
