import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline


def interpolate_c2ws(c2ws, factor=2, times=None, smooth=True):
    """
    Upsample camera-to-world (c2w) 4x4 poses by an integer factor.
    c2ws: A list of (4, 4) array of homogeneous c2w matrices.
    factor: 2 -> ~2N-1 samples, 4 -> ~4N-3, etc. (k-1 inserts between each pair)
    times: optional (N,) strictly increasing times. If None, uses 0..N-1
    smooth: if True use RotationSpline + CubicSpline; else per-segment slerp + linear.
    Returns: A list of (4, 4) array of upsampled c2w poses in time order.
    """
    c2ws = np.stack(c2ws, axis=0)
    assert c2ws.ndim == 3 and c2ws.shape[1:] == (4, 4)
    N = c2ws.shape[0]
    if N < 2:
        return c2ws.copy()

    if times is None:
        times = np.arange(N, dtype=float)
    else:
        times = np.asarray(times, dtype=float)
        assert times.shape == (N,)
        assert np.all(np.diff(times) > 0), "times must be strictly increasing"

    # Extract rotations & translations
    Rs = R.from_matrix(c2ws[:, :3, :3])
    Ts = c2ws[:, :3, 3]

    # Dense sampling times: insert (factor-1) between each pair
    k = int(factor)
    assert k >= 2
    t_dense = []
    for i in range(N - 1):
        seg = np.linspace(times[i], times[i + 1], k, endpoint=False)
        t_dense.append(seg)
    t_dense.append([times[-1]])
    t_dense = np.concatenate(t_dense)

    if smooth:
        # Smooth rotations
        rot_spline = RotationSpline(times, Rs)
        Rs_new = rot_spline(t_dense)

        # Smooth translations (C¹). Use natural boundary conditions by default.
        T_splines = [CubicSpline(times, Ts[:, j]) for j in range(3)]
        Ts_new = np.column_stack([spl(t_dense) for spl in T_splines])
    else:
        # Piecewise slerp + linear
        quats = Rs.as_quat()
        Rs_list = []
        Ts_list = []

        for i in range(N - 1):
            Ri = R.from_quat(quats[i])
            Rj = R.from_quat(quats[i + 1])

            # slerp parameters for this segment
            seg_u = np.linspace(0.0, 1.0, k, endpoint=False)
            slerp = R.slerp(0, 1, [Ri, Rj])  # SciPy 1.13+: classmethod
            # For older SciPy: from scipy.spatial.transform import Slerp; slerp = Slerp([0,1], R.from_quat([quats[i], quats[i+1]]))
            # Rs_seg = slerp(seg_u)  # with Slerp
            Rs_seg = slerp(seg_u)  # with Rotation.slerp

            Ti, Tj = Ts[i], Ts[i + 1]
            Ts_seg = (1.0 - seg_u)[:, None] * Ti[None, :] + seg_u[:, None] * Tj[None, :]
            Rs_list.append(Rs_seg)
            Ts_list.append(Ts_seg)

        # append the last keyframe
        Rs_list.append(R.from_quat(quats[-1]).reshape(1))
        Ts_list.append(Ts[-1].reshape(1, 3))
        Rs_new = R.concatenate([r for r in Rs_list])
        Ts_new = np.vstack(Ts_list)

    # Reassemble homogeneous matrices
    M = len(t_dense)
    out = np.tile(np.eye(4), (M, 1, 1))
    out[:, :3, :3] = Rs_new.as_matrix()
    out[:, :3, 3] = Ts_new

    return [out[i] for i in range(M)]
