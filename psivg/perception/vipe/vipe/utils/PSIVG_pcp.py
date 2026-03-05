"""
PCP: Point Cloud Processing

This module trim floating outliers from point clouds via random projection.

"""

import numpy as np


def _robust_scale(X, eps=1e-12):
    """Median/MAD scaling (affine-insensitive magnitudes across axes)."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.where(mad < eps, eps, mad)
    return (X - med) / mad


def random_projection_depth_scores(
    X, keep_alpha=0.95, n_dirs=200, rng=0, robust_scale=True, use_partition=True
):
    """
    Returns depth scores in [0,1] for each point:
    score = fraction of random directions where the point lies in the
             central [lo, hi] quantile interval of the 1D projection.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    assert d == 3, "This implementation expects 3D points."

    if robust_scale:
        Xr = _robust_scale(X)
        center = np.zeros(3)
    else:
        center = np.median(X, axis=0)
        Xr = X - center

    rng = np.random.default_rng(rng)
    U = rng.normal(size=(n_dirs, 3))
    U /= np.linalg.norm(U, axis=1, keepdims=True)  # unit directions

    lo_q = (1 - keep_alpha) / 2.0
    hi_q = 1 - lo_q

    inside = np.zeros(n, dtype=np.int32)

    # Indices for linear-time selection if use_partition=True
    i_lo = int(np.floor(lo_q * (n - 1)))
    i_hi = int(np.floor(hi_q * (n - 1)))

    for u in U:
        proj = Xr @ u  # shape (n,)
        if use_partition:
            # O(n) quantiles via partial selection
            qlo = np.partition(proj, i_lo)[i_lo]
            qhi = np.partition(proj, i_hi)[i_hi]
        else:
            qlo, qhi = np.quantile(proj, [lo_q, hi_q], method="linear")
        inside += (proj >= qlo) & (proj <= qhi)

    scores = inside.astype(float) / n_dirs
    return scores


def dynamic_depth_trim_threshold(
    scores,
    max_tail_frac=0.01,  # only consider dropping <= 1% on the low-score side
    gap_quantile=0.99,  # gap must exceed this quantile of typical gaps
    gap_ratio=10.0,  # and be >= gap_ratio * median(typical gaps)
    min_gap_steps=2,  # at least this many discrete score steps (1/n_dirs per step)
    n_dirs=None,  # pass the n_dirs used to set absolute min gap; optional
):
    """
    Decide a cutoff on the *low* side of scores by detecting a large spacing (gap)
    in the sorted scores, but only if the smaller side is <= max_tail_frac of points.

    Returns:
        threshold (float or None),
        info (dict with diagnostics)
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return None, {"reason": "empty"}

    s = np.sort(scores)
    diffs = np.diff(s)  # spacings between consecutive sorted scores

    max_drop = int(np.floor(max_tail_frac * n))
    if max_drop < 1:
        return None, {"reason": "tail<1 point", "max_drop": max_drop}

    # Candidate split indices i such that dropping i+1 points (<= max_drop)
    # uses the gap between s[i] and s[i+1].
    cand_upper = max_drop - 1
    if cand_upper < 0:
        return None, {"reason": "no valid split index"}

    cand_diffs = diffs[: cand_upper + 1]
    if cand_diffs.size == 0:
        return None, {"reason": "no gaps in eligible tail"}

    i_star = int(np.argmax(cand_diffs))
    gap = float(cand_diffs[i_star])
    drop_count = i_star + 1

    # Build a robust baseline of typical gaps excluding zeros (ties are common).
    nonzero = diffs[diffs > 0]
    if nonzero.size == 0:
        # All scores identical; distribution is perfectly "connected"
        return None, {
            "reason": "all scores equal",
            "gap": 0.0,
            "drop_count": drop_count,
        }

    med_gap = float(np.median(nonzero))
    q_gap = float(np.quantile(nonzero, gap_quantile, method="linear"))

    # Absolute minimum gap in terms of discrete score steps of size 1/n_dirs
    # If n_dirs unknown, fall back to a tiny epsilon.
    if n_dirs is None or n_dirs <= 0:
        abs_min_gap = 1e-12
    else:
        abs_min_gap = min_gap_steps / float(n_dirs)

    is_large = (gap >= abs_min_gap) and (gap >= q_gap) and (gap >= gap_ratio * med_gap)

    if not is_large:
        return None, {
            "reason": "no significant gap",
            "gap": gap,
            "abs_min_gap": abs_min_gap,
            "q_gap": q_gap,
            "med_gap": med_gap,
            "drop_count": drop_count,
            "max_drop": max_drop,
        }

    # Threshold midway across the winning gap
    thr = 0.5 * (s[i_star] + s[i_star + 1])
    return thr, {
        "reason": "trim",
        "gap": gap,
        "abs_min_gap": abs_min_gap,
        "q_gap": q_gap,
        "med_gap": med_gap,
        "drop_count": drop_count,
        "max_drop": max_drop,
        "i_star": i_star,
        "threshold": thr,
    }


def trim_by_random_projection_depth(
    X,  # (N, 3)
    keep_alpha=0.95,
    n_dirs=200,
    rng=0,
    robust_scale=True,
    use_partition=True,
    max_tail_frac=0.01,
    gap_quantile=0.90,
    gap_ratio=10.0,
    min_gap_steps=2,
):
    """
    Compute depth scores, choose a dynamic cutoff only if there's a real gap
    and the low-score cluster is <~1% of points, and return a boolean mask.

    Returns:
        mask (bool array of shape (n,)),
        threshold (float or None),
        scores (float array of shape (n,)),
        info (dict)
    """
    scores = random_projection_depth_scores(
        X,
        keep_alpha=keep_alpha,
        n_dirs=n_dirs,
        rng=rng,
        robust_scale=robust_scale,
        use_partition=use_partition,
    )

    thr, info = dynamic_depth_trim_threshold(
        scores,
        max_tail_frac=max_tail_frac,
        gap_quantile=gap_quantile,
        gap_ratio=gap_ratio,
        min_gap_steps=min_gap_steps,
        n_dirs=n_dirs,
    )

    if thr is None:
        mask = np.ones(len(scores), dtype=bool)
    else:
        mask = scores >= thr

    return mask, thr, scores, info


################################################################################


def principal_axes_3d(points, weights=None):
    """
    points: (n,3) array of 3D points
    weights: optional (n,) array of nonnegative weights

    returns:
      mu: (3,) centroid
      V:  (3,3) right singular vectors; columns are principal directions
          V[:,0] = principal axis (eigen-direction with largest variance)
          V[:,2] = smallest-variance direction (plane normal)
      evals: (3,) eigenvalues of the covariance (variances along axes)
      evr:   (3,) explained variance ratios (evals / sum(evals))
    """
    X = np.asarray(points, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("points must be an (n,3) array")

    if weights is None:
        mu = X.mean(axis=0)
        Xc = X - mu
        # SVD of centered data (most stable)
        # Xc = U Σ V^T, eigenvalues of covariance are Σ^2 / (n-1)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        evals = (S**2) / max(len(X) - 1, 1)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != X.shape[0]:
            raise ValueError("weights must have length n")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")
        if w.sum() == 0:
            raise ValueError("weights sum to zero")
        w = w / w.sum()
        mu = (w[:, None] * X).sum(axis=0)
        Xc = X - mu
        # Weighted SVD: scale rows by sqrt(w)
        Xw = np.sqrt(w[:, None]) * Xc
        U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
        # For normalized weights, covariance = Xw^T Xw, so eigenvalues = S^2
        evals = S**2

    V = Vt.T  # columns are directions
    evr = evals / evals.sum() if evals.sum() > 0 else evals

    return mu, V, evals, evr


def best_fit_line(points, weights=None):
    mu, V, *_ = principal_axes_3d(points, weights)
    direction = V[:, 0]  # unit vector (principal eigen-direction)
    return mu, direction  # line: L(t) = mu + t * direction


def best_fit_plane(points, weights=None):
    mu, V, *_ = principal_axes_3d(points, weights)
    normal = V[:, 2]  # unit normal (smallest variance)
    # Plane through mu with normal n: n · (x - mu) = 0
    d = -normal @ mu  # plane in Ax + By + Cz + D = 0 form
    return normal, d, mu


def project_onto_line(points, mu, direction):
    """Return parameters t and projected points on the line."""
    X = np.asarray(points, dtype=float)
    t = (X - mu) @ direction
    Xproj = mu + t[:, None] * direction
    return t, Xproj


def distances_to_line(points, mu, direction):
    t, Xproj = project_onto_line(points, mu, direction)
    return np.linalg.norm(points - Xproj, axis=1)


def distances_to_plane(points, normal, d):
    X = np.asarray(points, dtype=float)
    # normal must be unit-length for true distances
    return np.abs(X @ normal + d)


def simple_trim_outliers(
    pts: np.ndarray,
    low_q: float = 0.5,
    high_q: float = 99.5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Remove outliers from a 3D point cloud using per-axis quantile thresholds.

    Args:
        pts (np.ndarray): Input point cloud of shape (N, 3).
        low_q (float): Lower quantile (default: 0.5).
        high_q (float): Upper quantile (default: 99.5).
        verbose (bool): If True, print quantile values.

    Returns:
        np.ndarray: Boolean mask of inlier points.
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {pts.shape}")
    if not (0 <= low_q < high_q <= 100):
        raise ValueError("Quantiles must satisfy 0 <= low_q < high_q <= 100")

    low_v, high_v = np.percentile(pts, [low_q, high_q], axis=0)
    if verbose:
        print(f"simple_trim_outliers: low_v={low_v}, high_v={high_v}")

    mask = np.all((pts >= low_v) & (pts <= high_v), axis=1)
    return mask


def advanced_trim_outliers(pts: np.ndarray, threshold: int = 95.0):
    """
    Trim outliers from point cloud X.
    """
    n, d = pts.shape
    assert d == 3, "This implementation expects 3D points."
    assert threshold > 0 and threshold < 100, "Threshold must be between 0 and 100"
    # 1. Get the best fit plane
    normal, d, mu = best_fit_plane(pts)
    # 2. Get the distances to the plane
    distances = distances_to_plane(pts, normal, d)
    # 3. Get the mask of points within the threshold
    mask = distances <= np.percentile(distances, threshold)
    return mask


def plane_aligned_bbox(
    points, thickness=None, padding=0.0, normal_padding=0.0, weights=None
):
    """
    Plane-aligned 3D bounding box using the best-fit plane from the PCA helpers.

    Depends on previously-defined:
      - principal_axes_3d(points, weights) -> mu, V, evals, evr
      - best_fit_plane(points, weights)    -> normal, d, mu

    Args:
        points         : (n,3) array of 3D points.
        thickness      : If None, use the data's spread along the plane normal.
                         If a number, make a slab of that thickness centered at the plane.
        padding        : Margin added in-plane (u and v).
        normal_padding : Extra margin along the normal direction if thickness=None.
        weights        : Optional (n,) nonnegative weights.

    Returns:
        dict with:
          - center: (3,) centroid (box center)
          - axes:   (3,3) columns are [u, v, n] (orthonormal; n is plane normal)
          - half_sizes: (3,) half-lengths along [u, v, n]
          - corners8: (8,3) world coords of the box corners
                      order: z- (bottom 4), then z+ (top 4), each ccw in (u,v)
          - rect4:   (4,3) world coords of the in-plane rectangle (z=0 slice)
          - plane:   dict with 'normal', 'd' such that normal·x + d = 0
          - lines12: (12,2,3) array of edge segments for drawing the bbox
    """
    X = np.asarray(points, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("points must be an (n,3) array")

    # ---- Use helpers to get plane + axes ----
    mu, V, _, _ = principal_axes_3d(X, weights)  # columns: principal directions
    u, v, n = V[:, 0], V[:, 1], V[:, 2]
    R = np.column_stack((u, v, n))  # world <- local

    n_plane, d_plane, _ = best_fit_plane(X, weights)  # plane eq: n·x + d = 0

    # ---- Project to local plane frame ----
    X_local = (X - mu) @ R  # columns: [u, v, n]

    # In-plane extents with padding
    umin = X_local[:, 0].min() - padding
    umax = X_local[:, 0].max() + padding
    vmin = X_local[:, 1].min() - padding
    vmax = X_local[:, 1].max() + padding

    # Normal extent
    if thickness is None:
        zmin = X_local[:, 2].min() - normal_padding
        zmax = X_local[:, 2].max() + normal_padding
    else:
        zmin, zmax = -0.5 * float(thickness), 0.5 * float(thickness)

    # Half-sizes
    half_u = 0.5 * (umax - umin)
    half_v = 0.5 * (vmax - vmin)
    half_z = 0.5 * (zmax - zmin)

    # Local corners (z- then z+, CCW in u-v)
    corners_local = np.array(
        [
            [umin, vmin, zmin],
            [umax, vmin, zmin],
            [umax, vmax, zmin],
            [umin, vmax, zmin],
            [umin, vmin, zmax],
            [umax, vmin, zmax],
            [umax, vmax, zmax],
            [umin, vmax, zmax],
        ]
    )

    # Back to world coordinates
    corners8 = mu + corners_local @ R.T

    # In-plane rectangle (z = 0 slice)
    rect4 = (
        mu
        + np.array(
            [
                [umin, vmin, 0.0],
                [umax, vmin, 0.0],
                [umax, vmax, 0.0],
                [umin, vmax, 0.0],
            ]
        )
        @ R.T
    )

    # --- Build 12 edge segments from the 8 corners ---
    # Corner indexing matches corners_local order above.
    edges_idx = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical edges
    ]
    lines12 = np.stack([corners8[[i, j], :] for (i, j) in edges_idx], axis=0)

    return {
        "center": mu,
        "axes": R,  # columns: u, v, n
        "half_sizes": np.array([half_u, half_v, half_z]),
        "corners8": corners8,
        "rect4": rect4,
        "plane": {"normal": n_plane, "d": d_plane},
        "lines12": lines12,
    }


def fit_plane_ransac(
    points,
    max_trials=2000,
    stop_prob=0.99,
    distance_threshold=None,
    initial_inlier_ratio=0.3,
    up_vector=None,
    one_sided=False,
    min_inliers=100,
    random_state=None,
    refine_irls=True,
    eval_subset_size=None,
):
    """
    Robustly fit a plane to 3D points using RANSAC + (optional) IRLS refinement.

    Parameters
    ----------
    points : (N,3) array_like
        Input point cloud.
    max_trials : int
        Hard cap on RANSAC iterations (may adapt lower).
    stop_prob : float
        Desired probability that at least one all-inlier sample was drawn.
    distance_threshold : float or None
        Inlier threshold (orthogonal distance). If None, it is estimated via MAD.
    initial_inlier_ratio : float
        Initial guess for inlier ratio used to adaptively set iterations.
    up_vector : array_like or None
        If provided (e.g., [0,0,1]), the plane normal is oriented to have
        non-negative dot with it. Also enables one-sided scoring if desired.
    one_sided : bool
        If True and up_vector is provided, RANSAC/IRLS treat only points
        on the *above* side (along up_vector) as outliers; points below the
        plane do not increase the residual (great for “stuff on the ground”).
    min_inliers : int
        Minimum number of inliers to accept a candidate model.
    random_state : int or np.random.Generator or None
        RNG for reproducibility.
    refine_irls : bool
        Run a few Tukey IRLS iterations on the RANSAC inliers for extra robustness.
    eval_subset_size : int or None
        If set and smaller than N, score RANSAC candidates on a random subset
        for speed; final inliers are computed on the full set.

    Returns
    -------
    dict with keys:
        'normal' : (3,) unit normal vector [a,b,c]
        'point'  : (3,) a point on the plane (closest point to origin)
        'd'      : float, plane offset so that a*x + b*y + c*z + d = 0
        'inliers': (N,) bool mask of inliers
        'threshold': float, the inlier distance threshold used
        'iterations': int, number of RANSAC iterations performed
        'residuals': (N,) distances used for final inlier test
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    N = P.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 points")
    rng = np.random.default_rng(random_state)

    # center for numerical stability
    median = np.median(P, axis=0)
    P0 = P - median

    uv = None
    if up_vector is not None:
        uv = np.asarray(up_vector, dtype=float)
        n_uv = np.linalg.norm(uv)
        if n_uv >= 1e-12:
            uv = uv / n_uv
        else:
            uv = None

    # optional evaluation subset for speed
    if eval_subset_size is not None and eval_subset_size < N:
        eval_idx = rng.choice(np.arange(N), size=int(eval_subset_size), replace=False)
        P_eval = P0[eval_idx]
    else:
        eval_idx = None
        P_eval = P0

    def plane_from_three(a, b, c):
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            return None
        n = n / norm
        if uv is not None and np.dot(n, uv) < 0:
            n = -n
        d = -np.dot(n, a)
        return n, d

    def residuals_to_plane(n, d, X, use_one_sided=None):
        s = X @ n + d  # signed distance (since ||n||=1)
        if use_one_sided is None:
            use_one_sided = one_sided
        if use_one_sided and uv is not None:
            return np.maximum(s, 0.0)
        return np.abs(s)

    # --- Estimate a reasonable threshold if not provided (via PCA + MAD) ---
    if distance_threshold is None:
        cov = np.cov(P0, rowvar=False)
        _, evecs = np.linalg.eigh(cov)
        n0 = evecs[:, 0]  # smallest variance direction
        if uv is not None and np.dot(n0, uv) < 0:
            n0 = -n0
        d0 = -np.dot(n0, P0.mean(axis=0))
        # important: estimate scale with absolute distances (not one-sided)
        r0 = residuals_to_plane(n0, d0, P0, use_one_sided=False)
        mad = np.median(np.abs(r0 - np.median(r0)))
        sigma = 1.4826 * mad if mad > 0 else np.std(r0)
        if not np.isfinite(sigma) or sigma < 1e-9:
            diam = np.linalg.norm(P0.max(axis=0) - P0.min(axis=0))
            sigma = 0.005 * diam
        distance_threshold = 2.5 * sigma
    distance_threshold = float(distance_threshold)

    # --- RANSAC loop (adaptive number of iterations) ---
    best_model = None
    best_inlier_count = 0
    iters = 0
    s = 3  # sample size
    w = np.clip(initial_inlier_ratio, 1e-6, 1 - 1e-6)

    def compute_max_needed(w):
        denom = max(1 - w**s, 1e-12)
        return int(np.ceil(np.log(1 - stop_prob) / np.log(denom)))

    max_needed = max(100, min(max_trials, compute_max_needed(w)))
    indices = np.arange(N)

    while iters < max_needed and iters < max_trials:
        iters += 1
        # draw a non-degenerate 3-point sample
        valid = False
        for _ in range(5):
            ids = rng.choice(indices, size=3, replace=False)
            pl = plane_from_three(P0[ids[0]], P0[ids[1]], P0[ids[2]])
            if pl is None:
                continue
            n, d = pl
            area = 0.5 * np.linalg.norm(
                np.cross(P0[ids[1]] - P0[ids[0]], P0[ids[2]] - P0[ids[0]])
            )
            if area < 1e-10:
                continue
            valid = True
            break
        if not valid:
            continue

        r_eval = residuals_to_plane(n, d, P_eval)
        inliers_eval = r_eval <= distance_threshold
        count_eval = int(inliers_eval.sum())
        count_est = (
            int(count_eval * (N / len(P_eval))) if eval_idx is not None else count_eval
        )

        if count_est > best_inlier_count and count_eval >= min(10, len(P_eval)):
            best_model = (n, d)
            best_inlier_count = count_est
            w = np.clip(best_inlier_count / N, 1e-6, 1 - 1e-6)
            max_needed = max(50, min(max_trials, compute_max_needed(w)))

    # --- Compute inliers on full set and refine (or PCA fallback) ---
    if best_model is None:
        cov = np.cov(P0, rowvar=False)
        _, evecs = np.linalg.eigh(cov)
        n = evecs[:, 0]
        if uv is not None and np.dot(n, uv) < 0:
            n = -n
        mu = P0.mean(axis=0)
        d = -np.dot(n, mu)
    else:
        n, d = best_model

    resid_full = residuals_to_plane(n, d, P0)
    inliers_full = resid_full <= distance_threshold
    X = P0[inliers_full]

    # IRLS refinement on inliers
    if refine_irls and X.shape[0] >= 3:
        weights = np.ones(X.shape[0])
        for _ in range(3):
            wsum = weights.sum()
            mu = (X * weights[:, None]).sum(axis=0) / max(wsum, 1e-12)
            Xc = X - mu
            C = (Xc * weights[:, None]).T @ Xc / max(wsum, 1e-12)
            _, evecs = np.linalg.eigh(C)
            n = evecs[:, 0]
            if uv is not None and np.dot(n, uv) < 0:
                n = -n
            d = -np.dot(n, mu)
            # use absolute distances to compute robust scale
            res = residuals_to_plane(n, d, X, use_one_sided=False)
            mad = np.median(np.abs(res - np.median(res)))
            sigma = 1.4826 * mad if mad > 0 else np.std(res)
            c = max(4.685 * max(sigma, 1e-12), 1e-6)
            u = res / c
            w_new = (1 - u**2) ** 2
            w_new[np.abs(u) >= 1] = 0.0
            weights = w_new + 1e-6

    # final inliers with refined model
    resid_full = residuals_to_plane(n, d, P0)
    final_inliers = resid_full <= distance_threshold

    # convert plane back to original coordinates (a*x + b*y + c*z + d = 0)
    d_orig = d - float(n @ median)
    point_on_plane = -d_orig * n  # closest point to origin on the plane

    return {
        "normal": n,
        "point": point_on_plane,
        "d": d_orig,
        "inliers": final_inliers,
        "threshold": distance_threshold,
        "iterations": iters,
        "residuals": resid_full,
    }


# --- Convenience helpers ------------------------------------------------------


def plane_signed_distance(points, normal, d):
    """Signed orthogonal distances to plane a*x+b*y+c*z+d=0 (normal must be unit)."""
    P = np.asarray(points, float)
    n = np.asarray(normal, float)
    return P @ n + float(d)


def project_onto_plane(points, normal, d):
    """Orthogonally project 3D points onto the plane."""
    P = np.asarray(points, float)
    n = np.asarray(normal, float)
    s = plane_signed_distance(P, n, d)
    return P - s[:, None] * n


if __name__ == "__main__":
    import time

    n_scale = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    for n in n_scale:
        start_time = time.time()
        X = np.random.randn(n, 3)
        mask, thr, scores, info = trim_by_random_projection_depth(X)
        end_time = time.time()
        print(
            f"n: {n}, mask: {mask.shape}, thr: {thr}, scores: {scores.shape}, info: {info}"
        )
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Time per point: {(end_time - start_time) / n} seconds")
        print()
