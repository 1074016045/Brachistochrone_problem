import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid, solve_bvp, solve_ivp
from scipy.optimize import brentq, minimize

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False
    sp = None

os.makedirs("figures", exist_ok=True)

# Global parameters
g = 9.81
x_B = 10.0
y_B = 5.0
m = 75.0
mu = 0.10
rho = 1.225
C_d = 1.0
A = 0.0163265306122449  # Chosen so 0.5 * rho * C_d * A = 0.01 in the baseline drag sweep.
mu_values = [0.0, 0.05, 0.1, 0.2]
A_values = [0.0, A, 3.0 * A]
combined_drag_pairs = [(0.05, A), (0.10, A), (0.10, 3.0 * A)]

# Numerical controls
N_PATH = 700
N_COEFF = 5
EPS = 1e-10
PENALTY_SCALE = 1e5
MAX_COEFF = 3.0
INVALID_PATH_PENALTY = 1e6

np.set_printoptions(precision=5, suppress=True)

print("Configuration loaded.")
print(f"SymPy available: {SYMPY_AVAILABLE}")

# Helper functions used in multiple sections

def validate_path(x, y, yprime, x_end=x_B, y_end=y_B, tol=1e-6):
    """
    Lightweight geometric checks shared by all numerical sections.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yprime = np.asarray(yprime, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or yprime.ndim != 1:
        raise ValueError("Path arrays must be one-dimensional.")
    if not (len(x) == len(y) == len(yprime)):
        raise ValueError("Path arrays x, y, yprime must have matching lengths.")
    if len(x) < 2:
        raise ValueError("Path must contain at least two points.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(yprime)):
        raise ValueError("Path contains non-finite values.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing along the path.")
    if not np.isclose(x[0], 0.0, atol=tol):
        raise ValueError("Path must start at x=0 within tolerance.")
    if not np.isclose(x[-1], x_end, atol=tol):
        raise ValueError("Path must end at x=x_B within tolerance.")
    if not np.isclose(y[0], 0.0, atol=tol):
        raise ValueError("Path must start at y=0 within tolerance.")
    if not np.isclose(y[-1], y_end, atol=tol):
        raise ValueError("Path must end at y=y_B within tolerance.")

def build_path_from_coeffs(coeffs, x_end=x_B, y_end=y_B, n=N_PATH):
    """
    Build a smooth monotone path y(x) joining (0, 0) to (x_end, y_end).

    Parameterization:
    - Let s in [0,1], x = x_end * s.
    - Define w(s) = exp(p(s)) > 0 from a cosine basis p(s).
    - Set y(s) proportional to integral_0^s w(u) du.

    This guarantees y(0)=0, y(1)=y_end, and dy/dx > 0.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    s = np.linspace(0.0, 1.0, n)
    x = x_end * s

    p = np.zeros_like(s)
    for j, c in enumerate(coeffs, start=1):
        p += c * np.cos(j * np.pi * s)

    p = np.clip(p, -30.0, 30.0)
    w = np.exp(p)

    I = cumulative_trapezoid(w, s, initial=0.0)
    norm = I[-1]
    if norm <= 0:
        raise RuntimeError("Path normalization failed: non-positive integral.")

    y = y_end * I / norm
    dy_ds = y_end * w / norm
    dy_dx = dy_ds / x_end

    validate_path(x, y, dy_dx, x_end=x_end, y_end=y_end, tol=1e-6)
    if np.any(dy_dx <= 0):
        raise ValueError("Constructed path is not monotone increasing in y(x).")

    return x, y, dy_dx

def drag_force_constant(rho_air=rho, drag_coefficient=C_d, area=A):
    """
    Quadratic drag is written as F_drag = -C v^2 with C = 0.5 * rho * C_d * A.
    """
    return 0.5 * rho_air * drag_coefficient * area

def drag_acceleration_constant(rho_air=rho, drag_coefficient=C_d, area=A, mass=m):
    """
    The segment update uses k = C / m so that a_drag = -k v^2.
    """
    return drag_force_constant(rho_air=rho_air, drag_coefficient=drag_coefficient, area=area) / mass

def drag_summary(area, rho_air=rho, drag_coefficient=C_d, mass=m):
    """
    Convenience helper for reporting the derived drag constants C and k = C / m.
    """
    C_force = drag_force_constant(rho_air=rho_air, drag_coefficient=drag_coefficient, area=area)
    return C_force, C_force / mass

def build_segment_geometry(x, y):
    """
    Convert a pointwise path (x_i, y_i) into straight-segment geometry.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have matching lengths for segment geometry.")
    if len(x) < 2:
        raise ValueError("At least two path points are required.")

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)

    sin_theta = np.zeros_like(ds)
    cos_theta = np.zeros_like(ds)
    good = ds > EPS
    sin_theta[good] = dy[good] / ds[good]
    cos_theta[good] = dx[good] / ds[good]
    return dx, dy, ds, sin_theta, cos_theta

def solve_cycloid_parameters(x_end, y_end):
    """
    Solve for cycloid parameters a and theta_f using endpoint conditions:
        x_end = a (theta_f - sin theta_f)
        y_end = a (1 - cos theta_f)
    where a = k/2.
    """
    if x_end <= 0 or y_end <= 0:
        raise ValueError("Cycloid endpoint solver expects x_end > 0 and y_end > 0.")

    def residual(theta):
        denom = 1.0 - np.cos(theta)
        if denom <= 0:
            return np.nan
        a_local = y_end / denom
        return a_local * (theta - np.sin(theta)) - x_end

    thetas = np.linspace(1e-4, 2.0 * np.pi - 1e-4, 4000)
    vals = np.array([residual(t) for t in thetas])

    bracket = None
    for i in range(len(thetas) - 1):
        v1, v2 = vals[i], vals[i + 1]
        if np.isfinite(v1) and np.isfinite(v2) and v1 * v2 < 0:
            bracket = (thetas[i], thetas[i + 1])
            break

    if bracket is None:
        raise RuntimeError("Could not bracket theta_f for cycloid endpoint matching.")

    theta_f = brentq(residual, *bracket)
    a = y_end / (1.0 - np.cos(theta_f))
    k_const = 2.0 * a
    return theta_f, a, k_const

def cycloid_curve(a, theta_f, n=N_PATH):
    theta = np.linspace(0.0, theta_f, n)
    x = a * (theta - np.sin(theta))
    y = a * (1.0 - np.cos(theta))
    return theta, x, y

def travel_time_from_u(x, yprime, u):
    x = np.asarray(x, dtype=float)
    yprime = np.asarray(yprime, dtype=float)
    u = np.asarray(u, dtype=float)
    if not (len(x) == len(yprime) == len(u)):
        raise ValueError("x, yprime, and u=v^2 must have matching lengths.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing for time integration.")
    if np.any(~np.isfinite(u)):
        raise ValueError("u=v^2 contains non-finite values.")

    v = np.sqrt(np.clip(u, EPS, None))
    integrand = np.sqrt(1.0 + yprime**2) / v
    return np.trapezoid(integrand, x)

def compute_discrete_segment_dynamics(
    x,
    y,
    mu=0.0,
    rho_air=0.0,
    drag_coefficient=C_d,
    area=0.0,
    mass=m,
    grav=g,
    eps=EPS,
):
    """
    Shared discrete-segment update used by both travel-time evaluation and
    post-processing motion reconstruction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yprime = np.gradient(y, x)
    validate_path(x, y, yprime)

    dx, dy, ds, sin_theta, cos_theta = build_segment_geometry(x, y)
    C_force = drag_force_constant(rho_air=rho_air, drag_coefficient=drag_coefficient, area=area)
    k_drag = C_force / mass

    v_nodes = np.zeros(len(x), dtype=float)
    u_nodes = np.zeros(len(x), dtype=float)
    segment_acceleration = np.zeros(len(ds), dtype=float)
    segment_time = np.zeros(len(ds), dtype=float)

    valid = True
    penalty = 0.0
    messages = []

    for i in range(len(ds)):
        if ds[i] <= eps:
            messages.append(f"Skipped near-zero segment {i}.")
            v_nodes[i + 1] = v_nodes[i]
            u_nodes[i + 1] = u_nodes[i]
            continue

        # This is the source-of-truth segment model used in the optimizer:
        # gravity drives the motion, friction opposes it through N = m g cos(theta),
        # and air drag removes speed through -k v^2.
        segment_acceleration[i] = (
            grav * sin_theta[i]
            - mu * grav * cos_theta[i]
            - k_drag * v_nodes[i] ** 2
        )

        u_next_raw = v_nodes[i] ** 2 + 2.0 * segment_acceleration[i] * ds[i]
        if not np.isfinite(u_next_raw):
            valid = False
            penalty += INVALID_PATH_PENALTY
            u_next = eps
            messages.append(f"Non-finite velocity update on segment {i}.")
        elif u_next_raw <= eps:
            valid = False
            penalty += INVALID_PATH_PENALTY + PENALTY_SCALE * max(eps - u_next_raw, 0.0) ** 2
            u_next = eps
            messages.append(f"Velocity collapsed on segment {i}.")
        else:
            u_next = u_next_raw

        u_nodes[i + 1] = u_next
        v_nodes[i + 1] = np.sqrt(max(u_next, eps))

        denom = v_nodes[i] + v_nodes[i + 1]
        if denom <= eps:
            valid = False
            penalty += INVALID_PATH_PENALTY
            segment_time[i] = 2.0 * ds[i] / np.sqrt(eps)
            messages.append(f"Degenerate segment-time denominator on segment {i}.")
        else:
            segment_time[i] = 2.0 * ds[i] / denom

    total_time = float(np.sum(segment_time))
    if not np.isfinite(total_time):
        valid = False
        penalty += INVALID_PATH_PENALTY
        total_time = INVALID_PATH_PENALTY
        messages.append("Total time became non-finite.")

    t_nodes = np.concatenate(([0.0], np.cumsum(segment_time)))
    objective = total_time + penalty
    return {
        "x": x,
        "y": y,
        "yprime": yprime,
        "dx": dx,
        "dy": dy,
        "ds": ds,
        "sin_theta": sin_theta,
        "cos_theta": cos_theta,
        "u": u_nodes,
        "v": v_nodes,
        "segment_acceleration": segment_acceleration,
        "segment_time": segment_time,
        "t_nodes": t_nodes,
        "time": total_time,
        "objective": objective,
        "valid": valid,
        "message": " | ".join(messages) if messages else "Discrete segment model completed normally.",
        "C": C_force,
        "k": k_drag,
    }

def travel_time_discrete_segments(
    x,
    y,
    mu=0.0,
    rho_air=0.0,
    drag_coefficient=C_d,
    area=0.0,
    mass=m,
    grav=g,
    eps=EPS,
):
    """
    Main dissipative model: treat the curve as straight line segments and update
    the speed one segment at a time using constant tangential acceleration on
    each segment.
    """
    return compute_discrete_segment_dynamics(
        x,
        y,
        mu=mu,
        rho_air=rho_air,
        drag_coefficient=drag_coefficient,
        area=area,
        mass=mass,
        grav=grav,
        eps=eps,
    )

def simulate_motion_discrete_segments(
    x,
    y,
    mu=0.0,
    rho_air=0.0,
    drag_coefficient=C_d,
    area=0.0,
    mass=m,
    grav=g,
    eps=EPS,
    n_time_samples=700,
    samples_per_segment=None,
    time_samples=None,
):
    """
    Reconstruct the motion in time while keeping the same discrete segment
    physics as the optimizer and travel-time evaluator.
    """
    dynamics = compute_discrete_segment_dynamics(
        x,
        y,
        mu=mu,
        rho_air=rho_air,
        drag_coefficient=drag_coefficient,
        area=area,
        mass=mass,
        grav=grav,
        eps=eps,
    )

    total_time = dynamics["time"]
    t_nodes = dynamics["t_nodes"]
    n_segments = len(dynamics["segment_time"])
    messages = [dynamics["message"]]
    valid = bool(dynamics["valid"])

    if n_segments == 0 or total_time <= eps:
        t_samples = np.array([0.0])
        x_samples = np.array([dynamics["x"][0]])
        y_samples = np.array([dynamics["y"][0]])
        v_samples = np.array([dynamics["v"][0]])
        a_samples = np.array([0.0])
        segment_index = np.array([0], dtype=int)
        if total_time <= eps:
            messages.append("Total travel time is near zero; returned the start point only.")
        return {
            "t_samples": t_samples,
            "x_samples": x_samples,
            "y_samples": y_samples,
            "v_samples": v_samples,
            "a_samples": a_samples,
            "segment_index": segment_index,
            "total_time": total_time,
            "valid": valid and total_time > eps,
            "message": " | ".join(messages),
            "x_path": dynamics["x"],
            "y_path": dynamics["y"],
        }

    if time_samples is not None:
        t_samples = np.asarray(time_samples, dtype=float)
        if t_samples.ndim != 1:
            raise ValueError("time_samples must be one-dimensional.")
        if len(t_samples) == 0:
            t_samples = np.linspace(0.0, total_time, max(int(n_time_samples), 2))
        if np.any(~np.isfinite(t_samples)):
            raise ValueError("time_samples must be finite.")
        if np.any((t_samples < 0.0) | (t_samples > total_time)):
            t_samples = np.clip(t_samples, 0.0, total_time)
            messages.append("Requested time samples were clipped to the simulated time interval.")
    elif samples_per_segment is not None:
        samples_per_segment = max(int(samples_per_segment), 1)
        t_chunks = []
        for i in range(n_segments):
            dt_i = dynamics["segment_time"][i]
            if dt_i <= eps:
                continue
            t_chunks.append(np.linspace(t_nodes[i], t_nodes[i + 1], samples_per_segment, endpoint=False))
        t_chunks.append(np.array([total_time]))
        t_samples = np.unique(np.concatenate(t_chunks)) if t_chunks else np.array([0.0, total_time])
    else:
        t_samples = np.linspace(0.0, total_time, max(int(n_time_samples), 2))

    segment_index = np.searchsorted(t_nodes, t_samples, side="right") - 1
    segment_index = np.clip(segment_index, 0, n_segments - 1)

    tau = t_samples - t_nodes[segment_index]
    tau = np.clip(tau, 0.0, dynamics["segment_time"][segment_index])

    ds = dynamics["ds"][segment_index]
    vi = dynamics["v"][segment_index]
    ai = dynamics["segment_acceleration"][segment_index]
    dx = dynamics["dx"][segment_index]
    dy = dynamics["dy"][segment_index]

    s_local = vi * tau + 0.5 * ai * tau**2
    safe_ds = np.where(ds > eps, ds, 1.0)
    lam = np.clip(s_local / safe_ds, 0.0, 1.0)
    lam = np.where(ds > eps, lam, 0.0)

    x_samples = dynamics["x"][segment_index] + lam * dx
    y_samples = dynamics["y"][segment_index] + lam * dy
    v_samples = np.maximum(vi + ai * tau, 0.0)
    a_samples = ai.copy()

    time_match_error = abs(total_time - t_nodes[-1])
    if time_match_error > 1e-10:
        valid = False
        messages.append(f"Simulator time mismatch detected ({time_match_error:.3e}).")

    return {
        "t_samples": t_samples,
        "x_samples": x_samples,
        "y_samples": y_samples,
        "v_samples": v_samples,
        "a_samples": a_samples,
        "segment_index": segment_index,
        "total_time": total_time,
        "valid": valid,
        "message": " | ".join(messages),
        "x_path": dynamics["x"],
        "y_path": dynamics["y"],
        "segment_time": dynamics["segment_time"],
        "segment_acceleration": dynamics["segment_acceleration"],
        "v_nodes": dynamics["v"],
        "u_nodes": dynamics["u"],
        "time_match_error": time_match_error,
        "C": dynamics["C"],
        "k": dynamics["k"],
    }

def plot_motion_snapshots(
    x_path,
    y_path,
    simulation_result,
    num_snapshots=8,
    ax=None,
    title="Motion Snapshots Along the Path",
):
    """
    Show a few time-ordered points moving along the path.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
    else:
        fig = ax.figure

    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)
    x_samples = np.asarray(simulation_result["x_samples"], dtype=float)
    y_samples = np.asarray(simulation_result["y_samples"], dtype=float)
    t_samples = np.asarray(simulation_result["t_samples"], dtype=float)

    ax.plot(x_path, y_path, color="tab:gray", lw=2.5, label="Path")
    ax.scatter([x_path[0], x_path[-1]], [y_path[0], y_path[-1]], c="k", s=50, zorder=5)

    snap_count = min(max(int(num_snapshots), 2), len(t_samples))
    snap_idx = np.linspace(0, len(t_samples) - 1, snap_count, dtype=int)
    scatter = ax.scatter(
        x_samples[snap_idx],
        y_samples[snap_idx],
        c=t_samples[snap_idx],
        cmap="viridis",
        s=60,
        zorder=6,
        label="Time samples",
    )

    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$ (positive downward)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="time (s)")
    fig.tight_layout()
    return fig, ax

def plot_x_y_v_vs_time(simulation_result, include_acceleration=True, title_prefix="Motion Reconstruction"):
    """
    Plot the reconstructed coordinates and speed as functions of time.
    """
    nrows = 4 if include_acceleration else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(8.2, 7.6), sharex=True)
    if nrows == 1:
        axes = [axes]

    t = np.asarray(simulation_result["t_samples"], dtype=float)
    x = np.asarray(simulation_result["x_samples"], dtype=float)
    y = np.asarray(simulation_result["y_samples"], dtype=float)
    v = np.asarray(simulation_result["v_samples"], dtype=float)
    a = np.asarray(simulation_result["a_samples"], dtype=float)

    axes[0].plot(t, x, lw=2.1, color="tab:blue")
    axes[0].set_ylabel("$x(t)$")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, y, lw=2.1, color="tab:orange")
    axes[1].set_ylabel("$y(t)$")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, v, lw=2.1, color="tab:green")
    axes[2].set_ylabel("$v(t)$")
    axes[2].grid(alpha=0.25)

    if include_acceleration:
        axes[3].plot(t, a, lw=1.8, color="tab:red")
        axes[3].set_ylabel("$a(t)$")
        axes[3].grid(alpha=0.25)
        axes[3].set_xlabel("time (s)")
    else:
        axes[2].set_xlabel("time (s)")

    axes[0].set_title(title_prefix)
    fig.tight_layout()
    return fig, axes

def compute_energy_budget_discrete_segments(
    x,
    y,
    mu=0.0,
    rho_air=0.0,
    drag_coefficient=C_d,
    area=0.0,
    mass=m,
    grav=g,
    eps=EPS,
):
    """
    Compute a segment-wise energy budget using the same discrete update that
    drives the path optimization and motion simulation.
    """
    dynamics = compute_discrete_segment_dynamics(
        x,
        y,
        mu=mu,
        rho_air=rho_air,
        drag_coefficient=drag_coefficient,
        area=area,
        mass=mass,
        grav=grav,
        eps=eps,
    )

    x = dynamics["x"]
    y = dynamics["y"]
    ds = dynamics["ds"]
    dy = dynamics["dy"]
    cos_theta = dynamics["cos_theta"]
    v_nodes = dynamics["v"]
    C_force = dynamics["C"]

    kinetic_nodes = 0.5 * mass * np.maximum(dynamics["u"], 0.0)
    potential_drop_nodes = mass * grav * np.maximum(y, 0.0)

    gravity_work_segments = mass * grav * np.maximum(dy, 0.0)
    friction_loss_segments = mu * mass * grav * cos_theta * ds
    drag_loss_segments = C_force * v_nodes[:-1] ** 2 * ds
    dissipation_segments = friction_loss_segments + drag_loss_segments
    kinetic_change_segments = np.diff(kinetic_nodes)

    energy_balance_residual = kinetic_change_segments - (
        gravity_work_segments - dissipation_segments
    )

    cumulative_gravity_work = np.concatenate(([0.0], np.cumsum(gravity_work_segments)))
    cumulative_friction_loss = np.concatenate(([0.0], np.cumsum(friction_loss_segments)))
    cumulative_drag_loss = np.concatenate(([0.0], np.cumsum(drag_loss_segments)))
    cumulative_dissipation = np.concatenate(([0.0], np.cumsum(dissipation_segments)))
    cumulative_residual = np.concatenate(([0.0], np.cumsum(energy_balance_residual)))

    return {
        "x": x,
        "y": y,
        "t_nodes": dynamics["t_nodes"],
        "kinetic_nodes": kinetic_nodes,
        "potential_drop_nodes": potential_drop_nodes,
        "gravity_work_segments": gravity_work_segments,
        "friction_loss_segments": friction_loss_segments,
        "drag_loss_segments": drag_loss_segments,
        "dissipation_segments": dissipation_segments,
        "kinetic_change_segments": kinetic_change_segments,
        "energy_balance_residual": energy_balance_residual,
        "cumulative_gravity_work": cumulative_gravity_work,
        "cumulative_friction_loss": cumulative_friction_loss,
        "cumulative_drag_loss": cumulative_drag_loss,
        "cumulative_dissipation": cumulative_dissipation,
        "cumulative_residual": cumulative_residual,
        "total_gravity_work": float(np.sum(gravity_work_segments)),
        "total_friction_loss": float(np.sum(friction_loss_segments)),
        "total_drag_loss": float(np.sum(drag_loss_segments)),
        "total_dissipation": float(np.sum(dissipation_segments)),
        "final_kinetic_energy": float(kinetic_nodes[-1]),
        "valid": dynamics["valid"],
        "message": dynamics["message"],
        "C": dynamics["C"],
        "k": dynamics["k"],
    }

def plot_energy_budget_vs_time(energy_result, title="Energy Budget Along the Motion"):
    """
    Plot the cumulative energy bookkeeping based on the discrete segment model.
    """
    t = np.asarray(energy_result["t_nodes"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)

    axes[0].plot(t, energy_result["kinetic_nodes"], lw=2.1, color="tab:blue", label="Kinetic energy")
    axes[0].plot(t, energy_result["potential_drop_nodes"], lw=2.1, color="tab:orange", label="Potential energy drop")
    axes[0].set_ylabel("Energy (J)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(t, energy_result["cumulative_gravity_work"], lw=2.0, color="tab:green", label="Cumulative gravity work")
    axes[1].plot(t, energy_result["cumulative_friction_loss"], lw=2.0, color="tab:red", label="Friction loss")
    axes[1].plot(t, energy_result["cumulative_drag_loss"], lw=2.0, color="tab:purple", label="Drag loss")
    axes[1].plot(t, energy_result["cumulative_dissipation"], "--", lw=2.0, color="0.25", label="Total dissipation")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("Energy (J)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    return fig, axes

def plot_dissipation_comparison(case_labels, energy_results, title="Energy Dissipation Across Representative Motions"):
    """
    Compare cumulative dissipation against path position for representative cases.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for label in case_labels:
        x_nodes = np.asarray(energy_results[label]["x"], dtype=float)
        cumulative_dissipation = np.asarray(energy_results[label]["cumulative_dissipation"], dtype=float)
        ax.plot(x_nodes, cumulative_dissipation, lw=2.2, label=label)
        ax.scatter([x_nodes[-1]], [cumulative_dissipation[-1]], s=35, zorder=5)

    ax.set_xlabel("$x$")
    ax.set_ylabel("Cumulative dissipated energy (J)")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(loc="best")

    fig.tight_layout()
    return fig, ax

def integrate_u_profile(x, yprime, mu=0.0, k_drag=0.0, mass=m, grav=g):
    """
    Integrate u(x) = v(x)^2 from the linear ODE obtained from the model:

        du/dx = 2 g (y' - mu) - 2 (k_drag / m) u sqrt(1 + y'^2)

    For drag-only set mu=0. For friction-only set k_drag=0.
    """
    x = np.asarray(x, dtype=float)
    yprime = np.asarray(yprime, dtype=float)
    if len(x) != len(yprime):
        raise ValueError("x and yprime must have matching lengths in velocity ODE integration.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing in velocity ODE integration.")

    yp_interp = lambda xq: np.interp(xq, x, yprime)

    def rhs(xq, u_vec):
        yp = float(yp_interp(xq))
        a_term = 2.0 * (k_drag / mass) * np.sqrt(1.0 + yp**2)
        b_term = 2.0 * grav * (yp - mu)
        return [b_term - a_term * u_vec[0]]

    sol = solve_ivp(
        rhs,
        (x[0], x[-1]),
        y0=[0.0],
        t_eval=x,
        max_step=(x[-1] - x[0]) / 200.0,
        rtol=1e-7,
        atol=1e-9,
    )

    if sol.success and sol.y.shape[1] == len(x):
        u = sol.y[0]
        if np.any(~np.isfinite(u)):
            raise ValueError("Velocity ODE returned non-finite u=v^2 values.")
        return np.clip(u, 0.0, None), True

    # Deterministic fallback: explicit first-order integration
    u = np.zeros_like(x)
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        yp = yprime[i - 1]
        a_term = 2.0 * (k_drag / mass) * np.sqrt(1.0 + yp**2)
        b_term = 2.0 * grav * (yp - mu)
        u[i] = u[i - 1] + dx * (b_term - a_term * u[i - 1])
        if u[i] < 0:
            u[i] = 0.0

    return u, False

def friction_time_for_coeffs(coeffs, mu):
    x, y, yp = build_path_from_coeffs(coeffs)
    segment_result = travel_time_discrete_segments(
        x,
        y,
        mu=mu,
        rho_air=0.0,
        drag_coefficient=C_d,
        area=0.0,
        mass=m,
        grav=g,
        eps=EPS,
    )
    return {
        "objective": segment_result["objective"],
        "time": segment_result["time"],
        "x": x,
        "y": y,
        "yp": yp,
        "u": segment_result["u"],
        "v": segment_result["v"],
        "segment_time": segment_result["segment_time"],
        "segment_acceleration": segment_result["segment_acceleration"],
        "ds": segment_result["ds"],
        "sin_theta": segment_result["sin_theta"],
        "cos_theta": segment_result["cos_theta"],
        "valid": segment_result["valid"],
        "physics_message": segment_result["message"],
        "C": segment_result["C"],
        "k": segment_result["k"],
        "rho": 0.0,
        "C_d": C_d,
        "A": 0.0,
        "mu": mu,
    }

def drag_or_combined_time_for_coeffs(coeffs, mu, rho_air, drag_coefficient, area):
    x, y, yp = build_path_from_coeffs(coeffs)
    segment_result = travel_time_discrete_segments(
        x,
        y,
        mu=mu,
        rho_air=rho_air,
        drag_coefficient=drag_coefficient,
        area=area,
        mass=m,
        grav=g,
        eps=EPS,
    )
    return {
        "objective": segment_result["objective"],
        "time": segment_result["time"],
        "x": x,
        "y": y,
        "yp": yp,
        "u": segment_result["u"],
        "v": segment_result["v"],
        "segment_time": segment_result["segment_time"],
        "segment_acceleration": segment_result["segment_acceleration"],
        "ds": segment_result["ds"],
        "sin_theta": segment_result["sin_theta"],
        "cos_theta": segment_result["cos_theta"],
        "valid": segment_result["valid"],
        "physics_message": segment_result["message"],
        "C": segment_result["C"],
        "k": segment_result["k"],
        "rho": rho_air,
        "C_d": drag_coefficient,
        "A": area,
        "mu": mu,
        "solver_ok": False,
    }

def optimize_path_for_case(mu, rho_air, drag_coefficient, area, initial_guess=None, maxiter=500):
    if initial_guess is None:
        initial_guess = np.zeros(N_COEFF)

    bounds = [(-MAX_COEFF, MAX_COEFF)] * N_COEFF

    def objective(c):
        result = drag_or_combined_time_for_coeffs(
            c,
            mu=mu,
            rho_air=rho_air,
            drag_coefficient=drag_coefficient,
            area=area,
        )
        return result["objective"]

    res = minimize(
        objective,
        x0=np.array(initial_guess, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10},
    )

    coeffs = res.x if res.success else np.array(initial_guess, dtype=float)
    result = drag_or_combined_time_for_coeffs(
        coeffs,
        mu=mu,
        rho_air=rho_air,
        drag_coefficient=drag_coefficient,
        area=area,
    )
    result.update(
        {
            "success": bool(res.success),
            "message": str(res.message),
            "coeffs": coeffs,
        }
    )
    return result

def optimize_path_for_friction(mu, initial_guess=None):
    return optimize_path_for_case(
        mu=mu,
        rho_air=0.0,
        drag_coefficient=C_d,
        area=0.0,
        initial_guess=initial_guess,
        maxiter=400,
    )

def attempt_friction_bvp(mu, x_end=x_B, y_end=y_B):
    """
    Attempt a direct BVP solve of the Euler-Lagrange ODE using a small-x regularization.
    This does not alter the derivation; it only avoids the x=0 singular denominator.
    """
    x_eps = 1e-3 * x_end
    y_eps = max(mu * x_eps + 5e-4, y_end * x_eps / x_end)

    x_mesh = np.linspace(x_eps, x_end, 260)
    y_guess = y_eps + (y_end - y_eps) * (x_mesh - x_eps) / (x_end - x_eps)
    yp_guess = np.gradient(y_guess, x_mesh)

    def ode(x, z):
        y_loc = z[0]
        yp_loc = z[1]
        denom = np.clip(y_loc - mu * x, 1e-8, None)
        ypp = -((1.0 + yp_loc**2) * (1.0 - mu * yp_loc)) / (2.0 * denom)
        return np.vstack((yp_loc, ypp))

    def bc(z_a, z_b):
        return np.array([
            z_a[0] - y_eps,
            z_b[0] - y_end,
        ])

    try:
        sol = solve_bvp(
            ode,
            bc,
            x_mesh,
            np.vstack((y_guess, yp_guess)),
            tol=1e-4,
            max_nodes=8000,
            verbose=0,
        )
    except Exception:
        return {"success": False, "x": None, "y": None, "yp": None, "message": "solve_bvp raised exception"}

    if not sol.success:
        return {"success": False, "x": None, "y": None, "yp": None, "message": str(sol.message)}

    x_bvp = np.linspace(x_eps, x_end, N_PATH)
    y_bvp = sol.sol(x_bvp)[0]
    yp_bvp = sol.sol(x_bvp)[1]

    # Prepend start point for plotting continuity
    x_full = np.concatenate(([0.0], x_bvp))
    y_full = np.concatenate(([0.0], y_bvp))
    yp_full = np.concatenate(([yp_bvp[0]], yp_bvp))

    try:
        validate_path(x_full, y_full, yp_full, x_end=x_end, y_end=y_end, tol=5e-3)
    except ValueError:
        return {"success": False, "x": None, "y": None, "yp": None, "message": "BVP path failed endpoint/shape checks"}

    # Physical admissibility check for friction model denominator
    if np.any((y_full - mu * x_full) <= 0):
        return {"success": False, "x": None, "y": None, "yp": None, "message": "BVP path violated y-mu*x > 0"}

    return {
        "success": True,
        "x": x_full,
        "y": y_full,
        "yp": yp_full,
        "message": "BVP converged under near-start regularization",
    }

def optimize_path_for_drag_or_combined(mu, rho_air=rho, drag_coefficient=C_d, area=A, initial_guess=None):
    return optimize_path_for_case(
        mu=mu,
        rho_air=rho_air,
        drag_coefficient=drag_coefficient,
        area=area,
        initial_guess=initial_guess,
        maxiter=500,
    )

# %%
# Geometry illustration plot
x_geom = np.linspace(0.0, x_B, 300)
y_geom = y_B * (x_geom / x_B) ** 0.72

i_mid = 140
dx_local = x_geom[1] - x_geom[0]
yp_geom = np.gradient(y_geom, x_geom)

x0 = x_geom[i_mid]
y0 = y_geom[i_mid]
slope = yp_geom[i_mid]
alpha = np.arctan(slope)

# Tiny arc element around the chosen point
half = 7
idx1 = i_mid - half
idx2 = i_mid + half
x_seg = x_geom[idx1:idx2 + 1]
y_seg = y_geom[idx1:idx2 + 1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_geom, y_geom, lw=2.5, label="Path $y(x)$")
ax.scatter([0, x_B], [0, y_B], s=60, zorder=5)
ax.text(0, 0, "  A", fontsize=11, va="bottom")
ax.text(x_B, y_B, "  B", fontsize=11, va="bottom")

ax.plot(x_seg, y_seg, color="tab:red", lw=3.5, label="Infinitesimal arc $ds$")

# Tangent line sketch
xt = np.array([x0 - 1.3, x0 + 1.3])
yt = y0 + slope * (xt - x0)
ax.plot(xt, yt, "--", color="tab:green", lw=1.8, label="Tangent")

# Coordinate hints
ax.annotate("$dx$", xy=(x0 + 0.7, y0 + 0.55), xytext=(x0 + 0.15, y0 + 0.55),
            arrowprops=dict(arrowstyle="<->", lw=1.0), fontsize=10)
ax.annotate("$dy$", xy=(x0 + 0.85, y0 + 0.55), xytext=(x0 + 0.85, y0 + 0.05),
            arrowprops=dict(arrowstyle="<->", lw=1.0), fontsize=10)

# Angle marker
angle_r = 0.7
theta_arc = np.linspace(0.0, alpha, 100)
ax.plot(x0 + angle_r * np.cos(theta_arc), y0 + angle_r * np.sin(theta_arc), color="k", lw=1.2)
ax.text(x0 + 0.8, y0 + 0.15, r"$\alpha$", fontsize=11)
ax.text(x0 - 2.1, y0 - 0.2, r"$\tan\alpha = y'$", fontsize=11)

ax.set_title("Geometry of an Infinitesimal Segment Along $y(x)$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$ (positive downward)")
ax.grid(alpha=0.25)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("figures/01_geometry_setup.png", dpi=170)

# %%
# Symbolic check for the frictionless Beltrami manipulation
if SYMPY_AVAILABLE:
    y_sym, yp_sym, g_sym, C_sym, k_sym = sp.symbols("y yp g C k", positive=True)

    L_sym = sp.sqrt(1 + yp_sym**2) / sp.sqrt(2 * g_sym * y_sym)
    beltrami_expr = sp.simplify(L_sym - yp_sym * sp.diff(L_sym, yp_sym))
    expected_expr = 1 / (sp.sqrt(2 * g_sym * y_sym) * sp.sqrt(1 + yp_sym**2))

    check_beltrami = sp.simplify(beltrami_expr - expected_expr)

    print("Beltrami expression:")
    print(beltrami_expr)
    print("Check (should be 0):", check_beltrami)

    # Rearrangement to y(1 + yp^2) = k
    # From expected_expr = C => 1/(2gy(1+yp^2)) = C^2 => y(1+yp^2)=1/(2gC^2)=k
    implied_k = sp.simplify(1 / (2 * g_sym * C_sym**2))
    lhs_k = y_sym * (1 + yp_sym**2)
    print("From the constant relation, k =", implied_k)
    print("Canonical first integral form: y*(1 + yp^2) = k")
else:
    print("SymPy is not available in this environment; symbolic check skipped.")

# %%
# Numerical endpoint matching for the cycloid + frictionless path comparison

theta_f, a_cyc, k_cyc = solve_cycloid_parameters(x_B, y_B)
theta, x_cyc, y_cyc = cycloid_curve(a_cyc, theta_f, n=N_PATH)
yp_cyc = np.gradient(y_cyc, x_cyc, edge_order=2)
validate_path(x_cyc, y_cyc, yp_cyc, x_end=x_B, y_end=y_B, tol=5e-6)
assert np.isclose(x_cyc[-1], x_B, atol=5e-6), "Cycloid endpoint x mismatch."
assert np.isclose(y_cyc[-1], y_B, atol=5e-6), "Cycloid endpoint y mismatch."

# Straight-line comparator
x_line = np.linspace(0.0, x_B, N_PATH)
y_line = y_B * (x_line / x_B)
yp_line = np.full_like(x_line, y_B / x_B)
validate_path(x_line, y_line, yp_line, x_end=x_B, y_end=y_B, tol=5e-10)

# Frictionless travel times
T_cycloid_analytic = np.sqrt(a_cyc / g) * theta_f
u_cyc = 2.0 * g * np.clip(y_cyc, EPS, None)
T_cycloid_numeric = travel_time_from_u(x_cyc, yp_cyc, u_cyc)
assert np.all(u_cyc >= 0.0), "Frictionless cycloid produced negative u=v^2."

u_line = 2.0 * g * np.clip(y_line, EPS, None)
T_line_numeric = travel_time_from_u(x_line, yp_line, u_line)
assert np.all(u_line >= 0.0), "Straight-line comparator produced negative u=v^2."

print(f"Cycloid endpoint solution: theta_f = {theta_f:.6f} rad, a = {a_cyc:.6f}, k = {k_cyc:.6f}")
print(f"Frictionless travel time (cycloid, analytic): {T_cycloid_analytic:.6f} s")
print(f"Frictionless travel time (cycloid, numeric):  {T_cycloid_numeric:.6f} s")
print(f"Frictionless travel time (straight line):      {T_line_numeric:.6f} s")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_cyc, y_cyc, lw=2.8, label="Cycloid (frictionless optimum)")
ax.plot(x_line, y_line, "--", lw=2.2, label="Straight line")
ax.scatter([0, x_B], [0, y_B], c="k", s=55, zorder=5)
ax.set_title("Frictionless Case: Cycloid vs Straight Line")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$ (positive downward)")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig("figures/02_frictionless_paths.png", dpi=170)

fig, ax = plt.subplots(figsize=(6.8, 4.6))
labels = ["Cycloid", "Straight line"]
times = [T_cycloid_analytic, T_line_numeric]
x_pos = np.arange(len(labels))
ax.plot(x_pos, times, "o-", lw=2.2, color="tab:blue")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel("Travel time (s)")
ax.set_title("Frictionless Travel-Time Comparison")
for i, t in enumerate(times):
    ax.text(i, t + 0.02 * max(times), f"{t:.3f} s", ha="center", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("figures/03_frictionless_time_comparison.png", dpi=170)

# %%
# Dedicated free-body style diagram for the friction derivation
alpha_f = np.deg2rad(30.0)
t_hat = np.array([np.cos(alpha_f), -np.sin(alpha_f)])   # Tangent direction (motion down the slope)
n_hat = np.array([np.sin(alpha_f),  np.cos(alpha_f)])   # Outward normal

P = np.array([0.0, 0.0])
mg_vec = np.array([0.0, -1.75])                         # Gravity
mg_sin = 1.1 * t_hat                                    # Tangential gravity component
mg_cos_inward = -0.95 * n_hat                           # Normal gravity component into surface
N_vec = 0.95 * n_hat                                    # Normal reaction N = mg cos(alpha)
fric_vec = -0.55 * t_hat                                # Friction mu*N opposite motion

fig, ax = plt.subplots(figsize=(8.4, 5.4))

# Inclined track and local axes
track_s = np.linspace(-2.8, 2.8, 2)
track_xy = P[:, None] + np.outer(t_hat, track_s)
ax.plot(track_xy[0], track_xy[1], color="0.35", lw=3, label="Track tangent")

normal_s = np.linspace(-1.1, 1.1, 2)
normal_xy = P[:, None] + np.outer(n_hat, normal_s)
ax.plot(normal_xy[0], normal_xy[1], color="0.75", lw=1.8, linestyle="--", label="Normal direction")

ax.scatter([P[0]], [P[1]], s=70, color="k", zorder=5)
ax.text(P[0] + 0.08, P[1] + 0.08, "Particle", fontsize=10)

def draw_vec(vec, color, label, text_shift=(0.04, 0.04), lw=2.2):
    ax.arrow(P[0], P[1], vec[0], vec[1], head_width=0.08, head_length=0.12,
             length_includes_head=True, linewidth=lw, color=color)
    end = P + vec
    ax.text(end[0] + text_shift[0], end[1] + text_shift[1], label, color=color, fontsize=10)

draw_vec(mg_vec, "tab:blue", r"$mg$", text_shift=(0.04, -0.14), lw=2.4)
draw_vec(mg_sin, "tab:green", r"$mg\sin\alpha$", text_shift=(0.05, 0.02))
draw_vec(mg_cos_inward, "tab:olive", r"$mg\cos\alpha$ (into surface)", text_shift=(-0.8, -0.02))
draw_vec(N_vec, "tab:orange", r"$N=mg\cos\alpha$", text_shift=(0.03, 0.03))
draw_vec(fric_vec, "tab:red", r"$\mu N$", text_shift=(-0.32, -0.1))

# Angle alpha marker
phi = np.linspace(-alpha_f, 0.0, 120)
r = 0.55
ax.plot(r * np.cos(phi), r * np.sin(phi), color="k", lw=1.3)
ax.text(0.52, -0.15, r"$\alpha$", fontsize=11)

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-2.4, 2.8)
ax.set_ylim(-2.2, 1.8)
ax.set_title("Local Tangent/Normal Force Decomposition (Friction Case)")
ax.set_xlabel("Local horizontal axis")
ax.set_ylabel("Local vertical axis")
ax.grid(alpha=0.22)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("figures/04_friction_force_diagram.png", dpi=180)

# %%
# Symbolic/semi-symbolic check for the friction Euler-Lagrange equation
if SYMPY_AVAILABLE:
    x = sp.symbols("x", real=True)
    mu_sym, g_sym = sp.symbols("mu g", positive=True)
    y_fun = sp.Function("y")(x)
    yp_fun = sp.diff(y_fun, x)
    ypp_fun = sp.diff(y_fun, x, 2)

    L_fric = sp.sqrt(1 + yp_fun**2) / sp.sqrt(2 * g_sym * (y_fun - mu_sym * x))
    EL = sp.simplify(sp.diff(sp.diff(L_fric, yp_fun), x) - sp.diff(L_fric, y_fun))

    scale = 2 * sp.sqrt(2 * g_sym) * (y_fun - mu_sym * x) ** sp.Rational(3, 2) * (1 + yp_fun**2) ** sp.Rational(3, 2)
    scaled_EL = sp.simplify(sp.expand(EL * scale))

    target = 2 * (y_fun - mu_sym * x) * ypp_fun + (1 + yp_fun**2) * (1 - mu_sym * yp_fun)
    diff_expr = sp.simplify(sp.expand(scaled_EL - target))

    print("Scaled Euler-Lagrange expression:")
    print(scaled_EL)
    print("Target expression:")
    print(target)
    print("Difference (should be 0):")
    print(diff_expr)
else:
    print("SymPy is not available in this environment; symbolic friction check skipped.")

# %%
# Numerical friction-case optimization
# We first attempt a direct BVP solve of the derived ODE (with a near-start regularization),
# then use robust direct path optimization as fallback/primary production method.
# This keeps the derivation unchanged and only adjusts numerics for stability.

friction_results = {}
bvp_results = {}
initial = np.zeros(N_COEFF)

for mu in mu_values:
    bvp_results[mu] = attempt_friction_bvp(mu)
    result = optimize_path_for_friction(mu, initial_guess=initial)
    validate_path(result["x"], result["y"], result["yp"], x_end=x_B, y_end=y_B, tol=2e-5)
    if not result["valid"]:
        print(f"Warning: discrete friction model flagged an invalid path for mu={mu:.3f}. {result['physics_message']}")
    friction_results[mu] = result
    initial = result["coeffs"]  # continuation in mu improves stability

print("Friction optimization status by mu:")
for mu in mu_values:
    r = friction_results[mu]
    b = bvp_results[mu]
    print(
        f"mu={mu:0.3f}: BVP success={b['success']}, "
        f"optimization success={r['success']}, time={r['time']:.6f} s"
    )

fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(x_cyc, y_cyc, lw=3.0, color="black", label="Frictionless cycloid")
for mu in mu_values:
    r = friction_results[mu]
    ax.plot(r["x"], r["y"], lw=2.2, label=f"Optimized with friction $\\mu={mu:.2f}$")

mu_demo = 0.10
if bvp_results[mu_demo]["success"]:
    b = bvp_results[mu_demo]
    ax.plot(
        b["x"],
        b["y"],
        "--",
        lw=1.8,
        color="tab:gray",
        label=f"Direct ODE BVP attempt ($\\mu={mu_demo:.2f}$)",
    )

ax.scatter([0, x_B], [0, y_B], c="k", s=55, zorder=5)
ax.set_title("Path Shape with Kinetic Friction")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$ (positive downward)")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("figures/04_friction_paths.png", dpi=170)

mus_plot = np.array(mu_values)
times_friction = np.array([friction_results[mu]["time"] for mu in mu_values])

fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.plot(mus_plot, times_friction, "o-", lw=2.2)
ax.set_title("Travel Time vs Friction Coefficient")
ax.set_xlabel("$\mu$")
ax.set_ylabel("Optimized travel time (s)")
ax.grid(alpha=0.28)
fig.tight_layout()
fig.savefig("figures/05_friction_time_vs_mu.png", dpi=170)

# %%
# Numerical optimization for drag-only case (mu = 0)

drag_results = {}
initial = np.zeros(N_COEFF)

for area_drag in A_values:
    result = optimize_path_for_drag_or_combined(
        mu=0.0,
        rho_air=rho,
        drag_coefficient=C_d,
        area=area_drag,
        initial_guess=initial,
    )
    validate_path(result["x"], result["y"], result["yp"], x_end=x_B, y_end=y_B, tol=2e-5)
    if not result["valid"]:
        C_force, k_drag = drag_summary(area_drag, rho_air=rho, drag_coefficient=C_d, mass=m)
        print(
            f"Warning: discrete drag model flagged an invalid path for "
            f"A={area_drag:.5f}, C={C_force:.5f}, k={k_drag:.6f}. {result['physics_message']}"
        )
    drag_results[area_drag] = result
    initial = result["coeffs"]

print("Drag-only optimization status by derived drag constant k = C / m:")
for area_drag in A_values:
    r = drag_results[area_drag]
    print(
        f"A={area_drag:0.5f}, C={r['C']:.5f}, k={r['k']:.6f}: "
        f"success={r['success']}, valid={r['valid']}, time={r['time']:.6f} s"
    )

fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(x_cyc, y_cyc, color="black", lw=3.0, label="Frictionless cycloid")
for area_drag in A_values:
    r = drag_results[area_drag]
    ax.plot(r["x"], r["y"], lw=2.2, label=f"Optimized with drag $k={r['k']:.4f}$")

ax.scatter([0, x_B], [0, y_B], c="k", s=55, zorder=5)
ax.set_title("Optimized Paths with Quadratic Air Resistance")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$ (positive downward)")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("figures/06_drag_paths.png", dpi=170)

fig, ax = plt.subplots(figsize=(7.4, 4.8))
k_plot = np.array([drag_results[area_drag]["k"] for area_drag in A_values])
td_plot = np.array([drag_results[area_drag]["time"] for area_drag in A_values])
ax.plot(k_plot, td_plot, "o-", lw=2.2, color="tab:red")
ax.set_title("Travel Time vs Derived Drag Constant")
ax.set_xlabel("$k = C/m$")
ax.set_ylabel("Optimized travel time (s)")
ax.grid(alpha=0.28)
fig.tight_layout()
fig.savefig("figures/07_drag_time_vs_kdrag.png", dpi=170)

# %%
# Numerical optimization for combined friction + drag
combined_results = {}
initial = np.zeros(N_COEFF)

for mu_c, area_c in combined_drag_pairs:
    result = optimize_path_for_drag_or_combined(
        mu=mu_c,
        rho_air=rho,
        drag_coefficient=C_d,
        area=area_c,
        initial_guess=initial,
    )
    validate_path(result["x"], result["y"], result["yp"], x_end=x_B, y_end=y_B, tol=2e-5)
    if not result["valid"]:
        print(
            f"Warning: discrete combined model flagged an invalid path for "
            f"mu={mu_c:.3f}, A={area_c:.5f}, k={result['k']:.6f}. {result['physics_message']}"
        )
    combined_results[(mu_c, area_c)] = result
    initial = result["coeffs"]

print("Combined-case optimization status:")
for (mu_c, area_c), r in combined_results.items():
    print(
        f"mu={mu_c:.2f}, A={area_c:.5f}, C={r['C']:.5f}, k={r['k']:.6f}: "
        f"success={r['success']}, valid={r['valid']}, time={r['time']:.6f} s"
    )

# Choose one representative from each scenario for a 4-way path comparison
mu_rep = 0.10
area_rep = 3.0 * A

fric_rep = friction_results[mu_rep]
drag_rep = drag_results[area_rep]
comb_rep = combined_results[(0.10, area_rep)]

fig, ax = plt.subplots(figsize=(8.7, 5.4))
ax.plot(x_cyc, y_cyc, color="black", lw=3.0, label="1) Frictionless cycloid")
ax.plot(fric_rep["x"], fric_rep["y"], lw=2.2, label=f"2) Friction only ($\\mu={mu_rep:.2f}$)")
ax.plot(drag_rep["x"], drag_rep["y"], lw=2.2, label=f"3) Drag only ($k={drag_rep['k']:.4f}$)")
ax.plot(comb_rep["x"], comb_rep["y"], lw=2.2, label=f"4) Friction+Drag ($\\mu=0.10,\ k={comb_rep['k']:.4f}$)")

ax.scatter([0, x_B], [0, y_B], c="k", s=55, zorder=5)
ax.set_title("Comparison of Optimal Paths Across Model Assumptions")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$ (positive downward)")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("figures/08_four_case_path_comparison.png", dpi=170)

# Summary travel-time figure across model assumptions
summary_labels = [
    "Frictionless\nCycloid",
    "Friction only\n(mu=0.10)",
    f"Drag only\n(k={drag_rep['k']:.4f})",
    f"Friction + Drag\n(mu=0.10, k={comb_rep['k']:.4f})",
]
summary_times = [
    T_cycloid_analytic,
    fric_rep["time"],
    drag_rep["time"],
    comb_rep["time"],
]

fig, ax = plt.subplots(figsize=(8.0, 5.0))
x_pos = np.arange(len(summary_labels))
ax.plot(x_pos, summary_times, "o-", lw=2.3, color="tab:blue")
ax.set_xticks(x_pos)
ax.set_xticklabels(summary_labels)
ax.set_ylabel("Travel time (s)")
ax.set_title("Travel-Time Summary Across Four Modelling Cases")
for i, t in enumerate(summary_times):
    ax.text(i, t + 0.02 * max(summary_times), f"{t:.3f}", ha="center", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("figures/09_time_summary_four_cases.png", dpi=170)

# Also show combined-case sensitivities over selected parameter pairs
fig, ax = plt.subplots(figsize=(7.8, 4.8))
pair_labels = [f"$\\mu={mu_c:.2f}$\n$k={combined_results[(mu_c, area_c)]['k']:.4f}$" for mu_c, area_c in combined_drag_pairs]
pair_times = [combined_results[(mu_c, area_c)]["time"] for mu_c, area_c in combined_drag_pairs]
x_pos = np.arange(len(pair_labels))
ax.plot(x_pos, pair_times, "o-", lw=2.2, color="tab:gray")
ax.set_xticks(x_pos)
ax.set_xticklabels(pair_labels)
ax.set_title("Combined Model: Optimized Time for Representative $(\\mu, k)$ Pairs")
ax.set_ylabel("Travel time (s)")
for i, t in enumerate(pair_times):
    ax.text(i, t + 0.015 * max(pair_times), f"{t:.3f}", ha="center", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("figures/10_combined_pair_times.png", dpi=170)

# %%
# Representative time-resolved simulation on the optimized combined-case path
motion_example = simulate_motion_discrete_segments(
    comb_rep["x"],
    comb_rep["y"],
    mu=comb_rep["mu"],
    rho_air=comb_rep["rho"],
    drag_coefficient=comb_rep["C_d"],
    area=comb_rep["A"],
    mass=m,
    grav=g,
    n_time_samples=900,
)

print("Motion reconstruction status:")
print(f"valid={motion_example['valid']}, total_time={motion_example['total_time']:.6f} s")
print(f"time-match error vs segment sum = {motion_example['time_match_error']:.3e} s")
if not motion_example["valid"]:
    print(f"Warning: {motion_example['message']}")

fig, ax = plot_motion_snapshots(
    comb_rep["x"],
    comb_rep["y"],
    motion_example,
    num_snapshots=9,
    title="Combined-Case Motion Snapshots on the Optimized Path",
)
fig.savefig("figures/11_motion_snapshots_combined.png", dpi=170)

fig, axes = plot_x_y_v_vs_time(
    motion_example,
    include_acceleration=True,
    title_prefix="Combined-Case Motion: $x(t)$, $y(t)$, $v(t)$, $a(t)$",
)
fig.savefig("figures/12_motion_time_series_combined.png", dpi=170)

# %%
# Energy and dissipation analysis for representative motions
energy_cases = {
    "Frictionless": compute_energy_budget_discrete_segments(
        x_cyc,
        y_cyc,
        mu=0.0,
        rho_air=0.0,
        drag_coefficient=C_d,
        area=0.0,
        mass=m,
        grav=g,
    ),
    "Friction only": compute_energy_budget_discrete_segments(
        fric_rep["x"],
        fric_rep["y"],
        mu=fric_rep["mu"],
        rho_air=0.0,
        drag_coefficient=fric_rep["C_d"],
        area=0.0,
        mass=m,
        grav=g,
    ),
    "Drag only": compute_energy_budget_discrete_segments(
        drag_rep["x"],
        drag_rep["y"],
        mu=0.0,
        rho_air=drag_rep["rho"],
        drag_coefficient=drag_rep["C_d"],
        area=drag_rep["A"],
        mass=m,
        grav=g,
    ),
    "Friction + Drag": compute_energy_budget_discrete_segments(
        comb_rep["x"],
        comb_rep["y"],
        mu=comb_rep["mu"],
        rho_air=comb_rep["rho"],
        drag_coefficient=comb_rep["C_d"],
        area=comb_rep["A"],
        mass=m,
        grav=g,
    ),
}

print("Energy budget summary for representative motions:")
for label, energy in energy_cases.items():
    print(
        f"{label}: gravity={energy['total_gravity_work']:.3f} J, "
        f"friction loss={energy['total_friction_loss']:.3f} J, "
        f"drag loss={energy['total_drag_loss']:.3f} J, "
        f"final kinetic={energy['final_kinetic_energy']:.3f} J"
    )

fig, ax = plot_dissipation_comparison(
    list(energy_cases.keys()),
    energy_cases,
    title="Cumulative Energy Dissipation vs Position",
)
fig.savefig("figures/13_energy_dissipation_comparison.png", dpi=170)

fig, axes = plot_energy_budget_vs_time(
    energy_cases["Friction + Drag"],
    title="Combined-Case Energy Budget Along the Motion",
)
fig.savefig("figures/14_energy_budget_combined.png", dpi=170)
