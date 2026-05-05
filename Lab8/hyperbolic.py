import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_MAX = 1.0
T_MAX = 0.60
c = 1.0

Nx = 200
CFL = 0.75

lambda_fedorenko = 1.0

tvd_limiter_name = "minmod"

Nx_values_for_convergence = [50, 100, 200, 400]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICTURES_DIR = os.path.join(SCRIPT_DIR, "pictures")

SCHEME_FILE_CODES = {
    "CIR": "a",
    "MacCormack": "b",
    "Fedorenko": "c",
    "TVD": "d",
}


def initial_profile(x):
    x = np.asarray(x)
    left = 0.25 * X_MAX
    right = 0.55 * X_MAX
    height = 1.0
    return height * ((x >= left) & (x <= right)).astype(float)


def left_boundary_profile(t):
    t = np.asarray(t)
    center = 0.38 * T_MAX
    half_width = 0.18 * T_MAX
    height = 1.0
    return height * np.maximum(1.0 - np.abs(t - center) / half_width, 0.0)


def exact_solution(x, t):
    x = np.asarray(x)
    foot = x - c * t
    u = np.empty_like(x, dtype=float)

    from_initial = foot >= 0.0
    u[from_initial] = initial_profile(foot[from_initial])
    u[~from_initial] = left_boundary_profile(t - x[~from_initial] / c)
    return u


def limiter(r, name="minmod"):
    if name == "minmod":
        return np.maximum(0.0, np.minimum(1.0, r))

    if name == "vanleer":
        return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-14)

    if name == "superbee":
        return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))

    if name == "mc":
        return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, 0.5 * (1.0 + r)), 2.0))

    raise ValueError(f"Неизвестный TVD-ограничитель: {name}")


def make_grid(nx=Nx, cfl=CFL):
    h = X_MAX / nx
    tau0 = cfl * h / abs(c)
    nt = int(np.ceil(T_MAX / tau0))
    tau = T_MAX / nt
    sigma = c * tau / h

    if abs(sigma) > 1.0 + 1e-12:
        raise RuntimeError("Условие устойчивости |c| * tau / h <= 1 нарушено.")

    x = np.linspace(0.0, X_MAX, nx + 1)
    t = np.linspace(0.0, T_MAX, nt + 1)
    return x, t, h, tau, sigma


def apply_boundary(u_next, t_next):
    u_next[0] = left_boundary_profile(t_next)
    u_next[-1] = u_next[-2]
    return u_next


def initial_layer(x):
    u0 = initial_profile(x).astype(float)
    u0[0] = left_boundary_profile(0.0)
    return u0


def right_ghost_outflow(u):
    return u[-1]


def step_cir(u, sigma, t_next):
    u_next = u.copy()
    u_next[1:] = u[1:] - sigma * (u[1:] - u[:-1])
    apply_boundary(u_next, t_next)
    return u_next


def step_maccormack(u, sigma, t_now, t_next):
    u_pred = u.copy()
    u_pred[:-1] = u[:-1] - sigma * (u[1:] - u[:-1])
    u_pred[-1] = u[-1] - sigma * (u[-1] - u[-2])
    u_pred[0] = left_boundary_profile(t_next)

    u_next = u.copy()
    u_next[1:] = 0.5 * (u[1:] + u_pred[1:] - sigma * (u_pred[1:] - u_pred[:-1]))
    apply_boundary(u_next, t_next)
    return u_next


def step_fedorenko(u, sigma, t_next, lam=lambda_fedorenko):
    u_next = u.copy()

    left = u[:-1]
    center = u[1:]
    right = np.empty_like(center)
    right[:-1] = u[2:]
    right[-1] = right_ghost_outflow(u)

    delta2 = left - 2.0 * center + right
    grad_left = center - left

    gamma = (np.abs(delta2) <= lam * np.abs(grad_left) + 1e-14).astype(float)

    u_next[1:] = (
        center
        - sigma * (center - left)
        - 0.5 * gamma * sigma * (1.0 - sigma) * delta2
    )
    apply_boundary(u_next, t_next)
    return u_next


def step_tvd(u, sigma, t_next, limiter_name=tvd_limiter_name):
    u_next = u.copy()

    ue = np.empty(len(u) + 2, dtype=float)
    ue[1:-1] = u
    ue[0] = u[0]
    ue[-1] = right_ghost_outflow(u)

    left_state = ue[1:-1]    
    right_state = ue[2:]    
    delta_right = right_state - left_state
    delta_left = left_state - ue[:-2]

    r = np.zeros_like(delta_right)
    mask = np.abs(delta_right) > 1e-14
    r[mask] = delta_left[mask] / delta_right[mask]
    phi = limiter(r, limiter_name)

    flux_over_c = left_state + 0.5 * (1.0 - sigma) * phi * delta_right

    u_next[1:] = u[1:] - sigma * (flux_over_c[1:] - flux_over_c[:-1])
    apply_boundary(u_next, t_next)
    return u_next


def solve(scheme_name, nx=Nx, cfl=CFL, store_history=True):
    x, t, h, tau, sigma = make_grid(nx, cfl)

    u = initial_layer(x)

    if store_history:
        U = np.zeros((len(t), len(x)), dtype=float)
        U[0, :] = u
    else:
        U = None

    for n in range(len(t) - 1):
        t_now = t[n]
        t_next = t[n + 1]

        if scheme_name == "CIR":
            u = step_cir(u, sigma, t_next)
        elif scheme_name == "MacCormack":
            u = step_maccormack(u, sigma, t_now, t_next)
        elif scheme_name == "Fedorenko":
            u = step_fedorenko(u, sigma, t_next, lambda_fedorenko)
        elif scheme_name == "TVD":
            u = step_tvd(u, sigma, t_next, tvd_limiter_name)
        else:
            raise ValueError(f"Неизвестная схема: {scheme_name}")

        if store_history:
            U[n + 1, :] = u

    info = {
        "x": x,
        "t": t,
        "h": h,
        "tau": tau,
        "sigma": sigma,
        "nx": nx,
        "nt": len(t) - 1,
    }
    return U, u, info


def total_variation(u):
    return np.sum(np.abs(np.diff(u)))


def error_metrics(x, u_num, t_value=T_MAX):
    u_ex = exact_solution(x, t_value)
    h = x[1] - x[0]
    err = u_num - u_ex
    return {
        "L1": h * np.sum(np.abs(err)),
        "L2": np.sqrt(h * np.sum(err**2)),
        "Linf": np.max(np.abs(err)),
        "TV": total_variation(u_num),
        "min": np.min(u_num),
        "max": np.max(u_num),
    }


def convergence_study(nx_values=Nx_values_for_convergence, cfl=CFL):
    scheme_names = ["CIR", "MacCormack", "Fedorenko", "TVD"]
    results = {name: [] for name in scheme_names}

    for nx in nx_values:
        for name in scheme_names:
            _, u_final, info = solve(name, nx=nx, cfl=cfl, store_history=False)
            metrics = error_metrics(info["x"], u_final, T_MAX)
            results[name].append({"h": info["h"], **metrics})

    return results


def save_figure(fig, filename, dpi=200):
    os.makedirs(PICTURES_DIR, exist_ok=True)
    path = os.path.join(PICTURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_final_profiles(solutions, x, filename="hyp_compare.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, exact_solution(x, T_MAX), "k--", linewidth=2.0, label="точное")

    for name, U in solutions.items():
        ax.plot(x, U[-1, :], linewidth=1.8, label=name)

    ax.set_xlabel("x")
    ax.set_ylabel("u(T, x)")
    ax.set_title(f"Сравнение численных решений на финальном слое, T={T_MAX}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, filename)


def plot_solution_xt(U, x, t, title, filename):
    X, TT = np.meshgrid(x, t)

    fig, ax = plt.subplots(figsize=(8, 5))
    cs = ax.contourf(X, TT, U, levels=60)
    fig.colorbar(cs, ax=ax, label="u(t, x)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_solution_3d(U, x, t, title, filename):
    X, TT = np.meshgrid(x, t)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, TT, U, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(t, x)")
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.6, aspect=12, label="u(t, x)")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_convergence(results, filename="hyp_convergence.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, rows in results.items():
        h = np.array([row["h"] for row in rows])
        l1 = np.array([row["L1"] for row in rows])
        ax.loglog(h, l1, marker="o", linewidth=1.8, label=name)

    ax.invert_xaxis()
    ax.set_xlabel("h")
    ax.set_title("Сходимость при h → 0")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, filename)

def main():
    x, t, h, tau, sigma = make_grid(Nx, CFL)

    scheme_names = ["CIR", "MacCormack", "Fedorenko", "TVD"]
    solutions = {}

    for name in scheme_names:
        U, _, _ = solve(name, nx=Nx, cfl=CFL, store_history=True)
        solutions[name] = U

    plot_final_profiles(solutions, x, "hyp_compare.png")

    for name, U in solutions.items():
        code = SCHEME_FILE_CODES[name]
        plot_solution_xt(
            U,
            x,
            t,
            f"{name}: поле u(t, x)",
            f"hyp_{code}.png",
        )
        plot_solution_3d(
            U,
            x,
            t,
            f"{name}: поверхность u(t, x)",
            f"hyp_{code}_3d.png",
        )

    results = convergence_study(Nx_values_for_convergence, CFL)
    plot_convergence(results, "hyp_convergence.png")


if __name__ == "__main__":
    main()