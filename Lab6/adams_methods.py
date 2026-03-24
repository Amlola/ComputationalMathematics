import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from system_generator import build_lienard_system


AB_COEFFICIENTS = {
    1: np.array([1.0], dtype=float),
    2: np.array([3.0 / 2.0, -1.0 / 2.0], dtype=float),
    3: np.array([23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0], dtype=float),
    4: np.array([55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0], dtype=float),
}


def rk4_step(system_function, t, u, h):
    k1 = system_function(t, u)
    k2 = system_function(t + h / 2.0, u + (h / 2.0) * k1)
    k3 = system_function(t + h / 2.0, u + (h / 2.0) * k2)
    k4 = system_function(t + h, u + h * k3)
    return u + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def build_time_grid(t_span, h):
    t0, t1 = t_span
    if h <= 0:
        raise ValueError("step must be positive")

    n_steps = int(round((t1 - t0) / h))
    if n_steps <= 0:
        raise ValueError("time interval is too short for the selected step")

    if not np.isclose(t0 + n_steps * h, t1, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "For Adams-Bashforth methods the step must evenly divide the interval length. "
            "Choose h such that (t1 - t0) / h is an integer."
        )

    return np.linspace(t0, t1, n_steps + 1, dtype=float)


def adams_bashforth_solve(system_function, t_span, u0, h, order=4):
    if order not in AB_COEFFICIENTS:
        raise ValueError("order must be 1, 2, 3, or 4")

    t_values = build_time_grid(t_span, h)
    n_steps = len(t_values) - 1
    u_values = np.zeros((n_steps + 1, len(u0)), dtype=float)
    u_values[0] = np.array(u0, dtype=float)

    startup_steps = min(order - 1, n_steps)
    for n in range(startup_steps):
        u_values[n + 1] = rk4_step(system_function, t_values[n], u_values[n], h)

    coefficients = AB_COEFFICIENTS[order]
    for n in range(order - 1, n_steps):
        increment = np.zeros_like(u_values[n])
        for j, coefficient in enumerate(coefficients):
            increment += coefficient * system_function(t_values[n - j], u_values[n - j])
        u_values[n + 1] = u_values[n] + h * increment

    return t_values, u_values


def ensure_pictures_folder():
    os.makedirs("pictures", exist_ok=True)


def style_solution_axes(axes, title_prefix):
    axes[0].set_title(f"{title_prefix}: y(t)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("y")
    axes[0].grid(True)

    axes[1].set_title(f"{title_prefix}: z(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("z")
    axes[1].grid(True)

    axes[2].set_title(f"{title_prefix}: фазовая траектория")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")
    axes[2].grid(True)



def build_figure(system_function, t_span, u0, h, order=4, eps=1.0):
    t_values, u_values = adams_bashforth_solve(
        system_function=system_function,
        t_span=t_span,
        u0=u0,
        h=h,
        order=order,
    )

    y_values = u_values[:, 0]
    z_values = u_values[:, 1]

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    title_prefix = f"AB{order}, eps={eps:g}, h={h:g}"

    axes[0].plot(t_values, y_values, linewidth=2, label="y(t)")
    axes[1].plot(t_values, z_values, linewidth=2, label="z(t)")
    axes[2].plot(y_values, z_values, linewidth=2)

    style_solution_axes(axes, title_prefix)
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    return figure



def build_comparison_figure(system_function, t_span, u0, steps, order=4, eps=1.0):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    for h in steps:
        t_values, u_values = adams_bashforth_solve(
            system_function=system_function,
            t_span=t_span,
            u0=u0,
            h=h,
            order=order,
        )
        label = f"h={h:g}"
        axes[0].plot(t_values, u_values[:, 0], linewidth=2, label=label)
        axes[1].plot(t_values, u_values[:, 1], linewidth=2, label=label)
        axes[2].plot(u_values[:, 0], u_values[:, 1], linewidth=2, label=label)

    title_prefix = f"AB{order}, eps={eps:g}"
    style_solution_axes(axes, title_prefix)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.tight_layout()
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explicit Adams-Bashforth methods of order 1-4 for the Van der Pol equation in Lienard form"
    )
    parser.add_argument("order", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("filename", type=str)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--compare-steps", nargs="*", type=float, default=None)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    filename = args.filename
    if not filename.lower().endswith(".png"):
        filename += ".png"

    u0 = np.array([2.0, 0.0], dtype=float)
    t_span = (0.0, args.t_end)
    system_function = build_lienard_system(eps=args.eps)

    if args.compare_steps:
        figure = build_comparison_figure(
            system_function=system_function,
            t_span=t_span,
            u0=u0,
            steps=args.compare_steps,
            order=args.order,
            eps=args.eps,
        )
    else:
        figure = build_figure(
            system_function=system_function,
            t_span=t_span,
            u0=u0,
            h=args.step,
            order=args.order,
            eps=args.eps,
        )

    ensure_pictures_folder()
    path = os.path.join("pictures", filename)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)