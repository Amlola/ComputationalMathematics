import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from system_generator import build_lienard_system


def rk_step(system_function, t, u, h, order=2):
    if order == 1:
        k1 = system_function(t, u)
        u_next = u + h * k1
    elif order == 2:
        k1 = system_function(t, u)
        k2 = system_function(t + h / 2.0, u + (h / 2.0) * k1)
        u_next = u + h * k2
    elif order == 3:
        k1 = system_function(t, u)
        k2 = system_function(t + h / 2.0, u + (h / 2.0) * k1)
        k3 = system_function(t + h, u - h * k1 + 2.0 * h * k2)
        u_next = u + h * (k1 + 4.0 * k2 + k3) / 6.0
    elif order == 4:
        k1 = system_function(t, u)
        k2 = system_function(t + h / 2.0, u + (h / 2.0) * k1)
        k3 = system_function(t + h / 2.0, u + (h / 2.0) * k2)
        k4 = system_function(t + h, u + h * k3)
        u_next = u + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    else:
        raise ValueError("order must be 1, 2, 3, or 4")
    return u_next


def rk_solve(system_function, t_span, u0, h, order=2):
    t0, t1 = t_span
    n_steps = int(np.ceil((t1 - t0) / h))

    t_values = np.zeros(n_steps + 1, dtype=float)
    u_values = np.zeros((n_steps + 1, len(u0)), dtype=float)

    t_values[0] = t0
    u_values[0] = np.array(u0, dtype=float)

    t = t0
    u = np.array(u0, dtype=float)

    for n in range(n_steps):
        h_step = min(h, t1 - t)
        u = rk_step(system_function, t, u, h_step, order=order)
        t = t + h_step
        t_values[n + 1] = t
        u_values[n + 1] = u

    return t_values, u_values


def ensure_pictures_folder():
    os.makedirs("pictures", exist_ok=True)


def build_figure(system_function, t_span, u0, h, order=2):
    t_values, u_values = rk_solve(system_function, t_span, u0, h=h, order=order)

    x_values = u_values[:, 0]
    y_values = u_values[:, 1]

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(t_values, x_values, label="x(t)", linewidth=2)
    axes[0].plot(t_values, y_values, label="y(t)", linewidth=2)
    axes[0].set_xlabel("t")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x_values, y_values, linewidth=2)
    axes[1].set_title("Фазовый портрет")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(True)

    plt.tight_layout()
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("order", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("filename", type=str)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    order = args.order
    filename = args.filename
    h = args.step

    if not filename.lower().endswith(".png"):
        filename += ".png"

    u0 = np.array([2.0, 0.0], dtype=float)
    t_span = (0.0, 20.0)

    system_function = build_lienard_system(eps=0.8)

    figure = build_figure(
        system_function=system_function,
        t_span=t_span,
        u0=u0,
        h=h,
        order=order
    )

    ensure_pictures_folder()
    path = os.path.join("pictures", filename)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

    plt.show()