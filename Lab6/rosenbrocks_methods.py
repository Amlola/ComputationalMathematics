import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from Lab2.lu import solve_lu
from system_generator import f, jacobian


def solve_linear_system(A, b):
    return np.array(solve_lu(A.copy(), b.copy()), dtype=float)

def rosenbrock_step(system_function, jacobian_function, t, z, h):
    gamma_diag = 1.0 / 4.0

    gamma_matrix = np.array([
        [1.0 / 4.0,    0.0,           0.0,         0.0],
        [-6.0 / 25.0,  1.0 / 4.0,     0.0,         0.0],
        [2.0 / 3.0,    0.0,           1.0 / 4.0,   0.0],
        [-5.0 / 27.0, -25.0 / 108.0, -1.0 / 4.0,   1.0 / 4.0]
    ], dtype=float)

    alpha_matrix = np.array([
        [0.0,          0.0,         0.0, 0.0],
        [3.0 / 5.0,    0.0,         0.0, 0.0],
        [29.0 / 54.0, 25.0 / 54.0,  0.0, 0.0],
        [2.0 / 27.0,  25.0 / 27.0,  0.0, 0.0]
    ], dtype=float)

    sigma = np.array([
        8.0 / 27.0,
        125.0 / 216.0,
        0.0,
        1.0 / 8.0
    ], dtype=float)

    alpha_time = np.array([
        0.0,
        0.6,
        1.0,
        1.0
    ], dtype=float)

    J = jacobian_function(t, z)
    n = len(z)

    A = np.eye(n) - h * gamma_diag * J

    k = []

    rhs1 = h * system_function(t, z)
    k1 = solve_linear_system(A, rhs1)
    k.append(k1)

    z2 = z + alpha_matrix[1, 0] * k[0]
    rhs2 = h * system_function(t + alpha_time[1] * h, z2) + h * J @ (gamma_matrix[1, 0] * k[0])
    k2 = solve_linear_system(A, rhs2)
    k.append(k2)

    z3 = z + alpha_matrix[2, 0] * k[0] + alpha_matrix[2, 1] * k[1]
    rhs3 = h * system_function(t + alpha_time[2] * h, z3) + h * J @ (gamma_matrix[2, 0] * k[0] + gamma_matrix[2, 1] * k[1])
    k3 = solve_linear_system(A, rhs3)
    k.append(k3)

    z4 = z + alpha_matrix[3, 0] * k[0] + alpha_matrix[3, 1] * k[1] + alpha_matrix[3, 2] * k[2]
    rhs4 = h * system_function(t + alpha_time[3] * h, z4) + h * J @ (
        gamma_matrix[3, 0] * k[0] + gamma_matrix[3, 1] * k[1] + gamma_matrix[3, 2] * k[2]
    )
    k4 = solve_linear_system(A, rhs4)
    k.append(k4)

    z_next = z + sigma[0] * k[0] + sigma[1] * k[1] + sigma[2] * k[2] + sigma[3] * k[3]

    return z_next


def rosenbrock_solve(system_function, jacobian_function, t_span, z0, h):
    t0, t1 = t_span
    n_steps = int(np.ceil((t1 - t0) / h))

    t_values = np.zeros(n_steps + 1, dtype=float)
    z_values = np.zeros((n_steps + 1, len(z0)), dtype=float)

    t_values[0] = t0
    z_values[0] = np.array(z0, dtype=float)

    t = t0
    z = np.array(z0, dtype=float)

    for i in range(n_steps):
        h_step = min(h, t1 - t)
        z = rosenbrock_step(system_function, jacobian_function, t, z, h_step)
        t = t + h_step
        t_values[i + 1] = t
        z_values[i + 1] = z

    return t_values, z_values


def ensure_pictures_folder():
    os.makedirs("pictures", exist_ok=True)


def build_figure(system_function, jacobian_function, t_span, z0, h):
    t_values, z_values = rosenbrock_solve(system_function, jacobian_function, t_span, z0, h)

    x_values = z_values[:, 0]
    y_values = z_values[:, 1]

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
    parser.add_argument("filename", type=str)
    parser.add_argument("--step", type=float, default=1e-3)
    parser.add_argument("--t1", type=float, default=20.0)
    args = parser.parse_args()

    filename = args.filename
    h = args.step
    t1 = args.t1

    if not filename.lower().endswith(".png"):
        filename += ".png"

    z0 = np.array([2.0, 0.0], dtype=float)
    t_span = (0.0, t1)

    figure = build_figure(
        system_function=f,
        jacobian_function=jacobian,
        t_span=t_span,
        z0=z0,
        h=h
    )

    ensure_pictures_folder()
    path = os.path.join("pictures", filename)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

    plt.show()