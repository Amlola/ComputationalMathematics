import argparse
import os

import matplotlib.pyplot as plt

from generate_system import (
    nonlinear_rhs,
    nonlinear_rhs_dy,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICTURES_DIR = os.path.join(BASE_DIR, "pictures")


def thomas_algorithm(a, b, c, d):
    n = len(d)
    cp = [0.0] * n
    dp = [0.0] * n

    if abs(b[0]) < 1e-15:
        raise RuntimeError("Нулевой ведущий элемент в методе прогонки.")

    cp[0] = c[0] / b[0] if n > 1 else 0.0
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            raise RuntimeError("Метод прогонки неустойчив: слишком мал знаменатель.")
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    y = [0.0] * n
    y[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        y[i] = dp[i] - cp[i] * y[i + 1]

    return y


def solve_quasilinearization(a=0.0, b=1.0, n=200, tol=1e-8, max_iter=100, eps=1e-10):
    h = (b - a) / n
    xs = [a + i * h for i in range(n + 1)]

    y_prev = [2.0 * x for x in xs]
    y_prev[0] = 0.0
    y_prev[-1] = 2.0

    history = []
    curves = [(xs[:], y_prev[:], 0.0)]

    for k in range(1, max_iter + 1):
        m = n - 1
        lower = [0.0] * m
        diag = [0.0] * m
        upper = [0.0] * m
        rhs = [0.0] * m

        for j in range(m):
            i = j + 1
            x_i = xs[i]
            yk = y_prev[i]

            alpha = nonlinear_rhs_dy(x_i, yk, eps=eps)
            beta = nonlinear_rhs(x_i, yk, eps=eps) - alpha * yk

            lower[j] = 1.0
            diag[j] = -2.0 - (h * h) * alpha
            upper[j] = 1.0
            rhs[j] = (h * h) * beta

        rhs[0] -= 0.0
        rhs[-1] -= 2.0

        lower[0] = 0.0
        upper[-1] = 0.0

        y_inner = thomas_algorithm(lower, diag, upper, rhs)

        y_new = [0.0] * (n + 1)
        y_new[0] = 0.0
        y_new[-1] = 2.0
        for i in range(1, n):
            y_new[i] = y_inner[i - 1]

        max_diff = max(abs(y_new[i] - y_prev[i]) for i in range(n + 1))
        history.append(max_diff)
        curves.append((xs[:], y_new[:], max_diff))

        print(f"iter = {k:2d}, max|y^(k) - y^(k-1)| = {max_diff:.12e}")

        if max_diff < tol:
            return xs, y_new, history, curves, k

        y_prev = y_new

    raise RuntimeError("Метод квазилинеаризации не сошелся за заданное число итераций.")


def ensure_pictures_dir(pictures_dir):
    os.makedirs(pictures_dir, exist_ok=True)


def plot_results(xs, y_star, history, curves, iterations_count, pictures_dir=PICTURES_DIR):
    ensure_pictures_dir(pictures_dir)

    plt.figure(figsize=(8, 5))
    for i, (xv, yv, err) in enumerate(curves):
        label = "начальное приближение" if i == 0 else f"iter {i}: err={err:.2e}"
        plt.plot(xv, yv, label=label)
    plt.scatter([0, 1], [0, 2], zorder=3, label="Граничные условия")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Метод квазилинеаризации: интегральные кривые")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, "04_quasi_iterations_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(xs, y_star, linewidth=2, label="Итоговое решение")
    plt.scatter([0, 1], [0, 2], zorder=3, label="Граничные условия")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Метод квазилинеаризации: финальное решение")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, "05_quasi_final_solution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Номер итерации")
    plt.ylabel("max |y^(k) - y^(k-1)|")
    plt.title(
        "Метод квазилинеаризации: сходимость\n"
        f"Количество итераций решения линейной краевой задачи: {iterations_count}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, "06_quasi_convergence.png"), dpi=200, bbox_inches="tight")
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Метод квазилинеаризации для краевой задачи.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--pictures-dir", default=PICTURES_DIR)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    xs, y_star, history, curves, iterations_count = solve_quasilinearization(
        n=args.n,
        tol=args.tol,
        max_iter=args.max_iter,
    )

    print("\nМетод квазилинеаризации сошелся.")
    print(f"Количество итераций решения линейной краевой задачи: {iterations_count}")

    plot_results(xs, y_star, history, curves, iterations_count, args.pictures_dir)