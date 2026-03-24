import numpy as np
import matplotlib.pyplot as plt

from system_generator import f, jacobian

def norm2(v):
    return np.sqrt(np.sum(v * v))

def solve_lin(A, b, eps=1e-14):
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]

    det = a11 * a22 - a12 * a21
    if abs(det) < eps:
        raise RuntimeError("Сингулярная матрица в solve_lin")

    x1 = (b[0] * a22 - a12 * b[1]) / det
    x2 = (a11 * b[1] - b[0] * a21) / det

    return np.array([x1, x2], dtype=float)

def solve_newton(F, JF, x0, tol=1e-10, maxiter=100):
    x = x0.copy()

    for _ in range(maxiter):
        Fx = F(x)
        if norm2(Fx) < tol:
            return x, True

        Jx = JF(x)
        delta = solve_lin(Jx, -Fx)
        x_new = x + delta

        if norm2(x_new - x) < tol:
            return x_new, True

        x = x_new

    return x, False

def backward_euler_step(tn, zn, h):
    t_next = tn + h

    def G(z_next):
        return z_next - zn - h * f(t_next, z_next)

    def JG(z_next):
        return np.eye(len(zn)) - h * jacobian(t_next, z_next)

    z_init = zn + h * f(tn, zn)
    z_next, ok = solve_newton(G, JG, z_init)

    if not ok:
        raise RuntimeError(f"Ньютон не сошелся в BDF1 при t={tn}")

    return z_next

BDF_COEFFS = {
    1: np.array([1.0, -1.0], dtype=float),
    2: np.array([3.0 / 2.0, -2.0, 1.0 / 2.0], dtype=float),
    3: np.array([11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0], dtype=float),
    4: np.array([25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0], dtype=float),
}

def bdf_step(t_hist, z_hist, h, order):
    if order not in BDF_COEFFS:
        raise ValueError("Поддерживаются только порядки 1, 2, 3, 4")

    alpha = BDF_COEFFS[order]
    k = order

    if len(z_hist) < k:
        raise ValueError(f"Недостаточно истории для BDF{order}")

    tn = t_hist[-1]
    t_next = tn + h
    dim = len(z_hist[-1])

    known = np.zeros(dim, dtype=float)
    for j in range(1, k + 1):
        known += alpha[j] * z_hist[-j]

    alpha0 = alpha[0]

    def G(z_next):
        return alpha0 * z_next + known - h * f(t_next, z_next)

    def JG(z_next):
        return alpha0 * np.eye(dim) - h * jacobian(t_next, z_next)

    if order == 1:
        z_init = z_hist[-1] + h * f(tn, z_hist[-1])
    else:
        z_init = z_hist[-1] + (z_hist[-1] - z_hist[-2])

    z_next, ok = solve_newton(G, JG, z_init)

    if not ok:
        raise RuntimeError(f"Ньютон не сошелся в BDF{order} при t={tn}")

    return z_next


def solve_gear(t0, t_end, z0, h, order):
    if order not in (1, 2, 3, 4):
        raise ValueError("order должен быть 1, 2, 3 или 4")

    n_steps = int(np.ceil((t_end - t0) / h))
    dim = len(z0)

    t = np.zeros(n_steps + 1, dtype=float)
    z = np.zeros((n_steps + 1, dim), dtype=float)

    t[0] = t0
    z[0] = np.array(z0, dtype=float)

    start_steps = min(order - 1, n_steps)

    for n in range(start_steps):
        h_step = min(h, t_end - t[n])
        z[n + 1] = backward_euler_step(t[n], z[n], h_step)
        t[n + 1] = t[n] + h_step

    for n in range(start_steps, n_steps):
        h_step = min(h, t_end - t[n])

        if abs(h_step - h) > 1e-15:
            z[n + 1] = backward_euler_step(t[n], z[n], h_step)
            t[n + 1] = t[n] + h_step
            break

        t_hist = t[:n + 1]
        z_hist = z[:n + 1]

        z[n + 1] = bdf_step(t_hist, z_hist, h, order)
        t[n + 1] = t[n] + h

    return t, z


if __name__ == "__main__":
    t0 = 0.0
    t_end = 20.0
    z0 = np.array([2.0, 0.0], dtype=float)
    h = 1e-3

    order = 4

    t, z = solve_gear(t0, t_end, z0, h, order)

    x = z[:, 0]
    y = z[:, 1]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, label="x(t)")
    plt.plot(t, y, label="y(t)")
    plt.xlabel("t")
    plt.ylabel("solution")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"gear_bdf{order}.png", dpi=150, bbox_inches="tight")

    plt.show()