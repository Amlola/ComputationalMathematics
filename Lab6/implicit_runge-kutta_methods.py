import numpy as np
import matplotlib.pyplot as plt

from system_generator import f, jacobian

gamma = 1.0 - 1.0 / np.sqrt(2.0)

A = np.array([
    [gamma, 0.0],
    [1.0 - gamma, gamma]
], dtype=float)

b_rk = np.array([1.0 - gamma, gamma], dtype=float)
c_rk = np.array([gamma, 1.0], dtype=float)

def norm2(v):
    return np.sqrt(np.sum(v * v))

def solve_lin(A, b, eps=1e-14):
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]

    det = a11 * a22 - a12 * a21
    if abs(det) < eps:
        raise RuntimeError("Error in solve lin")

    x0 = (b[0] * a22 - a12 * b[1]) / det
    x1 = (a11 * b[1] - b[0] * a21) / det
    return np.array([x0, x1], dtype=float)


def solve_newton(F, JF, x0, tol=1e-10, maxiter=20):
    x = x0.copy()

    for i in range(1, maxiter + 1):
        Fx = F(x)
        if norm2(Fx) < tol:
            return x, i, True

        Jx = JF(x)
        delta = solve_lin(Jx, -Fx)

        x_new = x + delta

        if norm2(x_new - x) < tol:
            return x_new, i, True

        x = x_new

    return x, maxiter, False

def runge_step(tn, zn, h):
    def G1(Y1):
        return Y1 - zn - h * A[0, 0] * f(tn + c_rk[0] * h, Y1)

    def JG1(Y1):
        return np.eye(len(zn)) - h * A[0, 0] * jacobian(tn + c_rk[0] * h, Y1)

    Y1_init = zn.copy()
    Y1, it1, ok1 = solve_newton(G1, JG1, Y1_init)
    if not ok1:
        raise RuntimeError(
            f"Ньютон не сошелся на 1-й стадии при t={tn}, итераций: {it1}"
        )

    K1 = f(tn + c_rk[0] * h, Y1)

    def G2(Y2):
        return Y2 - zn - h * A[1, 0] * K1 - h * A[1, 1] * f(tn + c_rk[1] * h, Y2)

    def JG2(Y2):
        return np.eye(len(zn)) - h * A[1, 1] * jacobian(tn + c_rk[1] * h, Y2)

    Y2_init = Y1.copy()
    Y2, it2, ok2 = solve_newton(G2, JG2, Y2_init)
    if not ok2:
        raise RuntimeError(
            f"Ньютон не сошелся на 2-й стадии при t={tn}, итераций: {it2}"
        )

    K2 = f(tn + c_rk[1] * h, Y2)

    z_next = zn + h * (b_rk[0] * K1 + b_rk[1] * K2)
    return z_next


def solve_runge(t0, t_end, z0, h):
    n_steps = int(np.ceil((t_end - t0) / h))
    t = np.zeros(n_steps + 1)
    z = np.zeros((n_steps + 1, len(z0)))

    t[0] = t0
    z[0] = z0

    tn = t0
    zn = np.array(z0, dtype=float)

    for n in range(n_steps):
        h_step = min(h, t_end - tn)
        zn = runge_step(tn, zn, h_step)
        tn = tn + h_step

        t[n + 1] = tn
        z[n + 1] = zn

    return t, z


if __name__ == "__main__":
    t0 = 0.0
    t_end = 0.5
    z0 = np.array([2.0, 0.0], dtype=float)

    h = 1e-2

    t, z = solve_runge(t0, t_end, z0, h)

    x = z[:, 0]
    y = z[:, 1]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, label="x(t)")
    plt.plot(t, y, label="y(t)")
    plt.xlabel("t")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.title("Фазовый портрет")

    plt.tight_layout()
    plt.show()