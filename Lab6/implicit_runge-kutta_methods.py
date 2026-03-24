import numpy as np
import matplotlib.pyplot as plt

from system_generator import f, jacobian

def norm2(v):
    return np.sqrt(np.sum(v * v))

def solve_lin(A, b, eps=1e-14):
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    n = len(b)

    for k in range(n):
        pivot_row = k
        pivot_val = abs(A[k, k])

        for i in range(k + 1, n):
            if abs(A[i, k]) > pivot_val:
                pivot_val = abs(A[i, k])
                pivot_row = i

        if pivot_val < eps:
            raise RuntimeError("Сингулярная матрица в solve_lin")

        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        pivot = A[k, k]

        for i in range(k + 1, n):
            factor = A[i, k] / pivot
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        s = np.dot(A[i, i + 1:], x[i + 1:])
        if abs(A[i, i]) < eps:
            raise RuntimeError("Сингулярная матрица в solve_lin")
        x[i] = (b[i] - s) / A[i, i]

    return x

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

gamma = 1.0 - 1.0 / np.sqrt(2.0)

RK_METHODS = {
    "implicit_runge_1method": {
        "A": np.array([[0.5]], dtype=float),
        "b": np.array([1.0], dtype=float),
        "c": np.array([0.5], dtype=float),
        "order": 2,
    },
    "implicit_runge_2method": {
        "A": np.array([
            [gamma, 0.0],
            [1.0 - gamma, gamma]
        ], dtype=float),
        "b": np.array([1.0 - gamma, gamma], dtype=float),
        "c": np.array([gamma, 1.0], dtype=float),
        "order": 2,
    },
    "implicit_runge_3method": {
        "A": np.array([
            [0.5, -0.5],
            [0.5,  0.5]
        ], dtype=float),
        "b": np.array([0.5, 0.5], dtype=float),
        "c": np.array([0.0, 1.0], dtype=float),
        "order": 2,
    },
}

def unpack_stages(W, s, dim):
    return W.reshape((s, dim))

def pack_stages(Y):
    return Y.reshape(-1)

def implicit_rk_step(tn, zn, h, A, b, c):
    s = len(b)
    dim = len(zn)

    def G(W):
        Y = unpack_stages(W, s, dim)
        Gval = np.zeros((s, dim), dtype=float)

        Fvals = []
        for j in range(s):
            tj = tn + c[j] * h
            Fvals.append(f(tj, Y[j]))

        for i in range(s):
            rhs = zn.copy()
            for j in range(s):
                rhs += h * A[i, j] * Fvals[j]
            Gval[i] = Y[i] - rhs

        return pack_stages(Gval)

    def JG(W):
        Y = unpack_stages(W, s, dim)

        J = np.zeros((s * dim, s * dim), dtype=float)

        Jf_vals = []
        for j in range(s):
            tj = tn + c[j] * h
            Jf_vals.append(jacobian(tj, Y[j]))

        for i in range(s):
            for j in range(s):
                block = -h * A[i, j] * Jf_vals[j]
                if i == j:
                    block = block + np.eye(dim)

                i0 = i * dim
                i1 = (i + 1) * dim
                j0 = j * dim
                j1 = (j + 1) * dim
                J[i0:i1, j0:j1] = block

        return J

    W0 = np.tile(zn, s)

    W, iters, ok = solve_newton(G, JG, W0)
    if not ok:
        raise RuntimeError(
            f"Ньютон не сошелся для общей системы стадий при t={tn}, итераций: {iters}"
        )

    Y = unpack_stages(W, s, dim)

    K = np.zeros((s, dim), dtype=float)
    for j in range(s):
        tj = tn + c[j] * h
        K[j] = f(tj, Y[j])

    z_next = zn.copy()
    for j in range(s):
        z_next += h * b[j] * K[j]

    return z_next, Y, iters


def solve_implicit_rk(t0, t_end, z0, h, method="sdirk2"):
    if method not in RK_METHODS:
        raise ValueError(f"Неизвестный метод: {method}")

    A = RK_METHODS[method]["A"]
    b = RK_METHODS[method]["b"]
    c = RK_METHODS[method]["c"]

    n_steps = int(np.ceil((t_end - t0) / h))
    dim = len(z0)

    t = np.zeros(n_steps + 1, dtype=float)
    z = np.zeros((n_steps + 1, dim), dtype=float)

    t[0] = t0
    z[0] = np.array(z0, dtype=float)

    tn = t0
    zn = np.array(z0, dtype=float)

    total_newton_iters = 0

    for n in range(n_steps):
        h_step = min(h, t_end - tn)

        zn, Y, iters = implicit_rk_step(tn, zn, h_step, A, b, c)
        total_newton_iters += iters

        tn += h_step
        t[n + 1] = tn
        z[n + 1] = zn

    return t, z, total_newton_iters


if __name__ == "__main__":
    t0 = 0.0
    t_end = 20.0
    z0 = np.array([2.0, 0.0], dtype=float)
    h = 8e-2

    methods = [
        "implicit_runge_1method",
        "implicit_runge_2method",
        "implicit_runge_3method",
    ]

    for method in methods:
        t, z, nit = solve_implicit_rk(t0, t_end, z0, h, method=method)

        x = z[:, 0]
        y = z[:, 1]

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(t, x, label="x(t)")
        plt.plot(t, y, label="y(t)")
        plt.xlabel("t")
        plt.ylabel("solution")
        plt.title(method)
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Фазовый портрет: {method}")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{method}.png", dpi=150, bbox_inches="tight")

    plt.show()