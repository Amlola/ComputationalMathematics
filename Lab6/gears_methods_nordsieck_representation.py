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
        raise RuntimeError("Error in solve_lin")

    x1 = (b[0] * a22 - a12 * b[1]) / det
    x2 = (a11 * b[1] - b[0] * a21) / det

    return np.array([x1, x2], dtype=float)

def solve_newton(F, JF, x0, tol=1e-10, maxiter=20):
    x = x0.copy()

    for i in range(1, maxiter + 1):
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

def euler_step(tn, zn, h):
    t_next = tn + h

    def G(z_next):
        return z_next - zn - h * f(t_next, z_next)

    def JG(z_next):
        return np.eye(len(zn)) - h * jacobian(t_next, z_next)

    z_init = zn + h * f(tn, zn)
    z_next, ok = solve_newton(G, JG, z_init)

    if not ok:
        raise RuntimeError(f"Ньютон не сошелся в Euler при t={tn}")

    return z_next

P = np.array([
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 2.0],
    [0.0, 0.0, 1.0]
], dtype=float)

l_vec = np.array([2.0 / 3.0, 1.0, 1.0 / 3.0], dtype=float)

def build_R1_from_start(t0, z0, h):
    z1 = euler_step(t0, z0, h)
    t1 = t0 + h

    s1_0 = h * f(t0, z0)
    s1_1 = h * f(t1, z1)

    s2_1 = 0.5 * (s1_1 - s1_0)

    R1 = np.zeros((3, len(z0)))
    R1[0] = z1
    R1[1] = s1_1
    R1[2] = s2_1

    return t1, z1, R1

def gear_nordsieck_step(tn, Rn, h):
    t_next = tn + h

    R_pred = P @ Rn
    z_pred = R_pred[0]
    s1_pred = R_pred[1]

    def G(z_next):
        Delta = h * f(t_next, z_next) - s1_pred
        return z_next - z_pred - l_vec[0] * Delta

    def JG(z_next):
        return np.eye(len(z_pred)) - l_vec[0] * h * jacobian(t_next, z_next)

    z_next, ok = solve_newton(G, JG, z_pred)
    if not ok:
        raise RuntimeError(f"Ньютон не сошелся в нордсиковом шаге при t={tn}")

    Delta = h * f(t_next, z_next) - s1_pred

    R_next = np.zeros_like(Rn)
    R_next[0] = z_pred + l_vec[0] * Delta
    R_next[1] = R_pred[1] + l_vec[1] * Delta
    R_next[2] = R_pred[2] + l_vec[2] * Delta

    return R_next

def solve_gear_nordsieck(t0, t_end, z0, h):
    n_steps = int(np.ceil((t_end - t0) / h))

    t = np.zeros(n_steps + 1)
    z = np.zeros((n_steps + 1, len(z0)))

    t[0] = t0
    z[0] = np.array(z0, dtype=float)

    t1, z1, R = build_R1_from_start(t0, z[0], h)

    t[1] = t1
    z[1] = z1

    tn = t1

    for n in range(1, n_steps):
        h_step = min(h, t_end - tn)

        if abs(h_step - h) > 1e-15:
            z_next = euler_step(tn, z[n], h_step)
            tn += h_step
            t[n + 1] = tn
            z[n + 1] = z_next
            break
        else:
            R = gear_nordsieck_step(tn, R, h)
            tn += h
            t[n + 1] = tn
            z[n + 1] = R[0]

    return t, z


if __name__ == "__main__":
    t0 = 0.0
    t_end = 20
    z0 = np.array([2.0, 0.0], dtype=float)

    h = 1e-3

    t, z = solve_gear_nordsieck(t0, t_end, z0, h)

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
    plt.title("Фазовый портрет")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('gear_nordsieck.png', dpi=150, bbox_inches='tight')
    plt.show()