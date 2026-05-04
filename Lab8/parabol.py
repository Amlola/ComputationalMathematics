import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Nx = 50
Ny = 50
T = 0.1

hx = 1.0 / Nx
hy = 1.0 / Ny

x = np.linspace(0.0, 1.0, Nx + 1)
y = np.linspace(0.0, 1.0, Ny + 1)
X, Y = np.meshgrid(x, y)

tau_stable_explicit_split = 0.5 * min(hx**2, hy**2)

tau = 0.45 * tau_stable_explicit_split
Nt = int(np.ceil(T / tau))
tau = T / Nt

rx = tau / hx**2
ry = tau / hy**2

print(f"Nx={Nx}, Ny={Ny}, T={T}")
print(f"hx={hx:.5f}, hy={hy:.5f}")
print(f"tau={tau:.8f}, Nt={Nt}")
print(f"rx=tau/hx^2={rx:.5f}, ry=tau/hy^2={ry:.5f}")

def thomas_algorithm(a, b, c, d):
    n = len(d)

    cp = np.zeros(n, dtype=float)
    dp = np.zeros(n, dtype=float)

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

    y = np.zeros(n, dtype=float)
    y[-1] = dp[-1]

    for i in range(n - 2, -1, -1):
        y[i] = dp[i] - cp[i] * y[i + 1]

    return y


def apply_boundary_conditions(U):
    U[:, 0] = 0.0
    U[:, -1] = 1.0

    U[0, :] = 2.0
    U[-1, :] = 3.0

    U[0, 0] = 0.5 * (0.0 + 2.0)
    U[0, -1] = 0.5 * (1.0 + 2.0)
    U[-1, 0] = 0.5 * (0.0 + 3.0)
    U[-1, -1] = 0.5 * (1.0 + 3.0)

def initial_grid():
    U = np.zeros((Ny + 1, Nx + 1), dtype=float)
    apply_boundary_conditions(U)
    return U

def Lambda1(U):
    return (U[1:-1, :-2] - 2.0 * U[1:-1, 1:-1] + U[1:-1, 2:]) / hx**2

def Lambda2(U):
    return (U[:-2, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[2:, 1:-1]) / hy**2

def tridiagonal_coefficients_for_x(alpha):
    n = Nx - 1

    lower = np.full(n, -alpha / hx**2, dtype=float)
    main = np.full(n, 1.0 + 2.0 * alpha / hx**2, dtype=float)
    upper = np.full(n, -alpha / hx**2, dtype=float)

    lower[0] = 0.0
    upper[-1] = 0.0

    return lower, main, upper

def tridiagonal_coefficients_for_y(alpha):
    n = Ny - 1

    lower = np.full(n, -alpha / hy**2, dtype=float)
    main = np.full(n, 1.0 + 2.0 * alpha / hy**2, dtype=float)
    upper = np.full(n, -alpha / hy**2, dtype=float)

    lower[0] = 0.0
    upper[-1] = 0.0

    return lower, main, upper

def solve_implicit_x(rhs_internal, alpha):
    rhs = rhs_internal.copy()

    left_boundary = 0.0
    right_boundary = 1.0

    rhs[:, 0] += alpha * left_boundary / hx**2
    rhs[:, -1] += alpha * right_boundary / hx**2

    a, b, c = tridiagonal_coefficients_for_x(alpha)

    V_internal = np.zeros_like(rhs)

    for j in range(Ny - 1):
        V_internal[j, :] = thomas_algorithm(a, b, c, rhs[j, :])

    V = np.zeros((Ny + 1, Nx + 1), dtype=float)
    apply_boundary_conditions(V)
    V[1:-1, 1:-1] = V_internal

    return V

def solve_implicit_y(rhs_internal, alpha):
    rhs = rhs_internal.copy()

    bottom_boundary = 2.0
    top_boundary = 3.0

    rhs[0, :] += alpha * bottom_boundary / hy**2
    rhs[-1, :] += alpha * top_boundary / hy**2

    a, b, c = tridiagonal_coefficients_for_y(alpha)

    V_internal = np.zeros_like(rhs)

    for i in range(Nx - 1):
        V_internal[:, i] = thomas_algorithm(a, b, c, rhs[:, i])

    V = np.zeros((Ny + 1, Nx + 1), dtype=float)
    apply_boundary_conditions(V)
    V[1:-1, 1:-1] = V_internal

    return V

def scheme_a():
    U = initial_grid()

    for _ in range(Nt):
        U_tilde = solve_implicit_x(U[1:-1, 1:-1], tau)
        U = solve_implicit_y(U_tilde[1:-1, 1:-1], tau)

    return U

def scheme_b(xi=0.5):
    U = initial_grid()

    for _ in range(Nt):
        rhs_x = U[1:-1, 1:-1] + tau * (1.0 - xi) * Lambda1(U)
        U_half = solve_implicit_x(rhs_x, tau * xi)

        rhs_y = U_half[1:-1, 1:-1] + tau * (1.0 - xi) * Lambda2(U_half)
        U = solve_implicit_y(rhs_y, tau * xi)

    return U

def scheme_c():
    U = initial_grid()

    for _ in range(Nt):
        rhs_x = U[1:-1, 1:-1] + 0.5 * tau * Lambda2(U)
        U_tilde = solve_implicit_x(rhs_x, 0.5 * tau)

        rhs_y = U_tilde[1:-1, 1:-1] + 0.5 * tau * Lambda1(U_tilde)
        U = solve_implicit_y(rhs_y, 0.5 * tau)

    return U

def scheme_d():
    U = initial_grid()

    for _ in range(Nt):
        U_tilde = U.copy()
        U_tilde[1:-1, 1:-1] = U[1:-1, 1:-1] + tau * Lambda1(U)
        apply_boundary_conditions(U_tilde)

        U_next = U_tilde.copy()
        U_next[1:-1, 1:-1] = U_tilde[1:-1, 1:-1] + tau * Lambda2(U_tilde)
        apply_boundary_conditions(U_next)

        U = U_next

    return U

def plot_solution(U, title):
    plt.figure(figsize=(7, 6))
    cs = plt.contourf(X, Y, U, levels=50)
    plt.colorbar(cs, label="u(t, x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title}, T={T}")
    plt.tight_layout()

    plt.show()

def plot_solution_3d(U, title):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, U, cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(t, x, y)')
    ax.set_title(f"{title}, T={T}")

    fig.colorbar(surf, shrink=0.6, aspect=12, label="u(t, x, y)")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    solutions = {
        "a": scheme_a(),
        "б": scheme_b(xi=0.5),
        "в": scheme_c(),
        "г": scheme_d(),
    }

    plot_solution(
        solutions["a"],
        "Схема а"
    )

    plot_solution(
        solutions["б"],
        "Схема б"
    )

    plot_solution(
        solutions["в"],
        "Схема в"
    )

    plot_solution(
        solutions["г"],
        "Схема г"
    )

    plot_solution_3d(
        solutions["a"],
        "Схема а"
    )

    plot_solution_3d(
        solutions["б"],
        "Схема б"
    )

    plot_solution_3d(
        solutions["в"],
        "Схема в"
    )

    plot_solution_3d(
        solutions["г"],
        "Схема г"
    )
