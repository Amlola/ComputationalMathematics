import numpy as np
import matplotlib.pyplot as plt

def set_boundary(u):
    u[0, :] = 100.0
    u[:, -1] = 200.0
    u[-1, :] = 300.0
    u[:, 0] = 400.0

    u[0, 0] = (100.0 + 400.0) / 2.0
    u[0, -1] = (100.0 + 200.0) / 2.0
    u[-1, -1] = (200.0 + 300.0) / 2.0
    u[-1, 0] = (300.0 + 400.0) / 2.0

    return u

def initial_grid(N):
    u = np.zeros((N + 1, N + 1))

    u[:, :] = 250.0

    u = set_boundary(u)

    return u

def max_difference(u_new, u_old):
    return np.max(np.abs(u_new - u_old))

def jacobi(N=50, eps=1e-6, max_iter=100000):
    u_old = initial_grid(N)
    u_new = u_old.copy()

    for iteration in range(1, max_iter + 1):
        u_new[1:-1, 1:-1] = 0.25 * (
            u_old[:-2, 1:-1] +
            u_old[2:, 1:-1] +
            u_old[1:-1, :-2] +
            u_old[1:-1, 2:]
        )

        error = max_difference(u_new, u_old)

        if error < eps:
            return u_new, iteration, error

        u_old, u_new = u_new, u_old

    return u_old, max_iter, error

def gauss_seidel(N=50, eps=1e-6, max_iter=100000):
    u = initial_grid(N)

    for iteration in range(1, max_iter + 1):
        error = 0.0

        for i in range(1, N):
            for j in range(1, N):
                old_value = u[i, j]

                u[i, j] = 0.25 * (
                    u[i - 1, j] +
                    u[i + 1, j] +
                    u[i, j - 1] +
                    u[i, j + 1]
                )

                error = max(error, abs(u[i, j] - old_value))

        if error < eps:
            return u, iteration, error

    return u, max_iter, error

def sor(N=50, omega=1.7, eps=1e-6, max_iter=100000):
    u = initial_grid(N)

    for iteration in range(1, max_iter + 1):
        error = 0.0

        for i in range(1, N):
            for j in range(1, N):
                old_value = u[i, j]

                gs_value = 0.25 * (
                    u[i - 1, j] +
                    u[i + 1, j] +
                    u[i, j - 1] +
                    u[i, j + 1]
                )

                u[i, j] = (1.0 - omega) * old_value + omega * gs_value

                error = max(error, abs(u[i, j] - old_value))

        if error < eps:
            return u, iteration, error

    return u, max_iter, error

def plot_solution(u, L, title):
    N = u.shape[0] - 1
    x = np.linspace(0, L, N + 1)
    y = np.linspace(0, L, N + 1)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(X, Y, u, levels=30)
    plt.colorbar(contour, label="")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.show()

def plot_solution_3d(u, L, title):
    N = u.shape[0] - 1

    x = np.linspace(0, L, N + 1)
    y = np.linspace(0, L, N + 1)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(X, Y, u, cmap="viridis", edgecolor="none")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x, y)")
    ax.set_title(title)

    fig.colorbar(surface, ax=ax, shrink=0.6, label="")

    plt.show()

if __name__ == "__main__":
    N = 50
    eps = 1e-6

    u_jacobi, it_jacobi, err_jacobi = jacobi(N=N, eps=eps)
    u_gs, it_gs, err_gs = gauss_seidel(N=N, eps=eps)
    u_sor, it_sor, err_sor = sor(N=N, omega=1.8, eps=eps)

    print("Метод Якоби:")
    print(f"  итераций: {it_jacobi}")

    print("\nМетод Зейделя:")
    print(f"  итераций: {it_gs}")

    print("\nМетод верхней релаксации:")
    print(f"  итераций: {it_sor}")

    L = 0.1

    plot_solution(u_jacobi, L, "метод Якоби")
    plot_solution(u_gs, L, "метод Зейделя")
    plot_solution(u_sor, L, "метод верхней релаксации")

    plot_solution_3d(u_jacobi, L, "метод Якоби")
    plot_solution_3d(u_gs, L, "метод Зейделя")
    plot_solution_3d(u_sor, L, "метод верхней релаксации")