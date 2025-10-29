import os
import matplotlib.pyplot as plt
from generator import generate_system


def save_residual_plot(residuals, filename, method_name):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    r_plot = [max(r, 1e-300) for r in residuals]
    iters = list(range(len(r_plot)))

    plt.figure(figsize=(8, 5))
    plt.semilogy(iters, r_plot, linewidth=1.8)
    plt.xlabel("Итерация")
    plt.ylabel("||A x - b||_2")
    plt.title(f"Убывание невязки: {method_name}")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"График сохранён: {filename}")

def plot_all_methods():
    n = 100
    a = 10.0
    b_param = 10.0
    omega = 1.5

    A, b = generate_system(n, a, b_param)

    curves = []

    from jacoby import solve_jacoby
    A1 = [row[:] for row in A]
    b1 = b[:]
    _, res_jacobi = solve_jacoby(A1, b1, 1000, x0=None)
    curves.append(("Якоби", res_jacobi))

    from seidel import solve_seidel
    A2 = [row[:] for row in A]
    b2 = b[:]
    _, res_seidel = solve_seidel(A2, b2, 1000, x0=None)
    curves.append(("Зейдель", res_seidel))

    from upper_relaxation import solve_upper_relaxation
    A3 = [row[:] for row in A]
    b3 = b[:]
    _, res_sor = solve_upper_relaxation(A3, b3, 1000, omega, x0=None)
    curves.append((f"Верхней релаксации (omega={omega})", res_sor))

    from gradient_descent import solve_gradient_descent
    A4 = [row[:] for row in A]
    b4 = b[:]
    _, res_gd = solve_gradient_descent(A4, b4, 1000, x0=None)
    curves.append(("Градиентного спуска", res_gd))

    from minimal_residuals import solve_minimal_residuals
    A5 = [row[:] for row in A]
    b5 = b[:]
    _, res_mr = solve_minimal_residuals(A5, b5, 1000, x0=None)
    curves.append(("Минимальных невязок", res_mr))

    from bicgstab import solve_bicgstab_method
    A6 = [row[:] for row in A]
    b6 = b[:]
    _, res_bicgstab = solve_bicgstab_method(A6, b6, 1000, x0=None)
    curves.append(("BiCGStab", res_bicgstab))

    os.makedirs("pictures", exist_ok=True)
    plt.figure(figsize=(9, 6))

    for label, residuals in curves:
        r_plot = [max(r, 1e-300) for r in residuals]
        iters = list(range(len(r_plot)))
        plt.semilogy(iters, r_plot, linewidth=1.8, label=label)

    plt.xlabel("Итерация")
    plt.ylabel("||A x - b||_2")
    plt.title("Зависимость невязки от итерации")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pictures/compare.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    plot_all_methods()
