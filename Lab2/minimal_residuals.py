from math import sqrt
from generator import generate_system
from check_results import get_residual_norm2, check_results
from get_graphics import save_residual_plot


def solve_minimal_residuals(A, b, max_iter, x0=None):
    n = len(A)

    x = [0.0] * n if x0 is None else x0[:]

    residual = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    residuals = [get_residual_norm2(A, x, b)]

    norm_b = sqrt(sum(bi * bi for bi in b)) if any(b) else 1.0

    for _ in range(1, max_iter + 1):
        p = [sum(A[i][j] * residual[j] for j in range(n)) for i in range(n)]

        rp = sum(residual[i] * p[i] for i in range(n))
        pp = sum(p[i] * p[i] for i in range(n))
        if pp == 0.0:
            break

        alpha = rp / pp

        for i in range(n):
            x[i] += alpha * residual[i]
            residual[i] -= alpha * p[i]

        abs_res = get_residual_norm2(A, x, b)
        residuals.append(abs_res)

        rel_res = sqrt(sum(ri * ri for ri in residual)) / norm_b
        if rel_res <= 1e-12:
            break

    return x, residuals

if __name__ == "__main__":
    print("Minimal Residuals method\n")

    A, b = generate_system(100, 10.0, 10.0)

    max_iter = 1000

    x, residuals = solve_minimal_residuals(A, b, max_iter, x0=None)

    save_residual_plot(residuals,
                       filename="pictures/minimal_residuals.png",
                       method_name="Метод минимальных невязок")

    norm_b = (sum(bi * bi for bi in b)) ** 0.5 if any(b) else 1.0
    eps_abs = 1e-12 * norm_b

    check_results(A, x, b, eps_abs)
