from generator import generate_system
from check_results import get_residual_norm2
from check_results import check_results
from get_graphics import save_residual_plot


def solve_seidel(A, b, max_iter, x0=None):
    n = len(A)

    x = [0.0] * n if x0 is None else x0[:]

    residuals = [get_residual_norm2(A, x, b)]

    for _ in range(1, max_iter + 1):
        x_prev = x[:]
        for i in range(n):
            a_ii = A[i][i]
            if a_ii == 0.0:
                print("Метод Зейделя: a_ii = 0")
                return x, residuals

            sum_lower = 0.0
            row = A[i]
            for j in range(i):
                sum_lower += row[j] * x[j]

            sum_upper = 0.0
            for j in range(i + 1, n):
                sum_upper += row[j] * x_prev[j]

            x[i] = (b[i] - sum_lower - sum_upper) / a_ii

        r = get_residual_norm2(A, x, b)
        residuals.append(r)
        if r <= 1e-12:
            break

    return x, residuals

def build_T_seidel(A):
    n = len(A)
    
    T = [[0.0] * n for _ in range(n)]

    for j in range(n):
        u_j = [0.0]*n
        for i in range(n):
            if j > i:
                u_j[i] = A[i][j]
            else:
                u_j[i] = 0.0

        y = [0.0] * n
        for i in range(n):
            sum = 0.0
            for k in range(i):
                sum += A[i][k] * y[k]
            d_i = A[i][i]
            if d_i == 0.0:
                print("a_ii = 0: (L+D) необратима")
                return T
            y[i] = (u_j[i] - sum) / d_i

        for i in range(n):
            T[i][j] = -y[i]
    return T

def mat_norm_inf(M):
    return max(sum(abs(v) for v in row) for row in M) if M else 0.0

def check_seidel_convergence(A):
    T = build_T_seidel(A)
    normT = mat_norm_inf(T)
    print(f"{'Теоретически сходится (норма < 1)' if normT < 1.0 else 'Сходимость не гарантирована'}")

    return normT


if __name__ == "__main__":
    A, b = generate_system(100, 10.0, 10.0)

    print("Seidel method\n")

    check_seidel_convergence(A)

    A_copy = [row[:] for row in A]
    b_copy = b[:]

    max_iter = 1000
    x, residuals = solve_seidel(A_copy, b_copy, max_iter, x0=None)

    save_residual_plot(residuals, filename="pictures/seidel.png", method_name="Метод Зейделя")

    eps = 1e-12
    check_results(A, x, b, eps)