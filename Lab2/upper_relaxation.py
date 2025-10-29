from generator import generate_system
from check_results import get_residual_norm2, check_results
from get_graphics import save_residual_plot


def solve_upper_relaxation(A, b, max_iter, omega, x0=None):
    n = len(A)

    x = [0.0] * n if x0 is None else x0[:]

    residuals = [get_residual_norm2(A, x, b)]

    for _ in range(1, max_iter + 1):
        x_prev = x[:]
        for i in range(n):
            a_ii = A[i][i]
            if a_ii == 0.0:
                print("Метод Upper relaxation: a_ii = 0")
                return x, residuals

            sum_lower = 0.0
            row = A[i]
            for j in range(i):
                sum_lower += row[j] * x[j]

            sum_upper = 0.0
            for j in range(i + 1, n):
                sum_upper += row[j] * x_prev[j]

            x_gs = (b[i] - sum_lower - sum_upper) / a_ii 
            x[i] = (1.0 - omega) * x_prev[i] + omega * x_gs

        r = get_residual_norm2(A, x, b)
        residuals.append(r)
        if r <= 1e-12:
            break

    return x, residuals

def build_T_sor(A, omega):
    n = len(A)

    T = [[0.0] * n for _ in range(n)]

    for j in range(n):
        right = [0.0] * n
        for i in range(n):
            if i == j:
                right[i] = (1.0 - omega) * A[i][i] 
            elif j > i:
                right[i] = -omega * A[i][j]   
            else:
                right[i] = 0.0

        y = [0.0] * n
        for i in range(n):
            sum = 0.0
            for k in range(i):
                sum += (omega * A[i][k]) * y[k]
            d_i = A[i][i]
            if d_i == 0.0:
                print("(D + ωL) необратима: a_ii = 0 — критерий не вычислен")
                return T
            y[i] = (right[i] - sum) / d_i

        for i in range(n):
            T[i][j] = y[i]
    return T

def mat_norm_inf(M):
    return max(sum(abs(v) for v in row) for row in M) if M else 0.0

def check_sor_convergence(A, omega):
    T = build_T_sor(A, omega)
    normT = mat_norm_inf(T)
    print(f"{'Теоретически сходится (норма < 1)' if normT < 1.0 else 'Сходимость не гарантирована'}")
    return normT

if __name__ == "__main__":
    A, b = generate_system(100, 10.0, 10.0)

    omega = 1.5
    max_iter = 1000

    print(f"Upper relaxation method (omega = {omega})\n")

    check_sor_convergence(A, omega)

    A_copy = [row[:] for row in A]
    b_copy = b[:]
    x, residuals = solve_upper_relaxation(A_copy, b_copy, max_iter, omega, x0=None)

    save_residual_plot(residuals,
                       filename="pictures/upper_relaxation.png",
                       method_name=f"Метод верхней релаксации (ω={omega})")

    eps = 1e-12
    check_results(A, x, b, eps)
