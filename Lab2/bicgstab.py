from math import sqrt
from generator import generate_system
from check_results import get_residual_norm2, check_results
from get_graphics import save_residual_plot


def solve_bicgstab_method(A, b, max_iter, x0=None):
    n = len(A)

    x = [0.0] * n if x0 is None else x0[:]

    residual = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    shadow_residual = residual[:]

    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0

    v_vec = [0.0] * n
    p_vec = [0.0] * n

    residuals = [get_residual_norm2(A, x, b)]
    norm_b = sqrt(sum(bi * bi for bi in b)) if any(b) else 1.0

    for iteration in range(1, max_iter + 1):
        rho_new = sum(shadow_residual[i] * residual[i] for i in range(n))
        if abs(rho_new) < 1e-30:
            print("rho малый — итерации остановлены.")
            break

        if iteration == 1:
            p_vec = residual[:]
        else:
            beta = (rho_new / rho_prev) * (alpha / omega)
            for i in range(n):
                p_vec[i] = residual[i] + beta * (p_vec[i] - omega * v_vec[i])

        v_vec = [sum(A[i][j] * p_vec[j] for j in range(n)) for i in range(n)]

        shadow_v = sum(shadow_residual[i] * v_vec[i] for i in range(n))
        if abs(shadow_v) < 1e-30:
            print("Деление на ноль в вычислении alpha.")
            break

        alpha = rho_new / shadow_v

        s_vec = [residual[i] - alpha * v_vec[i] for i in range(n)]
        t_vec = [sum(A[i][j] * s_vec[j] for j in range(n)) for i in range(n)]

        tt = sum(ti * ti for ti in t_vec)
        if tt == 0.0:
            print("Нулевой знаменатель при вычислении omega.")
            break

        ts = sum(t_vec[i] * s_vec[i] for i in range(n))
        omega = ts / tt

        for i in range(n):
            x[i] += alpha * p_vec[i] + omega * s_vec[i]

        for i in range(n):
            residual[i] = s_vec[i] - omega * t_vec[i]

        abs_res = get_residual_norm2(A, x, b)
        residuals.append(abs_res)

        rel_res = sqrt(sum(ri * ri for ri in residual)) / norm_b
        if rel_res <= 1e-12 or abs(omega) < 1e-30:
            break

        rho_prev = rho_new

    return x, residuals

if __name__ == "__main__":
    print("BiCGStab method\n")

    A, b = generate_system(100, 10.0, 10.0)

    max_iter = 1000

    x, residuals = solve_bicgstab_method(A, b, max_iter, x0=None)

    save_residual_plot(residuals,
                       filename="pictures/bicgstab.png",
                       method_name="Cтабилизированный бисопряжённый градиент")

    norm_b = (sum(bi * bi for bi in b)) ** 0.5 if any(b) else 1.0
    eps_abs = 1e-12 * norm_b

    check_results(A, x, b, eps_abs)
