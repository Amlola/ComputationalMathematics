from generator import generate_system
from check_results import get_residual_norm2, check_results
from get_graphics import save_residual_plot


def solve_jacoby(A, b, max_iter, x0=None):
    n = len(A)

    x = [0.0] * n if x0 is None else x0[:]
    x_new = [0.0] * n

    residuals = []
    residual = get_residual_norm2(A, x, b)
    residuals.append(residual)

    for k in range(1, max_iter + 1):
        for i in range(n):
            a_ii = A[i][i]
            if a_ii == 0.0:
                print("Метод Якоби: нулевая диагональ")
                return x, residuals
            sum = 0.0
            row = A[i]
            for j in range(n):
                if j != i:
                    sum += row[j] * x[j]
            x_new[i] = (b[i] - sum) / a_ii

        x, x_new = x_new, x

        residual = get_residual_norm2(A, x, b)
        residuals.append(residual)
        if residual <= 1e-12:
            break

    return x, residuals

def is_strictly_diagonally_dominant(A):
    n = len(A)

    for i in range(n):
        diag = abs(A[i][i])
        off = 0.0
        for j in range(n):
            if j != i:
                off += abs(A[i][j])
        if diag <= off:
            return False
    return True

if __name__ == "__main__":

    A, b = generate_system(100, 10.0, 10.0)

    if not is_strictly_diagonally_dominant(A):
        print("Матрица не строго диагонально доминирующая — сходимость Якоби не гарантирована.")

    A_copy = [row[:] for row in A]
    b_copy = b[:]

    print("Jacoby method\n")
    max_iter = 1000
    x, residuals = solve_jacoby(A_copy, b_copy, max_iter, x0=None)

    save_residual_plot(residuals, filename="pictures/jacobi.png", method_name="Метод Якоби")

    eps = 1e-12
    check_results(A, x, b, eps)
