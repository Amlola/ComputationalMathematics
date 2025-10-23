from generator import generate_system
from check_results import check_results

def find_main_element(A, active_rows, active_cols):
    max_elem_row = active_rows[0]
    max_elem_col = active_cols[0]
    max_abs_val = abs(A[max_elem_row][max_elem_col])

    for i in active_rows:
        for j in active_cols:
            val = abs(A[i][j])
            if val > max_abs_val:
                max_abs_val = val
                max_elem_row, max_elem_col = i, j
    return max_elem_row, max_elem_col, max_abs_val

def eliminate_column(A, b, max_elem_row, max_elem_col, active_rows, active_cols):
    max_value = A[max_elem_row][max_elem_col]
    for i in active_rows:
        if i == max_elem_row:
            continue
        factor = A[i][max_elem_col] / max_value
        if factor == 0.0:
            continue
        for j in active_cols:
            A[i][j] -= factor * A[max_elem_row][j]
        b[i] -= factor * b[max_elem_row]
        A[i][max_elem_col] = 0.0

def reverse_stroke(equations, n):
    x = [0.0] * n
    while equations:
        col, max_val, coeffs, b = equations.pop()
        res = sum(coeffs[j] * x[j] for j in coeffs)
        x[col] = (b - res) / max_val
    return x

def solve_gauss(A, b):
    n = len(A)

    active_rows = list(range(n))
    active_cols = list(range(n))
    equations = []

    while active_cols:
        max_elem_row, max_elem_col, max_val = find_main_element(A, active_rows, active_cols)
        main_val = A[max_elem_row][max_elem_col]

        if abs(main_val) < 1e-15:
            print("Матрица вырождена")
            return [0.0] * n

        eliminate_column(A, b, max_elem_row, max_elem_col, active_rows, active_cols)

        coeffs = {j: A[max_elem_row][j] for j in active_cols if j != max_elem_col}
        equations.append((max_elem_col, main_val, coeffs, b[max_elem_row]))

        active_rows.remove(max_elem_row)
        active_cols.remove(max_elem_col)

    return reverse_stroke(equations, n)

if __name__ == "__main__":
    A, b = generate_system(100, 10.0, 10.0)

    A_copy = [row[:] for row in A]
    b_copy = b[:]

    print("Gaussian solution with choice of the principal element\n")
    x = solve_gauss(A_copy, b_copy)

    eps = 1e-12
    check_results(A, x, b, eps)
