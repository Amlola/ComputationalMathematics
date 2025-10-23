def get_residual_norm2(A, x, b):
    num_rows = len(A)
    res = 0.0
    for i in range(num_rows):
        Ax_i = 0.0
        row = A[i]
        for j in range(num_rows):
            Ax_i += row[j] * x[j]
        diff = Ax_i - b[i]
        res += diff * diff
    return res ** 0.5

def check_results(A, x, b, eps):
    r2 = get_residual_norm2(A, x, b)
    print("Невязка:")
    print(f"||r||_2 = {r2:.3e}\n")
    if r2 <= eps:
        print("CORRECT")
    else:
        print("INCORRECT")
    return r2
