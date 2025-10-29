from generator import generate_system
from check_results import check_results


def lu_decompose(A):
    n = len(A)

    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for k in range(n):
        for j in range(k, n):
            result = 0.0
            for col in range(k):
                result += L[k][col] * U[col][j]
            U[k][j] = A[k][j] - result

        if abs(U[k][k]) < 1e-12:
            print("LU-разложение невозможно: один из ведущих главных миноров вырожден")
            return None, None

        for i in range(k, n):
            if i == k:
                L[i][k] = 1.0
            else:
                result = 0.0
                for col in range(k):
                    result += L[i][col] * U[col][k]
                L[i][k] = (A[i][k] - result) / U[k][k]

    return L, U

def direct_substitution(L, b):
    n = len(L)

    y = [0.0] * n
    for i in range(n):
        result = 0.0
        for j in range(i):
            result += L[i][j] * y[j]
        y[i] = b[i] - result

    return y

def reverse_substitution(U, y):
    n = len(U)

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        result = 0.0
        for j in range(i + 1, n):
            result += U[i][j] * x[j]
        if abs(U[i][i]) < 1e-15:
            print("Обратный ход невозможен: нулевой диагональный элемент U")
            return [0.0] * n
        
        x[i] = (y[i] - result) / U[i][i]

    return x

def solve_lu(A, b):
    L, U = lu_decompose(A)
    if L is None:
        return [0.0] * len(b)
    
    y = direct_substitution(L, b)
    x = reverse_substitution(U, y)

    return x

if __name__ == "__main__":
    A, b = generate_system(100, 10.0, 10.0)

    A_copy = [row[:] for row in A]
    b_copy = b[:]

    print("Solution by the LU decomposition method\n")
    x = solve_lu(A_copy, b_copy)

    eps = 1e-12
    check_results(A, x, b, eps)
