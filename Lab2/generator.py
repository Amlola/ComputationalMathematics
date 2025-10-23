def generate_system(n, a, b):

    A = [[0.0 for i in range(n)] for i in range(n)]
    b_vec = [float(i + 1) for i in range(n)]

    for i in range(n):
        A[i][i] = a

        if i - 1 >= 0:
            A[i][i - 1] = 1.0

        if i + 1 < n:
            A[i][i + 1] = 1.0

        if i + 2 < n:
            A[i][i + 2] = 1.0 / b

    return A, b_vec
