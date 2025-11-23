def f(x):
    return 3 * x + 4 * x**3 - 12 * x**2 - 5


def derivative(x):
    return 3 + 12 * x**2 - 24 * x 


def half_division(a, b, eps, max_iter):
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        print("Метод половинного деления: Концы отрезка одного знака")
        return None, 0

    iterations = 0

    while (b - a) / 2 > eps and iterations < max_iter:
        c = (a + b) / 2
        fc = f(c)

        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iterations += 1

    return (a + b) / 2, iterations


def phi(x):
    alpha = 0.02
    return x - alpha * f(x)


def simple_iteration(x0, eps, max_iter):
    x = x0
    iterations = 0

    for i in range(max_iter):
        x_new = phi(x)

        iterations += 1

        if abs(x_new - x) < eps:
            return x_new, iterations

        x = x_new

    print("Метод простой итерации не сошёлся")
    return None, iterations


def newton(x0, eps, max_iter):
    x = x0
    iterations = 0

    for i in range(max_iter):
        df = derivative(x)
        if df == 0:
            print("метод Ньютона: нулевая производная")
            return None, iterations

        x_new = x - f(x) / df
        iterations += 1

        if abs(x_new - x) < eps:
            return x_new, iterations

        x = x_new

    print("Метод Ньютона не сошёлся")
    return None, iterations


def modified_newton(x0, eps, max_iter):
    x = x0
    df0 = derivative(x0)
    iterations = 0

    if df0 == 0:
        print("Модифицированный метод Ньютона: нулевая производная")
        return None, iterations

    for i in range(max_iter):
        x_new = x - f(x) / df0
        iterations += 1

        if abs(x_new - x) < eps:
            return x_new, iterations

        x = x_new

    print("Модифицированный метод Ньютона не сошёлся")
    return None, iterations


if __name__ == "__main__":
    eps = 1e-12
    max_iteration = 500

    a = 2
    b = 3

    x0 = 2.5

    print("\nМетод половинного деления:")
    r1, it1 = half_division(a, b, eps, max_iteration)
    print("x =", r1,"\nf(x) =", f(r1))
    print("Число итераций:", it1)
    print("/*-----------------------------*\\")

    print("\nМетод простой итерации:")
    r2, it2 = simple_iteration(x0, eps, max_iteration)
    print("x =", r2, "\nf(x) =", f(r2))
    print("Число итераций:", it2)
    print("/*-----------------------------*\\")

    print("\nМетод Ньютона:")
    r3, it3 = newton(x0, eps, max_iteration)
    print("x =", r3, "\nf(x) =", f(r3))
    print("Число итераций:", it3)
    print("/*-----------------------------*\\")

    print("\nМодифицированный метод Ньютона:")
    r4, it4 = modified_newton(x0, eps, max_iteration)
    print("x =", r4, "\nf(x) =", f(r4))
    print("Число итераций:", it4)
    print("/*-----------------------------*\\")

