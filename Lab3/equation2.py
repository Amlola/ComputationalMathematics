import math


def f(x):
    return x**2 - math.exp(x/5)


def derivative(x):
    return 2 * x - 0.2 * math.exp(x / 5)


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


def simple_iteration(x0, eps, max_iter):

    def phi_left(x):
        return -math.sqrt(math.exp(x/5))

    def phi_right(x):
        return math.sqrt(math.exp(x/5))

    if x0 < 0:
        phi = phi_left
    else:
        phi = phi_right

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

        x_new = x - f(x)/df
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
        x_new = x - f(x)/df0
        iterations += 1

        if abs(x_new - x) < eps:
            return x_new, iterations

        x = x_new

    print("Модифицированный метод Ньютона не сошёлся")
    return None, iterations


if __name__ == "__main__":
    eps = 1e-12
    max_iteration = 500

    left_interval = (-3, 0)
    right_interval = (0, 3)

    x0_left = -1.0
    x0_right = 2.0

    print("\nМетод половинного деления (левый корень):")
    r1, it1 = half_division(left_interval[0], left_interval[1], eps, max_iteration)
    print("x =", r1, "\nf(x) =", f(r1))
    print("Число итераций:", it1)
    print("/*-----------------------------*\\")

    print("\nМетод половинного деления (правый корень):")
    r2, it2 = half_division(right_interval[0], right_interval[1], eps, max_iteration)
    print("x =", r2, "\nf(x) =", f(r2))
    print("Число итераций:", it2)
    print("/*-----------------------------*\\")

    print("\nМетод простой итерации (левый корень):")
    r3, it3 = simple_iteration(x0_left, eps, max_iteration)
    print("x =", r3, "\nf(x) =", f(r3))
    print("Число итераций:", it3)
    print("/*-----------------------------*\\")

    print("\nМетод простой итерации (правый корень):")
    r4, it4 = simple_iteration(x0_right, eps, max_iteration)
    print("x =", r4, "\nf(x) =", f(r4))
    print("Число итераций:", it4)
    print("/*-----------------------------*\\")

    print("\nМетод Ньютона (левый корень):")
    r5, it5 = newton(x0_left, eps, max_iteration)
    print("x =", r5, "\nf(x) =", f(r5))
    print("Число итераций:", it5)
    print("/*-----------------------------*\\")

    print("\nМетод Ньютона (правый корень):")
    r6, it6 = newton(x0_right, eps, max_iteration)
    print("x =", r6, "\nf(x) =", f(r6))
    print("Число итераций:", it6)
    print("/*-----------------------------*\\")

    print("\nМодифицированный метод Ньютона (левый корень):")
    r7, it7 = modified_newton(x0_left, eps, max_iteration)
    print("x =", r7, "\nf(x) =", f(r7))
    print("Число итераций:", it7)
    print("/*-----------------------------*\\")

    print("\nМодифицированный метод Ньютона (правый корень):")
    r8, it8 = modified_newton(x0_right, eps, max_iteration)
    print("x =", r8, "\nf(x) =", f(r8))
    print("Число итераций:", it8)
    print("/*-----------------------------*/")