import math


def f1(x, y):
    return math.cos(x - 1) + y - 0.5


def f2(x, y):
    return x - math.cos(y) - 3


def y_from_f1(x):
    return 0.5 - math.cos(x - 1)


def get_F(x):
    return f2(x, y_from_f1(x))


def half_division(a, b, eps, max_iter):
    fa = get_F(a)
    fb = get_F(b)

    if fa * fb > 0:
        print("Метод половинного деления: Концы отрезка одного знака")
        return None, None, 0

    iterations = 0

    while (b - a) / 2 > eps and iterations < max_iter:
        c = (a + b) / 2
        fc = get_F(c)

        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iterations += 1

    x = (a + b) / 2
    y = y_from_f1(x)
    return x, y, iterations


def simple_iteration(x0, y0, eps, max_iter):
    x = x0
    y = y0
    iterations = 0

    while iterations < max_iter:
        x_new = 3 + math.cos(y)
        y_new = 0.5 - math.cos(x_new - 1)

        if max(abs(x_new - x), abs(y_new - y)) < eps:
            return x_new, y_new, iterations + 1

        x = x_new
        y = y_new
        iterations += 1

    print("Метод простой итерации не сошёлся")
    return x, y, iterations


def newton(x0, y0, eps, max_iter):
    x = x0
    y = y0
    iterations = 0

    while iterations < max_iter:
        F1 = f1(x, y)
        F2 = f2(x, y)

        if max(abs(F1), abs(F2)) < eps:
            return x, y, iterations

        J11 = -math.sin(x - 1)
        J12 = 1.0
        J21 = 1.0
        J22 = math.sin(y)

        det = J11 * J22 - J12 * J21
        if det == 0:
            print("метод Ньютона: Якобиан вырожден")
            return x, y, iterations

        dx = (-F1 * J22 - (-F2) * J12) / det
        dy = (J11 * (-F2) - J21 * (-F1)) / det

        x = x + dx
        y = y + dy
        iterations += 1

    print("Метод Ньютона не сошёлся")
    return x, y, iterations


def modified_newton(x0, y0, eps, max_iter):
    x = x0
    y = y0
    iterations = 0

    J11 = -math.sin(x0 - 1)
    J12 = 1.0
    J21 = 1.0
    J22 = math.sin(y0)

    det = J11 * J22 - J12 * J21
    if det == 0:
        print("Модифицированный метод Ньютона: Якобиан вырожден")
        return x, y, iterations

    inv11 = J22 / det
    inv12 = -J12 / det
    inv21 = -J21 / det
    inv22 = J11 / det

    while iterations < max_iter:
        F1 = f1(x, y)
        F2 = f2(x, y)

        if max(abs(F1), abs(F2)) < eps:
            return x, y, iterations

        dx = -(inv11 * F1 + inv12 * F2)
        dy = -(inv21 * F1 + inv22 * F2)

        x = x + dx
        y = y + dy
        iterations += 1

    print("Модифицированный метод Ньютона не сошёлся")
    return x, y, iterations


if __name__ == "__main__":
    eps = 1e-12
    max_iteration = 500

    a = 2.0
    b = 4.0

    x0 = 3.0
    y0 = 0.5

    print("\nМетод половинного деления:")
    xh, yh, it_h = half_division(a, b, eps, max_iteration)
    print("x =", xh, "\ny =", yh)
    print("f1(x, y) =", f1(xh, yh))
    print("f2(x, y) =", f2(xh, yh))
    print("Число итераций:", it_h)
    print("/*-----------------------------*\\")

    print("\nМетод простой итерации:")
    xs, ys, it_s = simple_iteration(x0, y0, eps, max_iteration)
    print("x =", xs, "\ny =", ys)
    print("f1(x, y) =", f1(xs, ys))
    print("f2(x, y) =", f2(xs, ys))
    print("Число итераций:", it_s)
    print("/*-----------------------------*\\")

    print("\nМетод Ньютона:")
    xn, yn, it_n = newton(x0, y0, eps, max_iteration)
    print("x =", xn, "\ny =", yn)
    print("f1(x, y) =", f1(xn, yn))
    print("f2(x, y) =", f2(xn, yn))
    print("Число итераций:", it_n)
    print("/*-----------------------------*\\")

    print("\nМодифицированный метод Ньютона:")
    xm, ym, it_m = modified_newton(x0, y0, eps, max_iteration)
    print("x =", xm, "\ny =", ym)
    print("f1(x, y) =", f1(xm, ym))
    print("f2(x, y) =", f2(xm, ym))
    print("Число итераций:", it_m)
    print("/*-----------------------------*\\")
