import math


def f1(x, y):
    return (x - 1.4)**2 - (y - 0.6)**2 - 1


def f2(x, y):
    return 4.2 * x**2 + 8.8 * y**2 - 1.42


def y_from_f2_plus(x):
    v = 1.42 - 4.2 * x**2
    if v < 0:
        return None
    return math.sqrt(v / 8.8)


def y_from_f2_minus(x):
    v = 1.42 - 4.2 * x**2
    if v < 0:
        return None
    return -math.sqrt(v / 8.8)


def get_F_plus(x):
    y = y_from_f2_plus(x)
    if y is None:
        return None
    return f1(x, y)


def get_F_minus(x):
    y = y_from_f2_minus(x)
    if y is None:
        return None
    return f1(x, y)


def half_division_plus(a, b, eps, max_iter):
    fa = get_F_plus(a)
    fb = get_F_plus(b)

    if fa is None or fb is None or fa * fb > 0:
        print("Плюс-ветвь метод половинного деления: концы отрезка одного знака или вне области")
        return None, None, 0

    iterations = 0

    while (b - a) / 2 > eps and iterations < max_iter:
        c = (a + b) / 2
        fc = get_F_plus(c)

        if fc is None:
            c = (a + b) / 2
            b = c
            iterations += 1
            continue

        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iterations += 1

    x = (a + b) / 2
    y = y_from_f2_plus(x)
    return x, y, iterations


def half_division_minus(a, b, eps, max_iter):
    fa = get_F_minus(a)
    fb = get_F_minus(b)

    if fa is None or fb is None or fa * fb > 0:
        print("Минус-ветвь метод половинного деления: концы отрезка одного знака или вне области")
        return None, None, 0

    iterations = 0

    while (b - a) / 2 > eps and iterations < max_iter:
        c = (a + b) / 2
        fc = get_F_minus(c)

        if fc is None:
            c = (a + b) / 2
            b = c
            iterations += 1
            continue

        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iterations += 1

    x = (a + b) / 2
    y = y_from_f2_minus(x)
    return x, y, iterations


def simple_iteration(x0, y0, eps, max_iter):
    x = x0
    y = y0
    iterations = 0

    while iterations < max_iter:
        x_new = 1.4 - math.sqrt(1 + (y - 0.6)**2)

        v = 1.42 - 4.2 * x_new**2
        if v < 0:
            print("Метод простой итерации: вышли за эллипс")
            return x, y, iterations

        y_abs = math.sqrt(v / 8.8)
        if y >= 0:
            y_new = y_abs
        else:
            y_new = -y_abs

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

        J11 = 2 * (x - 1.4)
        J12 = -2 * (y - 0.6)
        J21 = 8.4 * x
        J22 = 17.6 * y

        det = J11 * J22 - J12 * J21
        if det == 0:
            print("Метод Ньютона: якобиан вырожден")
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

    J11 = 2 * (x0 - 1.4)
    J12 = -2 * (y0 - 0.6)
    J21 = 8.4 * x0
    J22 = 17.6 * y0

    det = J11 * J22 - J12 * J21
    if det == 0:
        print("Модифицированный метод Ньютона: якобиан вырожден")
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

    a_plus, b_plus = 0.3, 0.4
    a_minus, b_minus = -0.1, 0.0

    x0_1 = 0.35
    y0_1 = y_from_f2_plus(x0_1)  

    x0_2 = -0.02
    y0_2 = y_from_f2_minus(x0_2)

    print("\nМетод половинного деления (ветвь y > 0):")
    x1, y1, it1 = half_division_plus(a_plus, b_plus, eps, max_iteration)
    print("x =", x1, "\ny =", y1)
    print("f1 =", f1(x1, y1), "\nf2 =", f2(x1, y1))
    print("Число итераций:", it1)
    print("/*-----------------------------*\\")

    print("\nМетод половинного деления (ветвь y < 0):")
    x2, y2, it2 = half_division_minus(a_minus, b_minus, eps, max_iteration)
    print("x =", x2, "\ny =", y2)
    print("f1 =", f1(x2, y2), "\nf2 =", f2(x2, y2))
    print("Число итераций:", it2)
    print("/*-----------------------------*\\")

    print("\nМетод простой итерации (корень с y > 0):")
    xs1, ys1, its1 = simple_iteration(x0_1, y0_1, eps, max_iteration)
    print("x =", xs1, "\ny =", ys1)
    print("f1 =", f1(xs1, ys1), "\nf2 =", f2(xs1, ys1))
    print("Число итераций:", its1)
    print("/*-----------------------------*\\")

    print("\nМетод простой итерации (корень с y < 0):")
    xs2, ys2, its2 = simple_iteration(x0_2, y0_2, eps, max_iteration)
    print("x =", xs2, "\ny =", ys2)
    print("f1 =", f1(xs2, ys2), "\nf2 =", f2(xs2, ys2))
    print("Число итераций:", its2)
    print("/*-----------------------------*\\")

    print("\nМетод Ньютона (корень с y > 0):")
    xn1, yn1, itn1 = newton(x0_1, y0_1, eps, max_iteration)
    print("x =", xn1, "\ny =", yn1)
    print("f1 =", f1(xn1, yn1), "\nf2 =", f2(xn1, yn1))
    print("Число итераций:", itn1)
    print("/*-----------------------------*\\")

    print("\nМетод Ньютона (корень с y < 0):")
    xn2, yn2, itn2 = newton(x0_2, y0_2, eps, max_iteration)
    print("x =", xn2, "\ny =", yn2)
    print("f1 =", f1(xn2, yn2), "\nf2 =", f2(xn2, yn2))
    print("Число итераций:", itn2)
    print("/*-----------------------------*\\")

    print("\nМодифицированный метод Ньютона (корень с y > 0):")
    xm1, ym1, itm1 = modified_newton(x0_1, y0_1, eps, max_iteration)
    print("x =", xm1, "\ny =", ym1)
    print("f1 =", f1(xm1, ym1), "\nf2 =", f2(xm1, ym1))
    print("Число итераций:", itm1)
    print("/*-----------------------------*\\")

    print("\nМодифицированный метод Ньютона (корень с y < 0):")
    xm2, ym2, itm2 = modified_newton(x0_2, y0_2, eps, max_iteration)
    print("x =", xm2, "\ny =", ym2)
    print("f1 =", f1(xm2, ym2), "\nf2 =", f2(xm2, ym2))
    print("Число итераций:", itm2)
    print("/*-----------------------------*/")
