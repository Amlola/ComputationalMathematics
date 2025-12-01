years = [1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]


population = [84528, 92855, 106360, 120264, 141042, 173855, 204042, 226948, 253785, 279049, 317630]


def thomas_algo(a, b, c, d):
    system_size = len(b)

    alpha = [0.0] * system_size
    beta = [0.0] * system_size

    alpha[0] = c[0] / b[0]
    beta[0] = d[0] / b[0]

    eps = 1e-12

    for i in range(1, system_size):
        diag_mod = b[i] - a[i] * alpha[i - 1]

        if abs(diag_mod) < eps:
            print("Система вырождена")
            return [0]
        
        if i < system_size - 1:
            alpha[i] = c[i] / diag_mod
        else:
            alpha[i] = 0.0

        beta[i] = (d[i] - a[i] * beta[i - 1]) / diag_mod

    x = [0.0] * system_size
    x[-1] = beta[-1]
    for i in range(system_size - 2, -1, -1):
        x[i] = beta[i] - alpha[i] * x[i + 1]

    return x


def get_cubic_spline(x, y):
    number_of_years = len(x)

    if number_of_years < 3:
        print("Не хватает точек для кубического сплайна")
        return x, y, [0.0] * number_of_years

    h = [x[i + 1] - x[i] for i in range(number_of_years - 1)]

    system_size = number_of_years - 2
    if system_size == 0:
        return x, y, [0.0] * number_of_years

    a = [0.0] * system_size 
    b = [0.0] * system_size 
    c = [0.0] * system_size 
    d = [0.0] * system_size

    for i in range(1, number_of_years - 1):
        hi_1 = h[i - 1]

        if i < number_of_years - 1:
            hi = h[i]
        else:
            h[i] = h[-1]

        j = i - 1

        a[j] = hi_1
        b[j] = 2 * (hi_1 + hi)

        if j < system_size - 1:
            c[j] = hi
        else:
            c[j] = 0
            
        d[j] = 6 * ((y[i + 1] - y[i]) / hi - (y[i] - y[i - 1]) / hi_1)

    incomplete_solution = thomas_algo(a, b, c, d)

    solution = [0.0] + incomplete_solution + [0.0]

    return x, y, solution


def spline(x_nodes, y_nodes, second_derivatives, x):
    number_of_years = len(x_nodes)

    if number_of_years < 2:
        print("Не хватает точек для кубического сплайа")
        return -1

    if x <= x_nodes[0]:
        i = 0
    elif x >= x_nodes[-1]:
        i = number_of_years - 2
    else:
        i = 0
        while i < number_of_years - 2 and x > x_nodes[i + 1]:
            i += 1

    x_i = x_nodes[i]
    x_ip1 = x_nodes[i + 1]
    lenght = x_ip1 - x_i

    if lenght == 0:
        print("Нулевой длина отрезка. Две точки совпадают")
        return -1

    derivative_i = second_derivatives[i]
    derivative_i1 = second_derivatives[i + 1]
    yi = y_nodes[i]
    yi1 = y_nodes[i + 1]

    left_len = (x - x_i)
    right_len = (x_ip1 - x)

    curvature_left = derivative_i * (right_len**3) / (6 * lenght)
    curvature_right = derivative_i1 * (left_len**3) / (6 * lenght)

    linear_left = (yi - derivative_i * lenght * lenght / 6) * (right_len / lenght)
    linear_right = (yi1 - derivative_i1 * lenght * lenght / 6) * (left_len / lenght)

    return curvature_left + curvature_right + linear_left + linear_right


if __name__ == "__main__":
    x, y, solution = get_cubic_spline(years, population)

    predict_year = 2020

    print(f"{predict_year}: {spline(x, y, solution, predict_year)}")
