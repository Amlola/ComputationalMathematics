from math import sqrt


x = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
f = [0.0, 0.004, 0.015, 0.034, 0.059, 0.089, 0.123, 0.3, 0.2]

h = x[1] - x[0]


def linear_interpolation(x0):
    for i in range(len(x) - 1):
        if x[i] <= x0 <= x[i + 1]:
            return f[i] + (f[i + 1] - f[i]) * (x0 - x[i]) / (x[i + 1] - x[i])
    return 0.0


def rect_left():
    s = 0.0
    for i in range(len(f) - 1):
        s += f[i]
    return h * s


def rect_right():
    s = 0.0
    for i in range(1, len(f)):
        s += f[i]
    return h * s


def rect_mid():
    s = 0.0
    for i in range(len(x) - 1):
        xm = (x[i] + x[i + 1]) / 2 
        s += linear_interpolation(xm)            
    return h * s


def trapezoid():
    s = (f[0] + f[-1]) / 2
    for i in range(1, len(f) - 1):
        s += f[i]
    return h * s


def simpson():
    s1 = 0.0  
    s2 = 0.0  

    for i in range(1, len(f) - 1):
        if i % 2 == 1:
            s1 += f[i]
        else:
            s2 += f[i]

    return h / 3 * (f[0] + f[-1] + 4 * s1 + 2 * s2)


def gauss(n):
    if n == 2:
        t = [-1.0 / sqrt(3), 1.0 / sqrt(3)]
        w = [1.0, 1.0]
    elif n == 3:
        t = [-sqrt(3/5), 0.0, sqrt(3/5)]
        w = [5/9, 8/9, 5/9]
    else: 
        t = [-0.861136, -0.339981, 0.339981, 0.861136]
        w = [0.347855, 0.652145, 0.652145, 0.347855]

    s = 0.0
    for i in range(n):
        xi = 1 + t[i]       
        s += w[i] * linear_interpolation(xi)  

    return s


if __name__ == "__main__":
    print("Метод прямоугольники левые:", rect_left())
    print("Метод прямоугольники правые:", rect_right())
    print("Метод прямоугольники середины:", rect_mid())
    print("Метод трапеции:", trapezoid())
    print("Метод симпсона:", simpson())
    print("Метод гаусс по 2 точкам:", gauss(2))
    print("Метод гаусс по 3 точкам:", gauss(3))
    print("Метод гаусс по 4 точкам:", gauss(4))
