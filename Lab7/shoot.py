import math
import matplotlib.pyplot as plt

def rk4_step(f, x, Y, h):
    k1 = f(x, Y)
    k2 = f(x + 0.5 * h, [Y[i] + 0.5 * h * k1[i] for i in range(len(Y))])
    k3 = f(x + 0.5 * h, [Y[i] + 0.5 * h * k2[i] for i in range(len(Y))])
    k4 = f(x + h, [Y[i] + h * k3[i] for i in range(len(Y))])

    return [
        Y[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(len(Y))
    ]

def augmented_system(x, Y, eps=1e-10):
    y, yp, u1, u2 = Y

    y_safe = max(y, 0.0)
    sqrt_y = math.sqrt(y_safe)
    denom = max(sqrt_y, eps)

    dy_dx = yp
    dyp_dx = x * sqrt_y

    du1_dx = u2
    du2_dx = x * u1 / (2.0 * denom)

    return [dy_dx, dyp_dx, du1_dx, du2_dx]


def integrate_with_s(s, a=0.0, b=1.0, n=1000):
    h = (b - a) / n
    x = a
    Y = [0.0, s, 0.0, 1.0]

    xs = [x]
    sol = [Y[:]]

    for _ in range(n):
        Y = rk4_step(augmented_system, x, Y, h)
        x += h
        xs.append(x)
        sol.append(Y[:])

    return xs, sol


def shooting_newton(s0, tol=1e-8, max_iter=20, a=0.0, b=1.0, n=1000):
    s = s0
    s_history = []
    F_history = []
    curves = []

    for k in range(max_iter):
        xs, sol = integrate_with_s(s, a, b, n)

        y_b = sol[-1][0]
        u1_b = sol[-1][2]
        F = y_b - 2.0

        s_history.append(s)
        F_history.append(F)
        curves.append((xs[:], [state[0] for state in sol], s, F))

        print(f"iter = {k:2d}, s = {s:.12f}, y(1) = {y_b:.12f}, F = {F:.12e}")

        if abs(F) < tol:
            return s, xs, sol, s_history, F_history, curves

        if abs(u1_b) < 1e-14:
            raise RuntimeError("Производная F'(s) слишком мала, метод Ньютона неустойчив.")

        s = s - F / u1_b

    raise RuntimeError("Метод Ньютона не сошелся за заданное число итераций.")


def plot_results(xs_star, sol_star, s_history, F_history, curves):
    y_star = [state[0] for state in sol_star]

    plt.figure(figsize=(8, 5))
    for i, (xs, ys, s, F) in enumerate(curves):
        plt.plot(xs, ys, label=f"iter {i}: s={s:.6f}")
    plt.axhline(2.0, linestyle="--", label="y(1)=2 (целевое значение)")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Интегральные кривые y(x) на итерациях метода стрельбы")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(xs_star, y_star, linewidth=2, label="Итоговое решение")
    plt.scatter([0, 1], [0, 2], zorder=3, label="Граничные условия")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Финальная интегральная кривая")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(s_history)), s_history, marker="o")
    plt.xlabel("Номер итерации")
    plt.ylabel("s_k = y'(0)")
    plt.title("Изменение пристрелочного параметра")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(F_history)), [abs(f) for f in F_history], marker="o")
    plt.xlabel("Номер итерации")
    plt.ylabel("|F(s_k)| = |y(1; s_k) - 2|")
    plt.title("Убывание невязки")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    s0 = 2.0

    s_star, xs_star, sol_star, s_history, F_history, curves = shooting_newton(
        s0=s0,
        tol=1e-8,
        max_iter=20,
        n=2000
    )

    print("\nНайденный пристрелочный параметр:")
    print(f"s* = y'(0) = {s_star:.12f}")

    print(f"Проверка: y(1) = {sol_star[-1][0]:.12f}")

    print("\nИстория изменения пристрелочного параметра:")
    for k, s in enumerate(s_history):
        print(f"iter {k:2d}: s = {s:.12f}, F = {F_history[k]:.12e}")

    plot_results(xs_star, sol_star, s_history, F_history, curves)