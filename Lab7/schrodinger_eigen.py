import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shoot import rk_step
from quasilinearization import thomas_algorithm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICTURES_DIR = os.path.join(BASE_DIR, "pictures")


def sech(x):
    return 1.0 / math.cosh(x)


def potential(x):
    """Потенциальная яма U(x) = -2 sech^2(x)."""
    s = sech(x)
    return -2.0 * s * s


def make_schrodinger_extended_system(lam):
    """
    Расширенная система для метода стрельбы.

    y'' + (lambda - U(x)) y = 0,
    z = dy/dlambda,
    z'' + (lambda - U(x)) z = -y.
    """

    def system(x, Y):
        y, yp, z, zp = Y
        c = lam - potential(x)
        return [
            yp,
            -c * y,
            zp,
            -c * z - y,
        ]

    return system


def integrate_shooting(lam, L=8.0, n=4000, slope=1.0):
    """
    Интегрирует задачу Коши от -L до L:
    y(-L)=0, y'(-L)=slope.

    Амплитуда в задаче на собственные значения произвольна, поэтому slope=1.
    """
    a = -L
    b = L
    h = (b - a) / n

    x = a
    Y = [0.0, slope, 0.0, 0.0]
    xs = [x]
    sol = [Y[:]]

    system = make_schrodinger_extended_system(lam)
    for _ in range(n):
        Y = rk_step(system, x, Y, h)
        x += h
        xs.append(x)
        sol.append(Y[:])

    return xs, sol


def shooting_residual(lam, L=8.0, n=4000):
    """F(lambda) = y(L; lambda). Собственное значение дает F(lambda)=0."""
    _, sol = integrate_shooting(lam, L=L, n=n)
    return sol[-1][0]


def find_shooting_brackets(lambda_min=-2.0, lambda_max=-1e-6, samples=250, L=8.0, n=4000):
    """Ищет интервалы перемены знака F(lambda) на отрицательной части спектра."""
    if samples < 2:
        raise ValueError("samples должно быть не меньше 2")

    step = (lambda_max - lambda_min) / (samples - 1)
    grid = [lambda_min + i * step for i in range(samples)]
    values = [shooting_residual(lam, L=L, n=n) for lam in grid]

    brackets = []
    for i in range(1, samples):
        left = grid[i - 1]
        right = grid[i]
        f_left = values[i - 1]
        f_right = values[i]

        if f_left == 0.0:
            brackets.append((left, left))
        elif f_left * f_right < 0.0:
            brackets.append((left, right))

    return brackets, grid, values


def shooting_newton_eigenvalue(bracket, L=8.0, n=4000, tol=1e-10, max_iter=25):
    """
    Находит корень F(lambda)=0 методом Ньютона по расширенной системе.

    Используется страховка интервалом: если шаг Ньютона выходит за текущую скобку,
    берется середина интервала. Так метод остается устойчивым.
    """
    left, right = bracket
    if left == right:
        lam = left
        xs, sol = integrate_shooting(lam, L=L, n=n)
        return lam, xs, sol, [(0, lam, sol[-1][0], sol[-1][2], left, right)]

    f_left = shooting_residual(left, L=L, n=n)
    f_right = shooting_residual(right, L=L, n=n)
    if f_left * f_right > 0.0:
        raise ValueError("В bracket нет перемены знака F(lambda).")

    lam = 0.5 * (left + right)
    lambda_iterations = []

    for k in range(max_iter):
        xs, sol = integrate_shooting(lam, L=L, n=n)
        F = sol[-1][0]
        dF = sol[-1][2]
        lambda_iterations.append((k, lam, F, dF, left, right))

        if abs(F) < tol or abs(right - left) < tol:
            return lam, xs, sol, lambda_iterations

        if f_left * F <= 0.0:
            right = lam
            f_right = F
        else:
            left = lam
            f_left = F

        if abs(dF) > 1e-14:
            candidate = lam - F / dF
        else:
            candidate = 0.5 * (left + right)

        if not (left < candidate < right) or not math.isfinite(candidate):
            candidate = 0.5 * (left + right)

        lam = candidate

    raise RuntimeError("Метод стрельбы для lambda не сошелся за max_iter итераций.")


def normalize_on_grid(xs, ys):
    """Нормирует y так, чтобы int y^2 dx = 1, и фиксирует знак y(0)>0."""
    if len(xs) != len(ys):
        raise ValueError("xs и ys должны быть одной длины")

    integral = 0.0
    for i in range(len(xs) - 1):
        h = xs[i + 1] - xs[i]
        integral += 0.5 * h * (ys[i] * ys[i] + ys[i + 1] * ys[i + 1])

    norm = math.sqrt(integral) if integral > 0.0 else 1.0
    yn = [y / norm for y in ys]

    mid = len(yn) // 2
    if yn[mid] < 0.0:
        yn = [-y for y in yn]

    return yn


def shooting_eigenpairs(lambda_min=-2.0, lambda_max=-1e-6, samples=250, L=8.0, n=4000, tol=1e-10):
    brackets, grid, values = find_shooting_brackets(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        samples=samples,
        L=L,
        n=n,
    )

    eigenpairs = []
    for bracket in brackets:
        lam, xs, sol, lambda_iterations = shooting_newton_eigenvalue(bracket, L=L, n=n, tol=tol)
        ys = [row[0] for row in sol]
        ys = normalize_on_grid(xs, ys)
        eigenpairs.append(
            {
                "lambda": lam,
                "xs": xs,
                "ys": ys,
                "lambda_iterations": lambda_iterations,
                "bracket": bracket,
            }
        )

    return eigenpairs, grid, values


def build_tridiagonal_hamiltonian(L=8.0, n=800):
    """
    Строит трехдиагональную конечно-разностную матрицу оператора
    H = -d^2/dx^2 + U(x) на (-L, L), y(-L)=y(L)=0.
    """
    h = (2.0 * L) / n
    xs = [-L + i * h for i in range(n + 1)]
    inner_xs = xs[1:-1]
    m = len(inner_xs)

    off = -1.0 / (h * h)
    lower = [0.0] + [off] * (m - 1)
    diag = [(2.0 / (h * h)) + potential(x) for x in inner_xs]
    upper = [off] * (m - 1) + [0.0]

    return xs, inner_xs, lower, diag, upper, h


def sturm_count_less_than(mu, lower, diag):
    """Количество собственных значений трехдиагональной матрицы, меньших mu."""
    if not diag:
        return 0

    eps = 1e-30
    count = 0
    q = diag[0] - mu
    if q < 0.0:
        count += 1

    for i in range(1, len(diag)):
        if abs(q) < eps:
            q = eps if q >= 0.0 else -eps
        # lower[i] = элемент под диагональю в i-й строке
        q = diag[i] - mu - (lower[i] * lower[i]) / q
        if q < 0.0:
            count += 1

    return count


def sturm_bisection_eigenvalue(k, lower, diag, left=-3.0, right=0.0, tol=1e-12, max_iter=100):
    """
    Находит k-е собственное значение, считая от меньшего к большему.
    Нужно, чтобы count(left) < k <= count(right).
    """
    if sturm_count_less_than(left, lower, diag) >= k:
        raise ValueError("Левая граница слишком велика для sturm_bisection_eigenvalue.")
    if sturm_count_less_than(right, lower, diag) < k:
        raise ValueError("Правая граница слишком мала для sturm_bisection_eigenvalue.")

    a = left
    b = right
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if sturm_count_less_than(mid, lower, diag) >= k:
            b = mid
        else:
            a = mid
        if abs(b - a) < tol:
            break

    return 0.5 * (a + b)


def dot(u, v, h):
    return h * sum(ui * vi for ui, vi in zip(u, v))


def norm(u, h):
    value = dot(u, u, h)
    return math.sqrt(value) if value > 0.0 else 0.0


def apply_tridiagonal(lower, diag, upper, v):
    result = [0.0] * len(v)
    for i in range(len(v)):
        result[i] += diag[i] * v[i]
        if i > 0:
            result[i] += lower[i] * v[i - 1]
        if i < len(v) - 1:
            result[i] += upper[i] * v[i + 1]
    return result


def inverse_iteration(lower, diag, upper, inner_xs, h, shift, tol=1e-12, max_iter=50):
    """
    Обратная итерация для собственной функции.
    На каждом шаге решается трехдиагональная СЛАУ методом прогонки.
    """
    v = [sech(x) for x in inner_xs]
    v_norm = norm(v, h)
    v = [vi / v_norm for vi in v]

    inverse_iterations = []
    lambda_old = None

    for k in range(1, max_iter + 1):
        shifted_diag = [d - shift for d in diag]
        z = thomas_algorithm(lower[:], shifted_diag, upper[:], v[:])
        z_norm = norm(z, h)
        if z_norm == 0.0:
            raise RuntimeError("Нулевая норма в обратной итерации.")
        z = [zi / z_norm for zi in z]

        # Чтобы график не переворачивался по знаку между итерациями.
        if dot(z, v, h) < 0.0:
            z = [-zi for zi in z]

        Az = apply_tridiagonal(lower, diag, upper, z)
        lam = dot(z, Az, h) / dot(z, z, h)
        diff = max(abs(z[i] - v[i]) for i in range(len(v)))
        inverse_iterations.append((k, lam, diff))

        if lambda_old is not None and abs(lam - lambda_old) < tol and diff < math.sqrt(tol):
            return lam, z, inverse_iterations

        v = z
        lambda_old = lam

    return lambda_old, v, inverse_iterations


def finite_difference_eigenpairs(L=8.0, n=800, tol=1e-12):
    xs, inner_xs, lower, diag, upper, h = build_tridiagonal_hamiltonian(L=L, n=n)

    negative_count = sturm_count_less_than(0.0, lower, diag)
    eigenpairs = []

    for k in range(1, negative_count + 1):
        lam_sturm = sturm_bisection_eigenvalue(k, lower, diag, left=-3.0, right=0.0, tol=tol)

        # Сдвиг берем рядом с найденным значением, но не ровно в нем, чтобы СЛАУ не была вырожденной.
        shift = lam_sturm + 0.05
        if shift >= 0.0:
            shift = lam_sturm - 0.05

        lam, y_inner, inverse_iterations = inverse_iteration(
            lower=lower,
            diag=diag,
            upper=upper,
            inner_xs=inner_xs,
            h=h,
            shift=shift,
            tol=tol,
        )

        ys = [0.0] + y_inner + [0.0]
        ys = normalize_on_grid(xs, ys)
        eigenpairs.append(
            {
                "lambda_sturm": lam_sturm,
                "lambda": lam,
                "xs": xs,
                "ys": ys,
                "inverse_iterations": inverse_iterations,
                "shift": shift,
            }
        )

    return eigenpairs, negative_count


def ensure_pictures_dir(pictures_dir):
    os.makedirs(pictures_dir, exist_ok=True)


def plot_results(shooting_pairs, scan_grid, scan_values, fd_pairs, pictures_dir=PICTURES_DIR):
    ensure_pictures_dir(pictures_dir)

    plt.figure(figsize=(8, 5))
    plt.plot(scan_grid, scan_values, linewidth=1.5, label="F(lambda)=y(L)")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    for pair in shooting_pairs:
        plt.axvline(pair["lambda"], linestyle=":", linewidth=1.5, label=f"lambda={pair['lambda']:.8f}")
    plt.xlabel("lambda")
    plt.ylabel("F(lambda)")
    plt.title("Метод стрельбы: поиск собственных значений")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, "07_eigen_shooting_residual.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for i, pair in enumerate(shooting_pairs, start=1):
        plt.plot(pair["xs"], pair["ys"], linewidth=2, label=f"стрельба, lambda_{i}={pair['lambda']:.8f}")
    for i, pair in enumerate(fd_pairs, start=1):
        plt.plot(pair["xs"], pair["ys"], linestyle="--", linewidth=1.5, label=f"разности+прогонка, lambda_{i}={pair['lambda']:.8f}")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Собственные функции для U(x)=-2 sech^2(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, "08_eigen_functions.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for i, pair in enumerate(shooting_pairs, start=1):
        density = [y * y for y in pair["ys"]]
        plt.plot(pair["xs"], density, linewidth=2, label=f"|y_{i}(x)|^2")
    plt.xlabel("x")
    plt.ylabel("rho(x)=|y(x)|^2")
    plt.title("Функция распределения / плотность вероятности")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, "09_eigen_density.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Графики истории итераций не сохраняются в финальной версии отчета.
    # Для защиты достаточно трех обязательных рисунков: невязка F(lambda),
    # собственные функции y(x) и плотность распределения |y(x)|^2.
    # Итерации lambda остаются только в текстовом выводе программы.


def build_parser():
    parser = argparse.ArgumentParser(description="Задача XI.9.14: собственные значения для U(x)=-2 sech^2(x).")
    parser.add_argument("--L", type=float, default=8.0, help="Замена бесконечного интервала на [-L, L].")
    parser.add_argument("--shoot-n", type=int, default=4000, help="Число шагов RK4 в методе стрельбы.")
    parser.add_argument("--fd-n", type=int, default=800, help="Число разбиений для конечно-разностной схемы.")
    parser.add_argument("--samples", type=int, default=250, help="Число точек сканирования lambda для стрельбы.")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--pictures-dir", default=PICTURES_DIR)
    return parser


def main():
    args = build_parser().parse_args()

    shooting_pairs, scan_grid, scan_values = shooting_eigenpairs(
        lambda_min=-2.0,
        lambda_max=-1e-6,
        samples=args.samples,
        L=args.L,
        n=args.shoot_n,
        tol=args.tol,
    )

    fd_pairs, negative_count = finite_difference_eigenpairs(
        L=args.L,
        n=args.fd_n,
        tol=args.tol,
    )

    print("\nЗадача XI.9.14: U(x) = -2 sech^2(x)")
    print(f"Бесконечный интервал заменен на [-{args.L}, {args.L}].")
    print(f"Количество отрицательных уровней по методу Штурма для разностной матрицы: {negative_count}")

    print("\nМетод стрельбы с расширенной системой:")
    for i, pair in enumerate(shooting_pairs, start=1):
        print(f"lambda_{i} = {pair['lambda']:.12f}")
        print("История изменения lambda:")
        for k, lam, F, dF, left, right in pair["lambda_iterations"]:
            print(f"  iter={k:2d}, lambda={lam:.12f}, F={F:.6e}, F'={dF:.6e}")

    print("\nКонечно-разностная схема + метод прогонки + обратная итерация:")
    for i, pair in enumerate(fd_pairs, start=1):
        print(f"lambda_{i} = {pair['lambda']:.12f}  (Штурм: {pair['lambda_sturm']:.12f})")
        print(f"Количество итераций обратной итерации: {len(pair['inverse_iterations'])}")

    plot_results(shooting_pairs, scan_grid, scan_values, fd_pairs, args.pictures_dir)
    print(f"\nГрафики сохранены в: {args.pictures_dir}")


if __name__ == "__main__":
    main()
