import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from system_generator import build_lienard_system


AB_COEFFICIENTS = {
    1: np.array([1.0], dtype=float),
    2: np.array([3.0 / 2.0, -1.0 / 2.0], dtype=float),
    3: np.array([23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0], dtype=float),
    4: np.array([55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0], dtype=float),
}


def rk4_step(system_function, t_value, u_value, step):
    k1 = system_function(t_value, u_value)
    k2 = system_function(t_value + step / 2.0, u_value + step * k1 / 2.0)
    k3 = system_function(t_value + step / 2.0, u_value + step * k2 / 2.0)
    k4 = system_function(t_value + step, u_value + step * k3)
    return u_value + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def build_time_grid(t_start, t_end, step):
    if step <= 0:
        raise ValueError("Шаг интегрирования должен быть положительным")

    steps_count = int(round((t_end - t_start) / step))
    if steps_count <= 0:
        raise ValueError("Слишком короткий интервал интегрирования для выбранного шага")

    if not np.isclose(t_start + steps_count * step, t_end, rtol=1e-10, atol=1e-12):
        raise ValueError("Шаг должен точно делить длину интервала интегрирования")

    return np.linspace(t_start, t_end, steps_count + 1, dtype=float)


def adams_bashforth_method(system_function, initial_values, t_start, t_end, step, order):
    if order not in AB_COEFFICIENTS:
        raise ValueError("Поддерживаются только порядки 1, 2, 3 и 4")

    t_values = build_time_grid(t_start, t_end, step)
    solution = np.zeros((len(t_values), len(initial_values)), dtype=float)
    solution[0] = np.array(initial_values, dtype=float)

    startup_steps = min(order - 1, len(t_values) - 1)
    for index in range(startup_steps):
        solution[index + 1] = rk4_step(system_function, t_values[index], solution[index], step)

    coefficients = AB_COEFFICIENTS[order]
    for index in range(order - 1, len(t_values) - 1):
        increment = np.zeros_like(solution[index])
        for shift, coefficient in enumerate(coefficients):
            increment += coefficient * system_function(t_values[index - shift], solution[index - shift])
        solution[index + 1] = solution[index] + step * increment

    return t_values, solution


def build_plot(t_values, y_values, z_values, order, eps, step):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(t_values, y_values, label="y(t)")
    axes[0].plot(t_values, z_values, label="z(t)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("y, z")
    axes[0].set_title(f"Метод Адамса-Башфорта {order} порядка")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(y_values, z_values)
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("z")
    axes[1].set_title(f"Фазовая траектория, eps={eps:g}, h={step:g}")
    axes[1].grid(True)

    figure.tight_layout()
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Явные методы Адамса-Башфорта 1-4 порядка для уравнения Ван-дер-Поля в представлении Льенара"
    )
    parser.add_argument("order", type=int, choices=[1, 2, 3, 4], help="Порядок метода")
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Имя выходного PNG-файла в папке pictures (по умолчанию adams_methods_<order>.png)",
    )
    parser.add_argument("--step", type=float, default=0.5, help="Шаг интегрирования")
    parser.add_argument("--eps", type=float, default=1.0, help="Параметр eps")
    parser.add_argument("--t-start", type=float, default=0.0, help="Левая граница интервала")
    parser.add_argument("--t-end", type=float, default=100.0, help="Правая граница интервала")
    args = parser.parse_args()

    output_name = args.filename or f"adams_methods_{args.order}.png"
    if not output_name.lower().endswith(".png"):
        output_name = f"{output_name}.png"

    system_function = build_lienard_system(eps=args.eps)
    initial_values = np.array([2.0, 0.0], dtype=float)

    t_values, solution = adams_bashforth_method(
        system_function=system_function,
        initial_values=initial_values,
        t_start=args.t_start,
        t_end=args.t_end,
        step=args.step,
        order=args.order,
    )

    y_values = solution[:, 0]
    z_values = solution[:, 1]

    os.makedirs("pictures", exist_ok=True)
    output_path = os.path.join("pictures", output_name)

    figure = build_plot(t_values, y_values, z_values, args.order, args.eps, args.step)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {output_path}")