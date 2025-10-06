import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pathlib import Path

x = sp.symbols('x')
functions = [
    ('sin(x^2)', sp.sin(x ** 2)),
    ('cos(sin(x))', sp.cos(sp.sin(x))),
    ('exp(sin(cos(x)))', sp.exp(sp.sin(sp.cos(x)))),
    ('ln(x+3)', sp.log(x + 3)),
    ('sqrt(x+3)', sp.sqrt(x + 3)),
]

def get_funcs(expr):
    f = sp.lambdify(x, expr, 'numpy')
    fp = sp.lambdify(x, sp.diff(expr, x), 'numpy')
    return f, fp

x0 = 1.3

def diff_1(f, x0, h):
    return (f(x0 + h) - f(x0)) / h

def diff_2(f, x0, h):
    return (f(x0) - f(x0 - h)) / h

def diff_3(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

def diff_4(f, x0, h):
    return (4 / 3) * (f(x0 + h) - f(x0 - h)) / (2 * h) - (1 / 3) * (f(x0 + 2 * h) - f(x0 - 2 * h)) / (4 * h)

def diff_5(f, x0, h):
    return (3 / 2) * (f(x0 + h) - f(x0 - h)) / (2 * h) \
         - (3 / 5) * (f(x0 + 2 * h) - f(x0 - 2 * h)) / (4 * h) \
         + (1 / 10) * (f(x0 + 3 * h) - f(x0 - 3 * h)) / (6 * h)

methods = [diff_1, diff_2, diff_3, diff_4, diff_5]

def get_pictures_dir():
    current_path = Path(__file__).resolve().parent
    pictures = current_path / 'pictures'
    pictures.mkdir(exist_ok=True, parents=True)
    return pictures

def plot_one_graphic(name, expr, pictures):
    labels = ['diff_1', 'diff_2', 'diff_3', 'diff_4', 'diff_5']
    
    f, df = get_funcs(expr)
    h_steps = np.logspace(-20, 0, 1500, base=2.0) 
    true = float(df(x0))

    plt.figure(figsize=(10, 6))
    plt.title(f"${sp.latex(expr)}$ at x0 = {x0}")
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('h')
    plt.ylabel('abs_error')
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)

    for label, method in zip(labels, methods):
        errors = [abs(float(method(f, x0, h)) - true) for h in h_steps]
        plt.plot(h_steps, errors, label=label, linewidth=1.3)

    plt.legend()
    plt.tight_layout()
    outfile = pictures / (name.replace('/', '_') + '.png')
    plt.savefig(outfile, dpi=200)
    plt.close()

pictures = get_pictures_dir()
for name, expr in functions:
    plot_one_graphic(name, expr, pictures)
