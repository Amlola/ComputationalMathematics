import os
import matplotlib.pyplot as plt


from newton_interpolation import get_newton_coeffs, newton     
from spline_interpolation import get_cubic_spline, spline        
from mls import get_mls_coeffs, mls    


years = [1910, 1920, 1930, 1940, 1950,
         1960, 1970, 1980, 1990, 2000, 2010]


population = [84528, 92855, 106360, 120264, 141042,
              173855, 204042, 226948, 253785, 279049, 317630]


def get_pictures_folder():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pictures_dir = os.path.join(script_dir, "pictures")

    if not os.path.exists(pictures_dir):
        os.makedirs(pictures_dir)

    return pictures_dir


def save_plot(fig, filename):
    pictures_dir = get_pictures_folder()
    full_path = os.path.join(pictures_dir, filename)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    ref_year = 2020

    years_dense = list(range(years[0], ref_year + 1))

    newton_coeffs = get_newton_coeffs(years, population)
    newton_values = [newton(years, newton_coeffs, x) for x in years_dense]

    x_nodes, y_nodes, second_derivatives = get_cubic_spline(years, population)
    spline_values = [spline(x_nodes, y_nodes, second_derivatives, x) for x in years_dense]

    mls_coeffs = get_mls_coeffs(years, population)
    mls_values = [mls(mls_coeffs, x) for x in years_dense]

    fig_n, ax_n = plt.subplots(figsize=(10, 6))
    ax_n.scatter(years, population)
    ax_n.plot(years_dense, newton_values)
    ax_n.set_title("Метод Ньютона")
    ax_n.grid(True)
    save_plot(fig_n, "newton.png")

    fig_s, ax_s = plt.subplots(figsize=(10, 6))
    ax_s.scatter(years, population)
    ax_s.plot(years_dense, spline_values)
    ax_s.set_title("Cплайн")
    ax_s.grid(True)
    save_plot(fig_s, "spline.png")

    fig_m, ax_m = plt.subplots(figsize=(10, 6))
    ax_m.scatter(years, population)
    ax_m.plot(years_dense, mls_values)
    ax_m.set_title("МНК")
    ax_m.grid(True)
    save_plot(fig_m, "mls.png")
