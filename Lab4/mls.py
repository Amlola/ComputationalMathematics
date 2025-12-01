import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Lab2"))
from gauss import solve_gauss


years = [1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]


population = [84528, 92855, 106360, 120264, 141042, 173855, 204042, 226948, 253785, 279049, 317630]


def get_mls_coeffs(years, population):

    max_power_sum = 5
    num_xy_sums = 3

    left_sum = [0.0] * max_power_sum
    right_sum = [0.0] * num_xy_sums

    for x, y in zip(years, population):
        x_power = 1.0
        for i in range(max_power_sum):
            left_sum[i] += x_power
            x_power *= x

        x_power = 1.0
        for i in range(num_xy_sums):
            right_sum[i] += x_power * y
            x_power *= x

    A = [[0.0] * (num_xy_sums) for _ in range(3)]
    b = [0.0] * num_xy_sums

    for i in range(num_xy_sums):
        for j in range(num_xy_sums):
            A[i][j] = left_sum[i + j]
        b[i] = right_sum[i]         

    A_copy = [row[:] for row in A]
    b_copy = b[:]

    coeffs = solve_gauss(A_copy, b_copy)
    return coeffs


def mls(coeffs, predict_year):
    result = 0.0
    x_power = 1.0

    for coeff in coeffs:    
        result += coeff * x_power
        x_power *= predict_year
    return result


if __name__ == "__main__":

    coeffs = get_mls_coeffs(years, population)

    predict_year = 2020

    print(f"{predict_year}: {mls(coeffs, predict_year)}")
