years = [1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]


population = [84528, 92855, 106360, 120264, 141042, 173855, 204042, 226948, 253785, 279049, 317630]


def get_newton_coeffs(years, population):
    coeffs = population.copy() 

    for j in range(1, len(years)):
        for i in range(len(years) - 1, j - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (years[i] - years[i - j])

    return coeffs


def newton(years, coeffs, predict_year):

    result = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        result = result * (predict_year - years[i]) + coeffs[i]
    return result


if __name__ == "__main__":
    coeff = get_newton_coeffs(years, population)

    predict_year = 2020

    print(f"{predict_year}: {newton(years, coeff, predict_year)}")
