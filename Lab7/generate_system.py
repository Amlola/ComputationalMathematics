import math


def generate_system(eps=1e-10):
    def system(x, Y):
        y, yp, u1, u2 = Y

        y_safe = max(y, 0.0)
        sqrt_y = math.sqrt(y_safe)
        denom = max(sqrt_y, eps)

        dy_dx = yp
        dyp_dx = x * sqrt_y

        du1_dx = u2
        du2_dx = x * u1 / (2.0 * denom)

        return [dy_dx, dyp_dx, du1_dx, du2_dx]

    return system

def nonlinear_rhs(x, y, eps=1e-10):
    y_safe = max(y, 0.0)
    return x * math.sqrt(y_safe)


def nonlinear_rhs_dy(x, y, eps=1e-10):
    y_safe = max(y, 0.0)
    sqrt_y = math.sqrt(y_safe)
    denom = max(sqrt_y, eps)
    return x / (2.0 * denom)