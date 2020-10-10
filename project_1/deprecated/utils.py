import matplotlib.pyplot as plt


def plot_settings():
    """ Changes matplot font to something more LaTeX friendly. """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            # "font.sans-serif": ["Helvetica"]
        }
    )
    plt.rc("xtick", labelsize="x-small")
    plt.rc("ytick", labelsize="x-small")
    return


def polynomial_form(n):
    """
    Generates a string representation of n-th degree polynomial in the same
    form as generated by the design_matrix_2D function.
    """

    out = []

    out.append("$\\beta_0$")
    for i in range(1, n + 1):
        q = int(i * (i + 1) / 2)
        for k in range(i + 1):
            out.append((f"$\\beta_{q + k} x^{i-k} y^{k}$"))
    return out


def evaluate_poly(x, y, beta, n):
    """Evaluates a polynomial constructed by OLS_2D at (x,y) given an array containing betas
    Args:
        x (float: x-coordinate to evaluate polynomial (Can be array)
        y (float): y-coordinate to evaluate polynomial (Can be array)
        betas (Array): Free parameters in polynomial.
        n (int): degree of polynomial

    Returns:
        Float (or Array, depending on input): Polynomial evaluated at (x, y)
    """

    z = np.zeros(x.shape)
    z += beta[0]
    for i in range(1, n + 1):
        q = int(i * (i + 1) / 2)  # Number of terms with degree < i
        for k in range(i + 1):
            z += beta[q + k] * x ** (i - k) * y ** k

    return z


if __name__ == "__main__":
    res = polynomial_form(1)

    for r in res:
        print(r)