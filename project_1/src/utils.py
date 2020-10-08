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
    # plt.rc("xtick", labelsize="x-small")
    # plt.rc("ytick", labelsize="x-small")
    return


def polynomial_form(n):
    """Generates a string representation of n-th degree polynomial in the same
        form as generated by the design_matrix_2D function.
    Args:
        n (Int): Degree of polynomial to generate
    Returns:
        (List): String containing each term in the n-th degree polynomial
    """

    terms = []

    terms.append("c")
    for i in range(1, n + 1):
        q = int(i * (i + 1) / 2)
        for k in range(i + 1):
            terms.append(f"$x^{i-k} y^{k}$")
    return terms


def polynomial_no_terms(n):
    """Computes the number of terms in an n-th degree 2d polynomial
    Args:
        N (Int): Degree of polynomial
    """
    return int((n + 1) * (n + 2) / 2)


if __name__ == "__main__":
    print(polynomial_form(5))
