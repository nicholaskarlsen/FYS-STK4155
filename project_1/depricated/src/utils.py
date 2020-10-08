import matplotlib.pyplot as plt


def plot_settings():
    """ Changes matplot font to something more LaTeX friendly. """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        #"font.sans-serif": ["Helvetica"]
    })
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    return


def polynomial_form(n):
    """
    Generates a string representation of n-th degree polynomial in the same
    form as generated by the design_matrix_2D function.
    """

    print("c_0 : 1")
    for i in range(1, n+1):
        q = int(i * (i + 1) / 2)
        for k in range(i + 1):
            print(f"c_{q + k} : x^{i-k} * y^{k}")
    return

if __name__ == '__main__':
    polynomial_form(2)
