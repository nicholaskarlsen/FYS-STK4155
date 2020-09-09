import matplotlib.pyplot as plt


def plot_settings():
    """ Changes matplot font to something more LaTeX friendly. """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]
    })
    return