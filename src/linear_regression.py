import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import autograd.numpy as anp
from autograd import grad


def four_pi_sin(xin):
    return anp.array(anp.sin(4 * np.pi * xin))


def run_linear_regression(n=15):
    """ Executes Linear Regression

    :param n: The number of (x_n, t_n) pairs in the training set

    """
    # Set random seed for sanity
    np.random.seed(11)

    # Generate n random points from 0 - 1
    rn = np.random.uniform(0, 1, n)
    x_training = np.reshape(np.sort(rn, axis=0), (n, 1))

    # Calculate y values and add noise
    noise = np.reshape(np.random.normal(0, 0.1, n), (n, 1))
    y_training = four_pi_sin(x_training) + noise

    plt.plot(x_training, y_training)
    plt.show()


run_linear_regression()
