import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import autograd.numpy as anp
from autograd import grad


def four_pi_sin(xin):
    return anp.array(anp.sin(4 * np.pi * xin))


def gaussian_basis_fn(x, mu, sigma=0.1):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def polynomial_basis_fn(x, degree):
    return x ** degree


def make_design(x, basisfn, basisfn_locs=None):
    if basisfn_locs is None:
        return np.concatenate([np.ones(x.shape), basisfn(x)], axis=1)
    else:
        return np.concatenate([np.ones(x.shape)] + \
            [basisfn(x, loc) for loc in basisfn_locs], axis=1)


def calculate_weight_vector(A, lam, p, Y):
    A_T = A.transpose()
    I = np.identity(p + 1)

    return np.dot(np.dot(np.linalg.inv(np.dot(A_T, A) + np.dot(lam, I)), A_T), Y)


def create_polynomial_model(x_training, y_training, p, lam):
    # Generate design matrix A for polynomial basis
    polynomial_locs = [n for n in range(1, p + 1)]
    A = make_design(x_training, polynomial_basis_fn, polynomial_locs)

    # Calculate W and flatten
    W = calculate_weight_vector(A, lam, p, y_training)
    W = [item for sublist in W for item in sublist]

    return A, W, polynomial_locs


def create_gaussian_model(x_training, y_training, p, lam):
    # Generate design matrix A for gaussian radial basis
    # Centroids are p equally spaced between 0 and 1 (excluding 0 and 1)
    centroid_locs = np.linspace(0, 1, p+2)[1:-1]
    A = make_design(x_training, gaussian_basis_fn, centroid_locs)

    # Calculate W and flatten
    W = calculate_weight_vector(A, lam, p, y_training)
    W = [item for sublist in W for item in sublist]

    return A, W, centroid_locs


def run_linear_regression(n=15, p=5, lam=0):
    """ Executes Linear Regression

    :param n: The number of (x_n, t_n) pairs in the training set

    """
    # Set random seed for sanity
    np.random.seed(0)

    # Generate n random points from 0 - 1
    rn = np.random.uniform(0, 1, n)
    x_training = np.reshape(np.sort(rn, axis=0), (n, 1))

    # Calculate y values and add noise
    noise = np.reshape(np.random.normal(0, 0.1, n), (n, 1))
    y_training = four_pi_sin(x_training) + noise


    # Return weights for polynomial model
    A, W, locs = create_gaussian_model(x_training, y_training, p, lam)

    # Create line of best fit
    z = np.reshape(np.sort(np.linspace(0, 1, 1000), axis=0), (1000, 1))
    A2 = make_design(z, gaussian_basis_fn, locs)
    y_vals2 = np.dot(A2, W)


    # Plot model and training data
    y_vals = np.dot(A, W)
    plt.plot(z, y_vals2, label="Best Fit")
    plt.scatter(x_training , y_training, c='r', marker='X', label="Training Data")
    plt.legend()
    plt.show()


run_linear_regression(n=25, p=10)
