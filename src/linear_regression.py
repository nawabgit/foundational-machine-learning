import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import autograd.numpy as anp
from autograd import grad
from sklearn.model_selection import train_test_split
import math

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

    return A, W, polynomial_locs


def create_gaussian_model(x_training, y_training, p, lam):
    # Generate design matrix A for gaussian radial basis
    # Centroids are p equally spaced between 0 and 1 (excluding 0 and 1)
    centroid_locs = np.linspace(0, 1, p+2)[1:-1]
    A = make_design(x_training, gaussian_basis_fn, centroid_locs)

    # Calculate W and flatten
    W = calculate_weight_vector(A, lam, p, y_training)

    return A, W, centroid_locs


def calculate_model_loss(y_true, y_predicted, W, lam):
    return (np.sum(np.square(y_true - y_predicted)) / len(y_true)) + (lam * np.sum(np.square(W)))


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


def run_regression_training_split_p(test_size=0.33, n=30, p=10, lam=0):
    # Set random seed for sanity
    np.random.seed(0)
    # Generate n random points from 0 - 1
    rn = np.random.uniform(0, 1, n)
    x_data = np.reshape(np.sort(rn, axis=0), (n, 1))
    # Calculate y values and add noise
    noise = np.reshape(np.random.normal(0, 0.1, n), (n, 1))
    y_data = four_pi_sin(x_data) + noise

    # Split data into training and test sets
    x_training, x_test, y_training, y_test = train_test_split(x_data, y_data, test_size=test_size)

    for basis_fn in [polynomial_basis_fn, gaussian_basis_fn]:
        loss_values = []
        for i in range(p+1):
            # Generate the model for this p
            if basis_fn == polynomial_basis_fn:
                # Return weights for polynomial model
                A, W, locs = create_polynomial_model(x_training, y_training, i, lam)
            else:
                # Return weights for gaussian model
                A, W, locs = create_gaussian_model(x_training, y_training, i, lam)

            # Calculate predicted test data values
            A_test = make_design(x_test, basis_fn, locs)
            y_predicted = np.dot(A_test, W)

            loss_values.append(np.log(calculate_model_loss(y_test, y_predicted, W, lam)))

        from matplotlib.pyplot import figure
        figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
        x_values = range(p+1)
        plt.plot(x_values, loss_values)
        plt.xticks(x_values[0::2])

        plt.xlabel("Number of basis functions (p)")
        plt.ylabel("Log mean squared loss")

        plt.show()


def run_regression_training_split_lambda(test_size=0.33, n=30, p=10):
    # Set random seed for sanity
    np.random.seed(0)
    # Generate n random points from 0 - 1
    rn = np.random.uniform(0, 1, n)
    x_data = np.reshape(np.sort(rn, axis=0), (n, 1))
    # Calculate y values and add noise
    noise = np.reshape(np.random.normal(0, 0.1, n), (n, 1))
    y_data = four_pi_sin(x_data) + noise

    # Split data into training and test sets
    x_training, x_test, y_training, y_test = train_test_split(x_data, y_data, test_size=test_size)

    lambdas = [math.exp(x) for x in range(-30, -5)]
    for basis_fn in [polynomial_basis_fn, gaussian_basis_fn]:
        loss_values = []
        for lam in lambdas:
            # Generate the model for this p
            if basis_fn == polynomial_basis_fn:
                # Return weights for polynomial model
                A, W, locs = create_polynomial_model(x_training, y_training, p, lam)
            else:
                # Return weights for gaussian model
                A, W, locs = create_gaussian_model(x_training, y_training, p, lam)

            # Calculate predicted test data values
            A_test = make_design(x_test, basis_fn, locs)
            y_predicted = np.dot(A_test, W)

            loss_values.append(np.log(calculate_model_loss(y_test, y_predicted, W, lam)))

        from matplotlib.pyplot import figure
        figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
        x_values = range(-30, -5)
        plt.plot(x_values, loss_values)
        plt.xticks(x_values[0::2])

        plt.xlabel("Log strength of regularisation (λ)")
        plt.ylabel("Log mean squared loss")

        plt.show()

# run_linear_regression(n=25, p=10)
run_regression_training_split_lambda(n=30, p=9)
