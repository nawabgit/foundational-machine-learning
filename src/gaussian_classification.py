import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import scipy.stats as sp
import matplotlib as mpl

np.random.seed(0)


def generate_gaussian_data(n, mean, cov):
    points = np.random.multivariate_normal(mean, cov, n).T
    return points


def calculate_projected_y(w, points):
    return np.dot(w, points)


def calculate_fisher_score(mu_a, mu_b, s_a, s_b, n_a, n_b):
    return (mu_a - mu_b)**2 / ((n_a / (n_a + n_b))*(s_a**2) + (n_b / (n_a + n_b))*(s_b**2))


def calculate_odds(point, mu, S, points1, points2):
    dist = sp.multivariate_normal(mu, S)
    p_c = len(points1) / len(points2)
    p_x_c = dist.pdf(point)

    return p_c * p_x_c


figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')


def run_project_gaussian(mu_a, mu_b, S_a, S_b, n_a, n_b):
    # Generate first gaussian
    points1 = generate_gaussian_data(n_a, mu_a, S_a)

    # Generate second gaussian
    points2 = generate_gaussian_data(n_b, mu_b, S_b)
    #plt.plot(points1[0], points1[1], 'rx')
    #plt.plot(points2[0], points2[1], 'bx')

    # Project to lower dimension
    w = [1, 1]
    projected1 = calculate_projected_y(w, points1)
    projected2 = calculate_projected_y(w, points2)

    projected = np.concatenate((projected1, projected2))
    #plt.hist(projected, range=(-30, 30), bins=50, color="r")
    plt.hist(projected1, range=(-10, 20), bins=60, color="r", alpha=0.5)
    plt.hist(projected2, range=(-10, 20), bins=60, color="b", alpha=0.5)
    plt.show()
    #print("mu a = " + str([np.mean(points1[0]), np.mean(points1[1])]))
    #print("mu b = " + str([np.mean(points2[0]), np.mean(points2[1])]))
    #print("cov a = " + str(np.cov(points1)))
    #print("cov b = " + str(np.cov(points2)))


def run_compute_best_fisher( mu_a, mu_b, S_a, S_b, n_a, n_b):
    # Generate Gaussians
    points1 = generate_gaussian_data(n_a, mu_a, S_a)
    points2 = generate_gaussian_data(n_b, mu_b, S_b)

    w = np.array((1, 0)).T
    scores = []
    # For every degree, calculate fisher score
    best = 0
    best_w = 0
    for degree in range(1, 360):
        # Generate rotation matrix
        theta = np.radians(degree)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        # Direction vector w
        dir_w = np.dot(R, w)

        p1 = calculate_projected_y(dir_w, points1)
        p2 = calculate_projected_y(dir_w, points2)

        score = calculate_fisher_score(np.mean(p1), np.mean(p2), np.std(p1), np.std(p2), n_a, n_b)
        scores.append(score)

        if score > best:
            best = score
            best_w = dir_w

    plt.plot(range(1, 360), scores)
    plt.xlabel("Direction vector (θ)")
    plt.ylabel("Fisher Score")
    plt.show()
    print("w* = " + np.array2string(best_w))


def run_decision_boundary(mu_a, mu_b, S_a, S_b, n_a, n_b):
    # Generate Gaussians
    points1 = generate_gaussian_data(n_a, mu_a, S_a)
    points2 = generate_gaussian_data(n_b, mu_b, S_b)

    @np.vectorize
    def calculate_log(x, y):
        point = np.array([x, y])
        odds = calculate_odds(point, mu_a, S_a, points1, points2) / calculate_odds(point, mu_b, S_b, points2, points1)
        logodds = np.log(odds)
        return logodds

    X, Y = np.mgrid[-10:20, -10:20]
    Z = calculate_log(X,Y)
    levels = [-300, 0, 300]

    # Generate gaussian contours
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X;
    pos[:, :, 1] = Y
    dist_a = sp.multivariate_normal(mu_a, S_a)
    dist_b = sp.multivariate_normal(mu_b, S_b)

    # Plot Gaussian contours
    CSa = plt.contour(X, Y, dist_a.pdf(pos), color='b', label="Class a")
    CSb = plt.contour(X, Y, dist_b.pdf(pos), color='r', label="Class b")
    plt.clabel(CSa, inline=1, fontsize=0)
    plt.clabel(CSb, inline=1, fontsize=0)
    cmap = mpl.colors.ListedColormap(['b', 'r'])

    # Plot generated data points
    plt.plot(points1[0], points1[1], 'rx', alpha=1, label="Class a")
    plt.plot(points2[0], points2[1], 'bx', alpha=1, label="Class b")

    # Plot decision boundary
    plt.contourf(X, Y, Z, levels, alpha=0.3, cmap=cmap)
    plt.legend()
    plt.show()

    mu_a = [np.mean(points1[0]), np.mean(points1[1])]
    mu_b = [np.mean(points2[0]), np.mean(points2[1])]
    S_a = np.cov(points1)
    S_b = np.cov(points2)
    print("mu_a = " + str([np.mean(points1[0]), np.mean(points1[1])]))
    print("mu_b = " + str([np.mean(points2[0]), np.mean(points2[1])]))
    print("S_a = " + str(np.cov(points1)))
    print("S_b = " + str(np.cov(points2)))

    @np.vectorize
    def calculate_log(x, y):
        point = np.array([x, y])
        odds = calculate_odds(point, mu_a, S_a, points1, points2) / calculate_odds(point, mu_b, S_b, points2, points1)
        logodds = np.log(odds)
        return logodds

    X, Y = np.mgrid[-15:20, -15:20]
    Z = calculate_log(X,Y)
    levels = [-300, 0, 300]

    # Generate gaussian contours
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X;
    pos[:, :, 1] = Y
    dist_a = sp.multivariate_normal(mu_a, S_a)
    dist_b = sp.multivariate_normal(mu_b, S_b)

    # Plot Gaussian contours
    CSa = plt.contour(X, Y, dist_a.pdf(pos), color='b', label="Class a")
    CSb = plt.contour(X, Y, dist_b.pdf(pos), color='r', label="Class b")
    plt.clabel(CSa, inline=1, fontsize=0)
    plt.clabel(CSb, inline=1, fontsize=0)
    cmap = mpl.colors.ListedColormap(['b', 'r'])

    # Plot generated data points
    plt.plot(points1[0], points1[1], 'rx', alpha=1, label="Class a")
    plt.plot(points2[0], points2[1], 'bx', alpha=1, label="Class b")

    # Plot decision boundary
    plt.contourf(X, Y, Z, levels, alpha=0.3, cmap=cmap)
    plt.legend()
    plt.show()


n_a, n_b = 1000, 1000

mu_a = [2, 2]
mu_b = [7, 7]
S_a = [[3, 1],
       [1, 3]]
S_b = [[3, 1],
       [1, 3]]

#run_project_gaussian(mu_a, mu_b, S_a, S_b, n_a, n_b)
#run_compute_best_fisher(mu_a, mu_b, S_a, S_b, n_a, n_b)

n_a, n_b = 10, 10
#run_decision_boundary(mu_a, mu_b, S_a, S_b, n_a, n_b)