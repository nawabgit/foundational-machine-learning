import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure


def generate_gaussian_data(n, mean, cov):
    points = np.random.multivariate_normal(mean, cov, n).T
    return points


def calculate_projected_y(w, points):
    return np.dot(w, points)


def calculate_fisher_score(mu_a, mu_b, s_a, s_b, n_a, n_b):
    return (mu_a - mu_b)**2 / ((n_a / (n_a + n_b))*(s_a**2) + (n_b / (n_a + n_b))*(s_b**2))


figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')


def run_project_gaussian():
    # Generate first gaussian
    points1 = generate_gaussian_data(1000, [2, 2], [[3, 2], [2, 3]])
    #plt.plot(points1[0], points1[1], 'rx')

    # Generate second gaussian
    points2 = generate_gaussian_data(1000, [7, 7], [[3, 2], [2, 3]])
    #plt.plot(points2[0], points2[1], 'bx')

    # Project to lower dimension
    w = [-1.0173001,  -0.98239529]
    projected1 = calculate_projected_y(w, points1)
    projected2 = calculate_projected_y(w, points2)

    projected = np.concatenate((projected1, projected2))
    plt.hist(projected, range=(-30, 30), bins=50, color="r")
    #plt.hist(projected1, range=(-30, 30), bins=200, color="r")
    #plt.hist(projected2, range=(-10, 30), bins=200, color="r")
    plt.show()


def run_compute_best_fisher(n):
    # Generate first gaussian
    points1 = generate_gaussian_data(n, [2, 2], [[3, 2],
                                                 [2, 3]])

    # Generate second gaussian
    points2 = generate_gaussian_data(n, [7, 7], [[3, 2],
                                                 [2, 3]])

    w = np.array((1, 1)).T
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

        score = calculate_fisher_score(np.mean(p1), np.mean(p2), np.std(p1), np.std(p2), n, n)
        scores.append(score)

        if score > best:
            best = score
            best_w = dir_w

    plt.plot(range(1, 360), scores)
    plt.xlabel("Direction vector (θ)")
    plt.ylabel("Fisher Score")
    plt.show()
    print(best_w)

run_project_gaussian()
run_compute_best_fisher(1000)