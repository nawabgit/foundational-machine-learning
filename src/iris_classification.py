from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

data = datasets.load_iris()

setosa = data['data'][:50]
versicolor = data['data'][50:100]
virginica = data['data'][100:150]
classes = [setosa, versicolor, virginica]

# Calculate within class covariance sum
S_W = np.zeros((4, 4))
for c in classes:
    S_W = np.add(S_W, np.cov(c, rowvar=False))

# Calculate between-class mean sum
mu = np.mean(data['data'], axis=0).reshape(4, 1)
S_B = np.zeros((4, 4))
for c in classes:
    mu_c = np.mean(c, axis=0).reshape(4, 1)
    diff = mu_c - mu
    S_B = np.add(S_B, diff.dot(diff.T))


#https://sebastianraschka.com/Articles/2014_python_lda.html
# Calculate Eigenvalues and vectors S_W^-1 dot S_B
# Use eig instead of eigh since not symmetric
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eig_vecs = eig_vecs.T
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

# Check the generalised equation holds true
for i in range(len(eig_vecs)):
    w = eig_vecs[i]
    lam = eig_vals[i]
    print(np.dot(S_B, w) - (lam * np.dot(S_W, w)))

#https://sebastianraschka.com/Articles/2014_python_lda.html
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0], i[1])

w = eig_pairs[0][1]

classes = [setosa, versicolor, virginica]
projected1 = np.dot(setosa, w)
projected2 = np.dot(versicolor, w)
projected3 = np.dot(virginica, w)
plt.hist(projected1, range=(-2, 2), bins="auto", alpha=0.5, color="r", label='setosa')
plt.hist(projected2, range=(-2, 2), bins="auto", alpha=0.5, color="g", label='versicolor')
plt.hist(projected3, range=(-2, 2), bins="auto", alpha=0.5, color="b", label='virginica')
plt.legend(loc='upper right')
plt.show()
import pdb;pdb.set_trace()