import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import autograd.numpy as anp
from autograd import grad


def four_pi_sin(xin):
    return anp.array(anp.sin(4 * np.pi * xin))


X_plot = np.linspace(-1,1,200)
plt.plot(X_plot, four_pi_sin(X_plot))
plt.show()
