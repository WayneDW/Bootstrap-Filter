import numpy.random as npr
import numpy as np
import scipy.stats as stats


class SV:
    """
    The model parameters and pdf:s for the Stochastic Volatility model.
    """
    def __init__(self, sigma=1., beta=.5, phi=0.91):
        # global stats
        self.sigma = sigma
        self.beta = beta
        self.phi = phi

    def generateData(self, T):
        x = np.zeros(T)
        y = np.zeros(T)

        x[0] = self.sigma / np.sqrt(1 - self.phi ** 2) * np.random.normal(size=1)
        scale = self.beta * np.exp(x[0]/2)
        y[0] = scale * np.random.normal(size=1)

        for t in range(1, T):
            x[t] = self.phi * x[t - 1] + self.sigma * np.random.normal(size=1)
            scale = self.beta * np.exp(x[t]/2)
            y[t] = scale * np.random.normal(size=1)

        return x.reshape((1, -1)), y

    def particle_0(self, N):
        return np.random.normal(0, self.sigma / np.sqrt(1 - self.phi ** 2), size=(N, 1))

    def propagate(self, x):
        x_next = self.phi * x.squeeze() + self.sigma * npr.normal(size=x.size)
        return x_next.reshape((-1, 1))

    def log_g(self, x, y):
        return stats.norm.logpdf(y, loc=0, scale=np.sqrt(self.beta**2 * np.exp(x))).squeeze()
