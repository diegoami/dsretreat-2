import numpy as np
from random import randint

class Perceptron(object):
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """
        Learns parameter vector self.theta from data X and binary labels y.

        y should have -1 for negative class and +1 for positive.
        X will be augmented with an intercept term before fitting.
        """
        assert X.shape[0] == y.shape[0]
        xs = X.shape[0]
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((X, intercept), axis=1)
        self.theta = np.random.randn(X.shape[1])
        while True:
            di = randint(0,xs-1)
            if y[di] * np.dot( self.theta.T,  X[di]) <= 0 :
                self.theta = self.theta + np.dot(y[di],X[di])
           # print([y[di].T*np.dot(self.theta.T, X[di]) for i in range(xs)])
            if all([y[di] *np.dot(self.theta.T, X[di]) for i in range(xs)]) > 0:
                break



if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import perceptron
    import sample_perceptron

    def plot_decision_boundary(w, b, xx):
        # solve w_x * x + w_y*y + b = 0, for y
        yy = (-b - w[0] * xx) / w[1]
        plt.plot(xx, yy, 'k--')

    np.random.seed(42)
    N = 10
    X = np.concatenate(([5, 6] + np.random.randn(N, 2), [-1, -1] + np.random.randn(N, 2)))
    y = np.concatenate((np.ones(N), -np.ones(N)))

    plt.figure(figsize=(12, 8))
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='r', alpha=1)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='b', alpha=1)

    for _ in range(30):
        p = perceptron.Perceptron()
        p.fit(X, y)
        plot_decision_boundary(p.theta[0:2], p.theta[-1], np.linspace(-10, 10))

        intercept = np.ones((X.shape[0], 1))
        XX = np.concatenate((X, intercept), axis=1)
        perceptron_accuracy = 100.0 * np.sum([np.sign(np.dot(p.theta.T, x)) == yy for x, yy in zip(XX, y)]) / X.shape[0]
       # assert (perceptron_accuracy == 100.0)

    plt.xlim([-5.0, 10])
    plt.ylim([-5, 11])
    plt.grid()

    plt.show()