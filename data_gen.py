import numpy as np
import matplotlib.pyplot as plt


class LinSepBinaryData:

    @staticmethod
    def generate_linearly_separable(n=100, dim=2, seed=None):
        """
        Generate a linearly separable dataset
        """
        gen = np.random.default_rng(seed)
        weights = 2 * gen.random(dim) - 1
        bias = 2 * gen.random() - 1
        X = 10 * gen.random((n, dim)) - 5
        y = (X.dot(weights) + bias > 0).astype(int)
        return X, y.reshape(-1, 1)
    
    @staticmethod
    def plot_data2d(X, y, savepath=None):
        """
        Plot a linearly separable 2D dataset
        """
        fig, ax = plt.subplots(1, 1)
        ax.scatter(X[:, 0], X[:, 1], c=y, marker='x', cmap='bwr')
        if savepath:
            fig.savefig(savepath)
            fig.clear()
            return
        return fig


    @staticmethod
    def plot_data_and_separator2d(X, y, w, savepath=None):
        """
        Plot a linearly separable 2D dataset with the linear separator
        """
        fig, ax = plt.subplots(1, 1)
        ax.scatter(X[:, 0], X[:, 1], c=y, marker='x', cmap='bwr')
        x1min, x1max = np.min(X, axis=0)[0], np.max(X, axis=0)[0]
        x1s = np.array([x1min, x1max])
        x2s = -(w[0] * x1s + w[2])/(w[1] + 1e-12)
        ax.plot(x1s, x2s)

        if savepath:
            fig.savefig(savepath)
            fig.clear()
            return
        return fig
    
    @staticmethod
    def plot_data3d(X, y):
        """
        Plot a linearly separable 3D dataset
        """
        plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, marker='x', cmap='bwr')
        plt.show()
    
class LinRegressionData:

    @staticmethod
    def generate_regression(n=100, dim=1, sigma=1, seed=None):
        """
        Generate a regression dataset
        """
        gen = np.random.default_rng(seed)
        weights = 2 * gen.random(dim) - 1
        bias = 2 * gen.random() - 1
        X = 10 * gen.random((n, dim)) - 5
        y = X.dot(weights) + bias + sigma * gen.standard_normal(n)
        return X, y.reshape(-1, 1)
    
    @staticmethod
    def plot_data1d(X, y):
        """
        Plot a regression 1D dataset
        """
        plt.plot(X, y, 'x')
        plt.show()

    @staticmethod
    def plot_data_and_weight1d(X, y, w, savepath=None):
        fig, ax = plt.subplots(1, 1)
        ax.plot(X, y, 'x')
        xmin, xmax = np.min(X), np.max(X)
        xs = np.array([xmin, xmax]).reshape(1, -1)
        ys = w.T @ np.vstack((xs, [1, 1]))
        ax.plot(xs[0], ys[0])
        if savepath:
            fig.savefig(savepath)
            fig.clear()
            return
        return fig

def test_LinSepBinaryData():
    gen = LinSepBinaryData
    X, y = gen.generate_linearly_separable(n=100, dim=2, seed=42)
    gen.plot_data2d(X, y)

def test_LinRegressionData():
    gen = LinRegressionData
    X, y = gen.generate_regression(n=500, dim=1, seed=42)
    gen.plot_data1d(X, y)

if __name__ == '__main__':
    test_LinRegressionData()
    test_LinRegressionData()
