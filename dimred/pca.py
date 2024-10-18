import numpy as np
import matplotlib.pyplot as plt
from functools import partial

class PCA:
    def __init__(self) -> None:
        pass
    def fit(self, X: np.ndarray, n_components: int):
        X = X - X.mean(axis=0)
        evecs = np.linalg.svd(X)[2] # svd returns them in decreasing order of eigenvalues
        self.evecs = evecs[:n_components].T
        return self.transform(X)
    def transform(self, X):
        return X @ self.evecs
    
class KernelPCA:
    def __init__(self, kernel='rbf', gamma=10) -> None:
        if kernel == 'rbf':
            self.kernel_fn = partial(KernelPCA._rbf, gamma)
        else:
            raise NotImplementedError

    def _rbf(gamma, X):
        K = np.exp(-np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)**2 * gamma)
        K = K - K.mean(axis=0) - K.mean(axis=1) + K.mean()
        return K
    
    def fit(self, X: np.ndarray, n_components: int):
        X -= X.mean(axis=0)
        K = self.kernel_fn(X)
        evecs = np.linalg.svd(K, hermitian=True)[2]
        self.evecs = evecs[:n_components].T
        return self.transform(X)
    
    def transform(self, X):
        K = self.kernel_fn(X)
        return K @ self.evecs

def test_pca():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(X.shape)
    pca = PCA()
    X_pca = pca.fit(X, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.show()

def test_kernel_pca():
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=1000, factor=.3, noise=.05)
    print(X.shape)
    kpca = KernelPCA()
    X_kpca = kpca.fit(X, 2)
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
    plt.show()

if __name__ == '__main__':
    test_pca()
    test_kernel_pca()
    