import numpy as np
import matplotlib.pyplot as plt
from gd import VanillaGD
from data_gen import LinRegressionData
import os, shutil
from make_gif import make_gif

class GDRegressor:
    def __init__(self, loss='squared_error', penalty='l1', alpha=1e-4) -> None:
        self.loss_type = loss # ignored for now
        self.penalty = penalty
        self.alpha = alpha

    def loss(self, weights):
        reg = 0.
        if self.penalty == 'l1':
            reg = np.linalg.norm(weights[:-1], 1)
        elif self.penalty == 'l2':
            reg = np.linalg.norm(weights[:, -1], 2)
        return np.sum(np.square(self.X @ weights - self.y))/self.X.shape[0] + self.alpha * reg
    
    def gradient(self, weights):
        # print(self.X.T.shape, (self.X @ weights - self.y).shape)
        grad_reg = np.zeros_like(weights)
        if self.penalty == 'l1':
            grad_reg = np.sign(weights)
        elif self.penalty == 'l2':
            grad_reg = 2 * weights
        return 2 * self.X.T @ (self.X @ weights - self.y)/self.X.shape[0] + self.alpha * grad_reg

    def fit(self, X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        assert N == y.shape[0], 'X and y are incompatible in the number of samples'
        self.X = np.hstack((X, np.ones((N, 1))))
        self.y = y
        self.gd = VanillaGD(self.loss, self.gradient, (self.X.shape[1], 1))
        self.weights = self.gd.optimize()
        return self.weights
    
    def predict(self, X: np.ndarray):
        X = np.hstack((X, 1)) # 1 x (F+1)
        return X @ self.weights


class SGDRegressor:
    def __init__(self, loss='squared_error', penalty='l2', alpha=1e-4, batch_sz=32) -> None:
        self.loss_type = loss # ignoerd for now
        self.penalty = penalty
        self.alpha = alpha
        self.batch_sz = batch_sz

    def loss(self, weights):
        reg = 0.
        if self.penalty == 'l1':
            reg = np.linalg.norm(weights[:-1], 1)
        elif self.penalty == 'l2':
            reg = np.linalg.norm(weights[:, -1], 2)
        return np.sum(np.square(self.X @ weights - self.y))/self.X.shape[0] + self.alpha * reg
    
    def gradient(self, weights):
        # print(self.X.T.shape, (self.X @ weights - self.y).shape)
        idxs = self.sample()
        Xbatch = self.X[idxs]
        ybatch = self.y[idxs]
        grad_reg = np.zeros_like(weights)
        if self.penalty == 'l1':
            grad_reg = np.sign(weights)
        elif self.penalty == 'l2':
            grad_reg = 2 * weights
        return 2 * Xbatch.T @ (Xbatch @ weights - ybatch)/Xbatch.shape[0] + self.alpha * grad_reg

    def sample(self, batch_sz=None):
        batch_sz = batch_sz if batch_sz is not None else self.batch_sz
        return np.random.randint(low=0, high=self.X.shape[0], size=(batch_sz,))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        assert N == y.shape[0], 'X and y are incompatible in the number of samples'
        self.X = np.hstack((X, np.ones((N, 1))))
        self.y = y
        self.gd = VanillaGD(self.loss, self.gradient, (self.X.shape[1], 1))
        self.weights = self.gd.optimize(tol=1e-2) # does not converge with 5e-3 (oscillates)
        return self.weights
    
    def predict(self, X: np.ndarray):
        X = np.hstack((X, 1)) # 1 x (F+1)
        return X @ self.weights
 
    
def test_regressor(X, y, sgd=True):
    gen = LinRegressionData
    model = SGDRegressor() if sgd else GDRegressor() # that's all! Miraculous
    weights = model.fit(X, y)
    print("loss", model.loss(weights))
    print("iters:", model.gd.iters)
    gen.plot_data_and_weight1d(X, y, weights).savefig('test')
    # make gif of the training procedure
    imgdir = f'images/test_{"S" if sgd else ""}GDRegressor'
    if os.path.exists(imgdir): shutil.rmtree(imgdir)
    os.makedirs(imgdir)
    for i, w in enumerate(model.gd.snapshots):
        if not i%50 or i == model.gd.iters: gen.plot_data_and_weight1d(X, y, w, os.path.join(imgdir, f'{i:06}'))
    make_gif(f'test_{"S" if sgd else ""}GDRegressor')

def test_with_optimal():
    from data_gen import LinRegressionData
    gen = LinRegressionData
    X, y = gen.generate_regression(n=2000, sigma=2.5)
    model = GDRegressor()
    weights = model.fit(X, y)
    print(model.gd.iters)
    X2 = np.hstack((X, np.ones((X.shape[0], 1))))
    optimal = np.linalg.solve(X2.T @ X2, X2.T @ y).reshape(-1, 1)
    fig, ax = plt.subplots(1, 1)
    ax.plot(X, y, 'x')
    xmin, xmax = np.min(X), np.max(X)
    xs = np.array([xmin, xmax]).reshape(1, -1)
    ys = weights.T @ np.vstack((xs, [1, 1]))
    ax.plot(xs[0], ys[0], 'b-', linewidth=3)
    ys = optimal.T @ np.vstack((xs, [1, 1]))
    ax.plot(xs[0], ys[0], 'r-')
    fig.savefig('check-with-optimal')

if __name__ == '__main__':
    gen = LinRegressionData
    X, y = gen.generate_regression(sigma=0.5)
    test_regressor(X,y,0)
    test_regressor(X,y,1)
    # test_with_optimal()