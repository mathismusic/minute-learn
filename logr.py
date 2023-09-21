import numpy as np
from gd import VanillaGD
from data_gen import LinSepBinaryData
import os, shutil
from make_gif import make_gif

class GDLogR:
    def __init__(self, loss='squared_error', penalty='l2', alpha=1e-4, batch_sz=32) -> None:
        self.loss_type = loss # ignoerd for now
        self.penalty = penalty # ignored for now
        self.alpha = alpha # igored for now
        self.batch_sz = batch_sz

    def loss(self, weights):
        fx = 1 / (1 + np.exp(-self.X@weights))
        return -(self.y * np.log(fx + 1e-12) + (1-self.y) * np.log(1-fx + 1e-12)).sum()
    
    def gradient(self, weights):
        # print(self.X.T.shape, (self.X @ weights - self.y).shape)
        idxs = self.sample()
        Xbatch = self.X[idxs]
        ybatch = self.y[idxs]
        fXbatch = 1 / (1 + np.exp(-Xbatch@weights))
        return Xbatch.T @ (fXbatch - ybatch)
    
    def sample(self, batch_sz=None):
        batch_sz = batch_sz if batch_sz is not None else self.batch_sz
        return np.random.choice(np.arange(self.X.shape[0]), size=(batch_sz,), replace=False)
        return np.random.randint(low=0, high=self.X.shape[0], size=(batch_sz,))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        assert N == y.shape[0], 'X and y are incompatible in the number of samples'
        self.X = np.hstack((X, np.ones((N, 1))))
        self.y = y
        self.gd = VanillaGD(self.loss, self.gradient, (self.X.shape[1], 1))
        self.weights = self.gd.optimize(tol=5e-3, maxiter=5000)
        return self.weights
    
    def predict(self, X: np.ndarray):
        X = np.hstack((X, 1)) # 1 x (F+1)
        return X @ self.weights
    
def test_regressor(sgd=True):
    gen = LinSepBinaryData
    X, y = gen.generate_linearly_separable(n=500, seed=1)
    model = GDLogR(batch_sz=32 if sgd else len(X))
    weights = model.fit(X, y)
    print("loss", model.loss(weights))
    print("iters:", model.gd.iters)
    gen.plot_data_and_separator2d(X, y, weights).savefig('test-logr')
    # make gif of the training procedure
    imgdir = f'images/test_{"S" if sgd else ""}GDLogR'
    if os.path.exists(imgdir): shutil.rmtree(imgdir)
    os.makedirs(imgdir)
    for i, w in enumerate(model.gd.snapshots):
        if not i%50 or i == model.gd.iters: gen.plot_data_and_separator2d(X, y, w, os.path.join(imgdir, f'{i:06}'))
    make_gif(f'test_{"S" if sgd else ""}GDLogR')

if __name__ == '__main__':
    test_regressor(0)
    test_regressor(1)