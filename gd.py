import numpy as np

class VanillaGD:
    def __init__(self, loss_fn, grad_fn, init_weights: np.ndarray|tuple[int]) -> None:
        self.weights = init_weights if isinstance(init_weights, np.ndarray) else np.zeros(init_weights)
        self.loss_fn = loss_fn # unused rn
        self.grad_fn = grad_fn

    def optimize(self, lr=5e-3, tol=1e-3, maxiter=10000, callback=None):
        """
        perform gradient descent.
        """
        self.snapshots = []
        for _ in range(maxiter):
            self.snapshots.append(self.weights.copy())
            # print(self.loss_fn(self.weights))
            if callback: callback(self.weights)
            grad = self.grad_fn(self.weights)
            if np.linalg.norm(grad) < tol:
                break
            self.weights -= lr * grad
        self.iters = _
        return self.weights