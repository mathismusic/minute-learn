import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(42)

class KMeansClassifier:
    def __init__(self, nclusters, algo='++') -> None:
        self.algo = algo
        self.nclusters = nclusters
        self.centroids = None
        self.labels = None

    def update(self):
        for i in range(self.nclusters):
            self.centroids[i] = np.mean(self.X[self.labels == i], axis=0)

    def assign(self):
        dists = np.linalg.norm(self.X[:, None, :] - self.centroids[None, :, :], axis=2)
        self.labels = np.argmin(dists, axis=1)
        
    def fit(self, X: np.ndarray, max_iters=None):
        self.X = X
        self.centroids = self.initialize_centroids()
        self.labels = self.initialize_labels()
        iters = 0
        while max_iters is None or iters < max_iters:
            old_assignment = self.labels.copy()
            self.assign()
            if np.all(old_assignment == self.labels):
                break
            self.update()
            iters += 1
        return self.labels
    
    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2), axis=1)

    def initialize_centroids(self):
        if self.algo == 'vanilla':
            idxs = np.random.choice(self.X.shape[0], self.nclusters, replace=False)
            return self.X[idxs]
        elif self.algo == '++':
            centroids = self.X[np.random.choice(self.X.shape[0])][None, :]
            for _ in range(self.nclusters - 1):
                dists = np.linalg.norm(self.X[:, None, :] - centroids[None, :, :], axis=2)
                probs = np.min(dists, axis=1)
                probs /= probs.sum()
                new_centroid = self.X[np.random.choice(self.X.shape[0], p=probs)][None, :]
                centroids = np.concatenate([centroids, new_centroid], axis=0)
            return np.array(centroids)
        
    def initialize_labels(self):
        return np.zeros(self.X.shape[0])
    
def gmm(n_samples, n_clusters=4):
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters,
                           cluster_std=1.0, random_state=0)
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_true)
    plt.title('Gaussian mixture model')
    plt.savefig('gmm-colored.png')
    return X, y_true

if __name__ == '__main__':
    n_samples, n_clusters = 1200, 7
    X, y_true = gmm(n_samples, n_clusters)
    kmeans = KMeansClassifier(n_clusters)
    y_pred = kmeans.fit(X)
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_pred)
    plt.title('K-Means clustering')
    plt.savefig('kmeans-colored.png')