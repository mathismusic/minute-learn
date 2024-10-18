import numpy as np

def discrete(nsamples, nfeatures, range):
    X = np.random.randint(0, range, size=(nsamples, nfeatures))
    return X

def continuous(nsamples, nfeatures):
    X = np.random.rand(size=(nsamples, nfeatures))
    return X