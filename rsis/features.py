import logging
import numpy as np
import numpy.random as rng
import itertools as it

class FourierFeatureMap(object):
    def __init__(self, N, use_sin = True):
        # Implicit dividing freqs by two here bc range is -1, 1
        freqs = np.pi * np.arange(N)
        self.W = np.array(list(it.product(freqs, freqs))).T
        self.use_sin = use_sin

    def transform(self,P):
        prod = np.dot(P,self.W)
        feats = np.cos(prod)  # real component
        if self.use_sin:
            feats = np.append(feats, np.sin(prod), axis = 1)  # complex component
        logging.info('states %s x weights %s = features %s',
                     P.shape, self.W.shape, feats.shape)
        return feats


class TileFeatureMap(object):
    def __init__(self, N, size = 1.):
        W = []
        for s in size * rng.uniform(0.1, 0.9, size=N):
            x = rng.uniform(-size, size - s)
            y = rng.uniform(-size, size - s)
            W.append([x, y, x + s, y + s])
        self.X1, self.Y1, self.X2, self.Y2 = np.array(W).T

    def transform(self, P):
        return np.array([
            (self.X1 <= x) & (x <= self.X2) & (self.Y1 <= y) & (y <= self.Y2)
            for x, y in P], int)
