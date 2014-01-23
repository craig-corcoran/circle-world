import numpy as np
import numpy.random as rng
import itertools as it

class FourierFeatureMap(object):
    def __init__(self, N):
        # Implicit dividing freqs by two here bc range is -1, 1
        freqs = np.pi * np.arange(N-1,dtype=float)
        self.W = np.array(list(it.product(freqs[:N // 2 + 1], freqs))).T

    def transform(self,P):
        prod = np.dot(P,self.W)
        feats = np.cos(prod)
        feats = np.append(feats, np.sin(prod), axis = 1) 
        return feats
    
    @property
    def d(self):
        return self.W.shape[1]*2


class TileFeatureMap(object):
    def __init__(self, N, size = 1.):
        s = size * np.exp(rng.uniform(-3, -0.5, size=N))
        s.sort()
        cx = rng.uniform(-size, size, size=N)
        cy = rng.uniform(-size, size, size=N)
        self.X1 = cx - s
        self.Y1 = cy - s
        self.X2 = cx + s
        self.Y2 = cy + s

    def transform(self, P):
        return np.array([
            (self.X1 <= x) & (x <= self.X2) & (self.Y1 <= y) & (y <= self.Y2)
            for x, y in P], int)
