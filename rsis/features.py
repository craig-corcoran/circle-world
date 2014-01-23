import numpy as np
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
    def __init__(self, depth = 3):
        W = []
        active = [[-1., -1., 1., 1.]]
        for _ in range(depth):
            new = []
            for x1, y1, x2, y2 in active:
                xm = (x1 + x2) / 2.
                ym = (y1 + y2) / 2.
                new.append([x1, y1, xm, ym])
                new.append([x1, ym, xm, y2])
                new.append([xm, ym, x2, y2])
                new.append([xm, y1, x2, ym])
            active = new
            W.extend(active)
        self.X1, self.Y1, self.X2, self.Y2 = np.array(W).T

    def transform(self, P):
        return np.array([
            (self.X1 <= x) & (x <= self.X2) &
            (self.Y1 <= y) & (y <= self.Y2)
            for x, y in P], int)
