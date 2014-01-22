import logging
import numpy as np
import itertools as it

class FourierFeatureMap(object):
    def __init__(self, N, use_sin = True):
        # Implicit dividing freqs by two here bc range is -1, 1
        freqs = np.linspace(-2 * np.pi, 2 * np.pi, N)
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
