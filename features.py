import numpy
import itertools as it

class FourierFeatureMap(object):
    def __init__(self, N, use_sin = True):
        # Implicit dividing freqs by two here bc range is -1, 1
        freqs = numpy.pi * numpy.arange(N // 2 + 1,dtype=float)
        self.W = numpy.array(list(it.product(freqs, freqs))).T
        self.use_sin = use_sin

    def transform(self,P):
        feats = numpy.cos(numpy.dot(P,self.W))  # real component
        if self.use_sin:
            feats = numpy.concatenate([feats, numpy.sin(numpy.dot(P,self.W))])  # complex component
        return feats

