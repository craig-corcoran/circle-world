import numpy
import itertools as it

class FourierFeatureMap(object):
    def __init__(self, N, use_sin = True):
        # Implicit dividing freqs by two here bc range is -1, 1
        freqs = numpy.pi * numpy.arange(N // 2 + 1,dtype=float)
        self.W = numpy.array(list(it.product(freqs, freqs))).T
        self.use_sin = use_sin

    def transform(self,P):
        prod = numpy.dot(P,self.W)
        feats = numpy.cos(prod)  # real component
        if self.use_sin:
            feats = numpy.append(feats, numpy.sin(prod), axis = 1)  # complex component
        return feats
            

