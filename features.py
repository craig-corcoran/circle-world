import numpy
import itertools as it

class FourierFeatureMap(object):
    def __init__(self, N, use_sin = False):
        # Implicit dividing freqs by two here bc range is -1, 1
        freqs = numpy.pi * numpy.arange(N // 2 + 1,dtype=float)
        self.W = numpy.array(list(it.product(freqs, freqs))).T
        self.sin = use_sin

    def transform(self,P):
        if self.sin:
            return numpy.sin(numpy.dot(P,self.W))  # complex component
        else:
            return numpy.cos(numpy.dot(P,self.W))  # real component

