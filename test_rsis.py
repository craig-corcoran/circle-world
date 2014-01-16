import numpy

def test_circleWorld(n=100, N = 100):
    cworld = CircleWorld()
    P, R = cworld.get_samples(n)

    # transform the samples into fourier feature space
    freqs = numpy.arange(N,dtype=float) / 2
    W = numpy.array(list(it.product(freqs, freqs))).T
    X = numpy.cos(numpy.dot(P,W)) # real component
    Y = numpy.exp(1j*numpy.dot(P,W)) # both real and imaginary

    assert (numpy.real(Y) == X).all()
    assert (numpy.imag(X) == 0).all()
    assert (abs(X) <= 1).all()


def test_ROML(n=10, N = 10):
    # generate samples from circle world
    cworld = CircleWorld()
    P, R = cworld.get_samples(n)

    assert P.shape == (n,2)
    assert R.shape == (n,)

    # transform positions into fourier feature representation
    fmap = FourierFeatureMap(N)
    X = fmap.transform(P)

    assert X.shape == (n,(N//2+1)**2)

    model = ROML(X.shape[1], 2)

    # check that the theano loss and numpy loss are the same
    assert model.loss(X,R) == model._loss(X,R)

    # take gradient wrt [Phi, T, q, Mz, sr]
    shapes = map(numpy.shape, model.grad(X,R))
    assert shapes == [((N//2+1)**2,2), (2,2), (2,), (2,2), ()]

