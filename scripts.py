#import time

#import itertools as it
#import numpy
#from numpy import array, dot
#from numpy.random import standard_normal
#import matplotlib.pyplot as plt
#import theano.tensor as TT
#import theano.sandbox.linalg.ops as LA
#import theano
import numpy 
import rsis
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from features import FourierFeatureMap
from domains import CircleWorld
from model import RSIS

logger = rsis.get_logger(__name__)


# scipy optimizer
# wtf gradient descent?
# truncate fourier space
# l1 and l2 regularization / grid search
# closed-form coordinate descent?
# natural gradient

# use shift?
# regularization

def plot_filters(X, n_plot, file_name = 'basis.png', last = False):
    plt.clf()
    n_sp1 = numpy.sqrt(n_plot) if n_plot > 2 else n_plot
    n_sp2 = n_sp1 if n_plot > 2 else 1
    side = numpy.sqrt(X.shape[0])
    gs = gridspec.GridSpec(int(n_sp1), int(n_sp2))
    gs.update(wspace=0., hspace=0.)

    lim = abs(X).max()
    for i in xrange(n_plot):
        ax = plt.subplot(gs[i])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = X[:,-(i+1)] if last else X[:,i]
        plt.imshow(numpy.reshape(im, (side,side)), cmap = 'gray', vmin = -lim, vmax = lim)

    plt.tight_layout()
    plt.savefig(file_name)


def view_fourier_basis(N = 10, n_plot = 64,
                       shuffle = False, last = False, use_sin = False):
    # plot a regular grid
    P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
    fmap = FourierFeatureMap(N, use_sin)
    X = fmap.transform(P)

    if shuffle: # shuffle the columns of X
        numpy.random.shuffle(numpy.transpose(X))

    plot_filters(X, n_plot, 'fourier_basis.png', last)


def main(n=1000, N = 20, k = 4):
    cworld = CircleWorld()
    P, R = cworld.get_samples(n)

    fmap = FourierFeatureMap(N)

    X = fmap.transform(P)
    assert X.shape == (n, (N//2+1)**2)

    model = RSIS(N**2, k)

    print 'gradient dimensions: ', map(numpy.shape, model.grad(X,R))

    loss = numpy.sum(model.loss(X,R))
    losses = model.loss(X,R)
    _losses = model._loss(X,R)

    i = 0

    def log():
        print 'theano losses: ', model.loss(X,R)
        print 'numpy  losses: ', model._loss(X,R)
        print 'loss sum: ', numpy.sum(model.loss(X,R))
        print model.Mz
        print model.Sz
        print 'iteration: ', i

    def plot_learned():
        # plot a regular grid
        P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
        fmap = FourierFeatureMap(N)
        X = fmap.transform(P)
        Z = model.encode(X)
        plot_filters(Z, k, 'learned_basis_%05d.png' % i)

    log()
    it = 0
    waiting = 0
    best_params = None
    best_test_loss = None
    try:
        while (waiting < patience):
            it += 1
            logger.info('*** iteration ' + str(it) + '***')
            
            old_params = copy.deepcopy(basis.flat_params)
            for loss_, wrt_ in ((loss, wrt), ('bellman', ['w'])):
                basis.set_loss(loss_, wrt_)
                basis.set_params(scipy.optimize.fmin_cg(
                        basis.loss, basis.flat_params, basis.grad,
                        args = (S, R, Mphi, Mrew),
                        full_output = False,
                        maxiter = max_iter,
                        ))
            basis.set_loss(loss, wrt) # reset loss back from bellman
             
            delta = numpy.linalg.norm(old_params-basis.flat_params)
            logger.info('delta theta: %.2f' % delta)
            
            norms = numpy.apply_along_axis(numpy.linalg.norm, 0, basis.thetas[0])
            logger.info( 'column norms: %.2f min / %.2f avg / %.2f max' % (
                norms.min(), norms.mean(), norms.max()))
            
            err = basis.loss(basis.flat_params, S_val, R_val, Mphi, Mrew)
            
            if err < best_test_loss:
                
                if ((best_test_loss - err) / best_test_loss > min_imp) and (delta > min_delta):
                    waiting = 0
                else:
                    waiting += 1
                    logger.info('iters without better %s loss: %i' % (basis.loss_type, int(waiting)))

                best_test_loss = err
                best_params = copy.deepcopy(basis.flat_params)
                logger.info('new best %s loss: %.2f' % (basis.loss_type, best_test_loss))
                
            else:
                waiting += 1
                logger.info('iters without better %s loss: %i' % (basis.loss_type, int(waiting)))

            Bs = basis.encode(encoder.B, False)
            d_loss_learning = record_loss(d_loss_learning)

    except KeyboardInterrupt:
        logger.info( '\n user stopped current training loop')
    
    # set params to best params from last loss
    basis.set_params(vec = best_params)
    switch.append(it-1)


    delta = -1
    step = 1e-3
    wait = 5
    count = 0
    while count < wait:

        if (delta > -1e-8) & (delta < 0):
            print 'count'
            count += 1

        model.grad_step(X,R,step)
        losses = model.loss(X,R)
        _losses = model._loss(X,R)

        new_loss = numpy.sum(losses)
        delta = new_loss - loss

        if delta > 0:
            print 'reverting last step'
            model.revert_last_delta()
            if abs(loss - numpy.sum(model.loss(X,R))) > 1e-8:
                print 'revert different'
                loss = numpy.sum(model.loss(X,R))
        else:
            loss = new_loss

        i += 1
        if (i % 50) == 0:
            print 'slowing'
            log()
            step = 0.9 * step
            plot_learned()

        for j in xrange(len(losses)):
            #print abs(_losses[j] - losses[j])
            try:
                assert abs(_losses[j] - losses[j]) < 1e-8
            except AssertionError as e:
                print e
                print 'losses different'
                abs(_losses[j] - losses[j])


if __name__ == '__main__':
    #view_fourier_basis(N = 15, n_plot = 64, 
    #                   shuffle = False, last = False, use_sin = False)
    #test_circleWorld()
    #test_ROML()
    main()


