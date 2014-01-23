import os
import copy
import collections
import numpy 
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rsis


logger = rsis.get_logger(__name__)

# natural gradient
# check for sign changes and set to zero when using l1
# thorough hyperparameter search: fancy hill climbing?
# see if learning is stable if the model is initialized to the true model values
# regularize when solving for w using P and q

# performance measure: calculate TD error on sample set
# solve for value function in torus world, include reward loss function
# plot bellman error, components, and target loss over time (train + test)
# include z loss, horizon? alternating with reward horizon loss?
# compare to TD(0) (lambda?), lstd
# plot distribution from rolling the learned/true model out through time

def plot_filters(X, (n_sp1, n_sp2), file_name = 'basis.png', last = False):
    plt.clf()
    side = numpy.sqrt(X.shape[0])
    gs = gridspec.GridSpec(n_sp1, n_sp2)
    gs.update(wspace=0., hspace=0.)

    lim = abs(X).max()
    for i in xrange(min(n_sp1 * n_sp2, X.shape[1])):
        ax = plt.subplot(gs[i])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = X[:,-(i+1)] if last else X[:,i]
        plt.imshow(
                  numpy.reshape(im, (side,side)), 
                  cmap = 'gray', 
                  #vmin = -lim, 
                  #vmax = lim,
                  interpolation = 'nearest')

    plt.savefig(file_name)


def view_fourier_basis(N = 15, n_sp1 = 16, n_sp2 = 14,
                       shuffle = False, last = False):
    # plot a regular grid
    P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
    fmap = rsis.FourierFeatureMap(N)
    X = fmap.transform(P)

    if shuffle: # shuffle the columns of X
        numpy.random.shuffle(numpy.transpose(X))

    plot_filters(X, (n_sp1, n_sp2), 'plots/fourier_basis.png', last)


def main(
        batches = 100, # number of iterations to run cg for
        n = 100, # number of samples per minibatch
        N = 20, # grid/basis resolution, num of fourier funcs. grid is [2Nx2N]
        k = 16,  # number of compressed features, Phi is [dxk]
        h = None, # horizon for reward loss
        max_iter = 3, # max number of cg optimizer steps per iteration
        l1 = 2e-3, # applied to Phi
        l2 = 1e-6, # applied to params[1:]
        shift = 1e-12,
        gam = 1-1e-2,
        ):

    if h is None:
        
        logger.info('setting horizon to batch length: %i' % n)
        h = n

    #os.system('rm plots/*.png')
    os.system('mv plots/*.png plots/old/')
    
    #world = rsis.CircleWorld(gam = gam)
    world = rsis.TorusWorld()
    #fmap = rsis.FourierFeatureMap(N)
    fmap = rsis.TileFeatureMap(N**2)

    logger.info('dimension of raw features: ' + str(fmap.d))

    it = 0

    def sample_circle_world(n, plot = False):
        P, R = world.get_samples(n)

        if plot and (it % 10 == 0): 
            plot_samples(P)

        X = fmap.transform(P)
        return X, R

    def plot_samples(P):
        plt.clf()
        plt.scatter(P[:,0], P[:,1])
        plt.axis([-1.5,1.5,-1.5,1.5])
        plt.savefig('plots/sample_scatter%i' % it + '.png')

    def plot_learned(N):
        # plot a regular grid
        logger.info('plotting current features')
        P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
        X = fmap.transform(P)
        Z = model.encode(X)
        l = int(numpy.sqrt(k))
        tup = (l,l) if k > 2 else (1,k)
        plot_filters(Z, tup, 'plots/learned_basis_%05d.png' % it)

    def log():
        logger.info('train loss: %.5f', model.loss(X, R))
        logger.info('test loss: %.5f', model.loss(X_test, R_test))

        phi_norms = numpy.apply_along_axis(numpy.linalg.norm, 0, model.Phi)
        S_norms = numpy.apply_along_axis(numpy.linalg.norm, 0, model.Sz)
        T_norms = numpy.apply_along_axis(numpy.linalg.norm, 0, model.T)
        logger.info('Phi column norms: ' + str(phi_norms))
        logger.info('Sz column norms: ' + str(S_norms))
        logger.info('T column norms: ' + str(T_norms))
        logger.info('q norm: ' + str(numpy.linalg.norm(model.q)))
        logger.info('s: ' + str(model.sr))

        if it % 10 == 0:
            plot_learned(N)


    #view_position_scatterplot(world.get_samples(n)[0])

    X_test, R_test = sample_circle_world(4*n)

    model = rsis.Horizon_RSIS(fmap.d, k, h, l1=l1, l2=l2, shift=shift)
    
    losses = collections.OrderedDict([('rew_horizon', (model.optimize_loss, model.optimize_grad)),
                             ('smooth', (model.optimize_smooth_loss, model.optimize_smooth_grad))])
    try:
        for it in range(batches):
            logger.info('*** iteration ' + str(it) + '***')
            
            X, R = sample_circle_world(n)

            for k, val in losses.items():
                
                opt_loss, opt_grad = val

                logger.info('descending %s loss' % k)
                model.set_params(
                    scipy.optimize.fmin_cg(
                        model.opt_loss,
                        model.flat_params,
                        model.opt_grad,
                        args=(X, R),
                        full_output=False,
                        maxiter=max_iter)
                )
                log()

    except KeyboardInterrupt:
        logger.info( '\n user stopped current training loop')

    plot_learned(N)

    P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
    X = fmap.transform(P)
    R = world.reward_func(P) # true reward function
    V = world.value_func(P)

    x_test, r_test = sample_circle_world(n)
    lstd_val = model.lstd_value_func(x_test, r_test, gam, X)
    learned_val = model.value_func(X, world.gam)


    plot_filters(numpy.vstack([model.reward_func(X), R]).T, (1, 2), file_name='plots/rewards.png')
    plot_filters(numpy.vstack([learned_val, lstd_val, V]).T, (1, 3), file_name='plots/values.png')

    logger.info('final q: ' + str(model.q))
    logger.info('final w: ' + str(model.w))

if __name__ == '__main__':
    rsis.script(main)
    #view_fourier_basis()
