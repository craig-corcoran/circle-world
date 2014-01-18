import os
import copy
import numpy 
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rsis

logger = rsis.get_logger(__name__)

# add non-scaled loss component output
# viewing learned reward and value function (vs true)
# plot train/test loss over time

# l1 and l2 regularization / grid search
# closed-form coordinate descent?
# natural gradient
# use shift to avoid singular matrix inversion?

# LSTD and TD0 baselines
# solving for true value function
# performance measure
# check for sign changes and set to zero

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
                  vmin = -lim, 
                  vmax = lim,
                  interpolation = 'nearest')

    #plt.gcf().set_size_inches((n_sp2 / 2., n_sp1 / 2.))
    plt.savefig(file_name)


def view_fourier_basis(N = 15, n_sp1 = 16, n_sp2 = 15,
                       shuffle = False, last = False, use_sin = True):
    # plot a regular grid
    P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
    fmap = rsis.FourierFeatureMap(N, use_sin)
    X = fmap.transform(P)

    if shuffle: # shuffle the columns of X
        numpy.random.shuffle(numpy.transpose(X))

    plot_filters(X, (n_sp1, n_sp2), 'plots/fourier_basis.png', last)


def view_position_scatterplot(P):
    plt.scatter(P[:,0], P[:,1])
    plt.savefig('plots/scatter.png')


def main(
        n=1000, # number of samples
        N = 20, # grid/basis resolution, num of fourier funcs: d = N/2+1, grid is [2Nx2N]
        k = 4,  # number of compressed features, Phi is [dxk]
        max_iter = 3, # max number of cg optimizer steps per iteration
        l1 = 2e-5, # applied to Phi
        l2 = 1e-8, # applied to self.params[1:]
        shift = 1e-12,
        lock_phi0 = False, # allow the first column of Phi to change when ascending log prob
        ):

    # move previous plots to old folder
    os.system('rm plots/learned_basis*')
    
    cworld = rsis.CircleWorld()
    fmap = rsis.FourierFeatureMap(N, use_sin = True)

    it = 0

    def sample_circle_world(n):
        P, R = cworld.get_samples(n)
        X = fmap.transform(P)
        return X, R

    def plot_learned(N):
        # plot a regular grid
        logger.info('plotting current features')
        P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
        X = fmap.transform(P)
        Z = model.encode(X)
        l = int(numpy.sqrt(k))
        plot_filters(Z, (l, l), 'plots/learned_basis_%05d.png' % it)

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

        r, z, n, l = model._loss(X_test, R_test)
        logger.info('loss components: r %f z %f norm %f = %f -- reg %f',
                    r, z, n, r + z + n, l)

        r, z, n, l = model.unscaled_loss(X_test, R_test)
        logger.info('unscaled loss components: r %f z %f norm %f = %f -- reg %f',
                    r, z, n, r + z + n, l)

        plot_learned(2*N)

    X_test, R_test = sample_circle_world(4*n)

    view_position_scatterplot(cworld.get_samples(n)[0])

    #model = RSIS(X_test.shape[1], k, l1 = l1, l2 = l2, shift = shift)
    #model = rsis.CD_RSIS(X_test.shape[1], k, l1 = l1, l2 = l2, shift = shift)
    model = rsis.AR_RSIS(X_test.shape[1], k, l1 = l1, l2 = l2, shift = shift, lock_phi0 = lock_phi0)

    try:
        for it in range(50):
            logger.info('*** iteration ' + str(it) + '***')
            X, R = sample_circle_world(n)
            #model.set_noise_params(X,R)
            
            logger.info('descending reward loss')
            model.set_params(
                scipy.optimize.fmin_cg(
                    model.optimize_rew_loss, model.flat_params, model.optimize_rew_grad,
                    args = (X, R),
                    full_output = False,
                    maxiter = max_iter)
            )
                        
            logger.info('descending neg log probability')
            model.set_params(
                scipy.optimize.fmin_cg(
                    model.optimize_loss, model.flat_params, model.optimize_grad,
                    args = (X, R),
                    full_output = False,
                    maxiter = max_iter)
            )
            log()
    except KeyboardInterrupt:
        logger.info( '\n user stopped current training loop')


if __name__ == '__main__':
    rsis.script(main)
