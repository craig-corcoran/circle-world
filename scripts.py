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
                  #vmin = -lim, 
                  #vmax = lim,
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




# thorough hyperparameter search
# plot distribution through time according to the learned model (v. true model)
# see if learning is stable if the model is initialized to the true model values
# calculate TD error on sample set
# use full feature space (reduces to LSTD-like alg)
# see if features are effective without using T and q to solve for w
# regularize when solving for w using P and q

# other features, domains

# alternative ideas:
# use T^i Z_t+i for i in [1,h] where h is a small integer, can it predict further in time?
# include TD error in loss


def main(
        n = 1000, # number of samples per minibatch
        N = 20, # grid/basis resolution, num of fourier funcs: d = N/2+1, grid is [2Nx2N]
        k = 16,  # number of compressed features, Phi is [dxk]
        h = 1,
        max_iter = 3, # max number of cg optimizer steps per iteration
        l1 = 1e-4, # applied to Phi
        l2 = 1e-6, # applied to self.params[1:]
        offdiag = 2e-1, # applied to offdiag elements of Phi.T * Phi
        shift = 1e-12,
        gam = 1-1e-2,
        dyn_wt = 8e-1,
        lock_phi0 = False, # allow the first column of Phi to change when ascending log prob
        ):

    # move previous plots to old folder
    #os.system('rm plots/*.png')
    os.system('mv plots/*.png plots/old/')
    
    cworld = rsis.CircleWorld(gam = gam)
    fmap = rsis.FourierFeatureMap(N, use_sin = True)

    it = 0

    def sample_circle_world(n, plot = False):
        P, R = cworld.get_samples(n)

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
        #P = numpy.reshape(numpy.mgrid[-2:2:2*N*1j,-2:2:2*N*1j], (2,4*N*N)).T
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

        #r, z, n, l = model._loss(X_test, R_test)
        #logger.info('loss components: r %f z %f norm %f = %f -- reg %f',
        #            r, z, n, r + z + n, l)
        r, z, n, l, d = model._loss(X_test, R_test)
        logger.info('loss components: r %f z %f d %f norm %f = %f -- reg %f',
                    r, z, d, n, r + z + n + d, l)

        #r, z, n, l = model.unscaled_loss(X_test, R_test)
        #logger.info('unscaled loss components: r %f z %f norm %f = %f -- reg %f',
                    #r, z, n, r + z + n, l)
        r, z, n, l, d = model.unscaled_loss(X_test, R_test)
        logger.info('unscaled loss components: r %f z %f d %f norm %f = %f -- reg %f',
                    r, z, d, n, r + z + n + d, l)
        if it % 10 == 0:
            plot_learned(2*N)

    X_test, R_test = sample_circle_world(n)

    #view_position_scatterplot(cworld.get_samples(n)[0])

    #model = RSIS(X_test.shape[1], k, l1 = l1, l2 = l2, shift = shift)
    #model = rsis.CD_RSIS(X_test.shape[1], k, h, l1=l1, l2=l2, shift=shift)
    model = rsis.DYN_RSIS(X_test.shape[1], k, h, l1=l1, l2 = l2, dyn_wt=dyn_wt, shift=shift)
    #model = rsis.AR_RSIS(X_test.shape[1], k, l1 = l1, l2 = l2, shift = shift,
                        #lock_phi0 = lock_phi0, offdiag = offdiag)

    try:
        for it in range(100):
            logger.info('*** iteration ' + str(it) + '***')
            X, R = sample_circle_world(n)
            #model.set_noise_params(X,R)
            
            #logger.info('descending reward loss')
            #model.set_params(
                #scipy.optimize.fmin_cg(
                    #model.optimize_rew_loss, model.flat_params, model.optimize_rew_grad,
                    #args = (X, R),
                    #full_output = False,
                    #maxiter = max_iter)
            #)
                        
            logger.info('descending neg log probability')
            model.set_params(
                scipy.optimize.fmin_cg(
                    model.optimize_loss,
                    model.flat_params,
                    model.optimize_grad,
                    args=(X, R),
                    full_output=False,
                    maxiter=max_iter)
            )
            log()
    except KeyboardInterrupt:
        logger.info( '\n user stopped current training loop')

    
    P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
    #P = numpy.reshape(numpy.mgrid[-2:2:2*N*1j,-2:2:2*N*1j], (2,4*N*N)).T
    X = fmap.transform(P)
    R = cworld.reward_func(P) # true reward function
    V = cworld.value_func(P)

    x_test, r_test = sample_circle_world(4*n)
    
    plot_filters(numpy.vstack([model.reward_func(X), R]).T, (1, 2), file_name='plots/rewards.png')
    plot_filters(numpy.vstack([model.value_func(X, cworld.gam), V]).T, (1, 2), file_name='plots/values.png')

    logger.info('final q: ' + str(model.q))
    logger.info('final w: ' + str(model.w))

if __name__ == '__main__':
    rsis.script(main)
