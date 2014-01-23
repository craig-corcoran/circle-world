import os
import copy
import numpy 
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rsis

logger = rsis.get_logger(__name__)

# plot train/test loss over time
# natural gradient
# LSTD and TD0 baselines
# solving for true value function
# performance measure
# check for sign changes and set to zero
# thorough hyperparameter search
# plot distribution through time according to the learned model (v. true model)
# see if learning is stable if the model is initialized to the true model values
# calculate TD error on sample set
# use full feature space (reduces to LSTD-like alg)
# see if features are effective without using T and q to solve for w
# regularize when solving for w using P and q

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
        n = 1000, # number of samples per minibatch
        N = 20, # grid/basis resolution, num of fourier funcs. grid is [2Nx2N]
        k = 4,  # number of compressed features, Phi is [dxk]
        h = 10,
        max_iter = 3, # max number of cg optimizer steps per iteration
        l1 = 2e-3, # applied to Phi
        l2 = 0., # applied to self.params[1:]
        shift = 1e-12,
        gam = 1-1e-2,
        lock_phi0 = False, # allow the first column of Phi to change when ascending log prob
        ):

    # move previous plots to old folder
    #os.system('rm plots/*.png')
    os.system('mv plots/*.png plots/old/')
    
    cworld = rsis.CircleWorld(gam = gam)
    fmap = rsis.FourierFeatureMap(N)

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


    #view_position_scatterplot(cworld.get_samples(n)[0])

    X_test, R_test = sample_circle_world(4*n)

    model = rsis.Horizon_RSIS(fmap.d, k, h, l1=l1, l2=l2, shift=shift)
    try:
        for it in range(100):
            logger.info('*** iteration ' + str(it) + '***')
            
            X, R = sample_circle_world(n)
            
            #logger.info('descending reward loss')
            #model.set_params(
                #scipy.optimize.fmin_cg(
                    #model.optimize_smooth_loss, 
                    #model.flat_params, 
                    #model.optimize_smooth_grad,
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

    plot_learned(N)

    P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
    X = fmap.transform(P)
    R = cworld.reward_func(P) # true reward function
    V = cworld.value_func(P)

    x_test, r_test = sample_circle_world(n)
    lstd_val = model.lstd_value_func(x_test, r_test, gam, X)
    learned_val = model.value_func(X, cworld.gam)


    plot_filters(numpy.vstack([model.reward_func(X), R]).T, (1, 2), file_name='plots/rewards.png')
    plot_filters(numpy.vstack([learned_val, lstd_val, V]).T, (1, 3), file_name='plots/values.png')

    logger.info('final q: ' + str(model.q))
    logger.info('final w: ' + str(model.w))

if __name__ == '__main__':
    rsis.script(main)
    #view_fourier_basis()
