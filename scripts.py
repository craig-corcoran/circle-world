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
        n = 500, # number of samples per minibatch
        N = 20, # grid/basis resolution, num of fourier funcs. grid is [2Nx2N]
        k = 16,  # number of compressed features, Phi is [dxk]
        h = 20, # horizon for reward loss
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
    
    logger.info('constructing world and feature map')
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

    def evaluate(x_test = None, r_test = None):
        if x_test is None:
            x_test, r_test = sample_circle_world(4*n)

        lstd_w = numpy.linalg.solve(C + shift * numpy.identity(fmap.d), b)
        model_val = model.value_func(x_test, gam)
        lstd_val = numpy.dot(x_test, lstd_w)

        model_err = td_error(model_val, r_test)
        lstd_err = td_error(lstd_val, r_test)

        logger.info('sample model error: %s' % str(model_err))
        logger.info('sample lstd error: %s' % str(lstd_err))


    def td_error(v, r, all_steps = True):
        ntot = 0
        los = 0.
        for i in xrange(1, r.shape[0] if all_steps else 2):
            los += numpy.sum((r[:-i] + gam * v[i:] - v[:-i])**2) 
            ntot += r[:-i].shape[0]
        return los / ntot

    def log():
        logger.info('train loss: %.5f', model.loss(X, R))
        logger.info('test loss: %.5f', model.loss(X_test, R_test))
        evaluate(X_test, R_test)

        phi_norms = numpy.apply_along_axis(numpy.linalg.norm, 0, model.Phi)
        T_norms = numpy.apply_along_axis(numpy.linalg.norm, 0, model.T)
        #S_norms = numpy.apply_along_axis(numpy.linalg.norm, 0, model.Sz)

        logger.info('average Phi column norm: ' + str(numpy.average(phi_norms)))
        logger.info('average T column norms: ' + str(numpy.average(T_norms)))
        logger.info('q norm: ' + str(numpy.linalg.norm(model.q)))
        #logger.info('Sz column norms: ' + str(S_norms))
        #logger.info('s: ' + str(model.sr))

        if it % 10 == 0:
            plot_learned(N)

    #view_position_scatterplot(world.get_samples(n)[0])
    logger.info('constructing theano model')
    model = rsis.Horizon_RSIS(fmap.d, k, h, l1=l1, l2=l2, shift=shift)
    
    losses = collections.OrderedDict(
                    [
                    ('smooth', (model.optimize_smooth_loss, model.optimize_smooth_grad)),
                    ('rew_horizon', (model.optimize_loss, model.optimize_grad)),
                    ])
    
    X_test, R_test = sample_circle_world(4*n)
    C = numpy.zeros((fmap.d, fmap.d))
    b = numpy.zeros(fmap.d)
    try:
        for it in range(batches):
            logger.info('*** iteration ' + str(it) + '***')
            
            X, R = sample_circle_world(n)
            C += numpy.dot(X[:-1].T, X[:-1]-gam*X[1:])
            b += numpy.dot(X[:-1].T, R[:-1])

            for key, val in losses.items():
                
                opt_loss, opt_grad = val

                logger.info('descending %s loss' % key)
                model.set_params(
                    scipy.optimize.fmin_cg(
                        opt_loss,
                        model.flat_params,
                        opt_grad,
                        args=(X, R),
                        full_output=False,
                        maxiter=max_iter)
                )
                log()

    except KeyboardInterrupt:
        logger.info( '\n user stopped current training loop')

    def plot_value_rew():
        
        P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
        X = fmap.transform(P)
        R = world.reward_func(P) # true reward function
        
        lstd_w = numpy.linalg.solve(C, b) 
        lstd_val = numpy.dot(X, lstd_w)
        model_val = model.value_func(X, gam)
        x, r = sample_circle_world(4*n)
        model_lstd_val = model.lstd_value_func(x, r, gam, X)
        model_rew = model.reward_func(X)

        value_list = [model_val, model_lstd_val, lstd_val]
        if hasattr(world, 'value_func'):
            value_list.append(world.value_func(P))

        plot_filters(numpy.vstack([model_rew, R]).T, (1, 2), file_name='plots/rewards.png')
        plot_filters(numpy.vstack(value_list).T, (1, len(value_list)), file_name='plots/values.png')

    plot_learned(N)
    plot_value_rew()    
    evaluate(X_test, R_test)

    logger.info('final q: ' + str(model.q))
    logger.info('final w: ' + str(model.w))

if __name__ == '__main__':
    rsis.script(main)
    #view_fourier_basis()
