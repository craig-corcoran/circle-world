import os
import time
import datetime
import copy
import collections
import numpy 
import pandas
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rsis


logger = rsis.get_logger(__name__)

# check for sign changes and set to zero when using l1
# thorough hyperparameter search: fancy hill climbing?
# see if learning is stable if the model is initialized to the true model values
# plot bellman error components, and target loss over time (train + test)
# compare to TD(0) (lambda?) 
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
        batches = 30, # number of iterations to run cg for
        n = 1000, # number of samples per minibatch
        N = 20, # grid/basis resolution, num of fourier funcs. grid is [2Nx2N]
        k = 16,  # number of compressed features, Phi is [dxk]
        max_iter = 3, # max number of cg optimizer steps per iteration
        l1 = 1e-12,
        l2d = 1e-12,
        l2k = 1e-12,
        gam = 1-1e-2,
        ):

    #os.system('rm plots/*.png')
    os.system('mv plots/*.png plots/old/')
    
    
    def sample_circle_world(n, plot = False, seed=None):
        P, R = world.get_samples(n, seed=seed)

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
        l = int(numpy.sqrt(k)) # XXX
        tup = (l,k//l) if k > 2 else (1,k)
        plot_filters(Z, tup, 'plots/learned_basis_.%05d.png' % it)

    def evaluate(x_test = None, r_test = None):
        if x_test is None:
            x_test, r_test = sample_circle_world(4*n)

        model_tderr = model.td_error(model.get_model_value(x_test), r_test)
        lstd_tderr = model.td_error(model.get_lstd_value(x_test), r_test) 
        
        #model_rerr = model.reward_error(x_test, r_test) / n
        #model_zerr = model.transition_error(x_test) / (n-1)
        
        logger.info('h/o sample model td error: %05f' % model_tderr)
        logger.info('h/o sample lstd td error: %05f' % lstd_tderr)
        

        return {'model-td': model_tderr,
                'lstd-td': lstd_tderr}

    logger.info('constructing world and feature map')
    #world = rsis.CircleWorld(gam = gam)
    world = rsis.TorusWorld()
    #fmap = rsis.FourierFeatureMap(N)
    fmap = rsis.TileFeatureMap(N**2)

    logger.info('dimension of raw features: ' + str(fmap.d))

 
    logger.info('constructing theano model')
    t = time.time()
    model = rsis.LowRankLSTD(fmap.d, k, gam, l1=l1, l2d=l2d, l2k=l2k)
    logger.info('time to compile model: ' + str(time.time() - t))
    
    X_test, R_test = sample_circle_world(10000) #, seed=0)

    loss_values = []
    it = 0
    try:
        for it in range(batches):
            
            logger.info('*** iteration ' + str(it) + '***')
            
            X, R = sample_circle_world(n)

            model.update_statistics(X, R)

            t = time.time()
            model.set_params(
                scipy.optimize.fmin_cg(
                    model.optimize_loss,
                    model.flat_params,
                    model.optimize_grad,
                    args=(X, R),
                    full_output=False,
                    maxiter=max_iter)
            )
            logger.info('time for cg iteration: %f' % (time.time()-t))
            
            print 'Phi norms: ',numpy.apply_along_axis(numpy.linalg.norm, 0, model.Phi)
             
            if it % 10 == 0:
                plot_learned(N)

            loss_values.append(evaluate(X_test, R_test))

    except KeyboardInterrupt:
        logger.info('\n user stopped current training loop')
    
    timestamp = str(datetime.datetime.now()).replace(' ','.').replace(':','.')


    plot_learned(N)

    df = pandas.DataFrame(loss_values)
    df.to_csv('data/losses%s.csv' % timestamp)
    
    def plot_loss_curve():
        
        # XXX add line for td error of the true value function 

        plt.clf()
        
        n_cols = len(df.columns)
        for i, c in enumerate(df.columns):
            y = df[c].values
            plt.subplot(n_cols, 1, i)
            plt.semilogy(numpy.arange(1, len(y)+1), y, label = c)

        for i in xrange(n_cols):
            plt.subplot(n_cols, 1, i)
            plt.legend()        
        
        plt.savefig('plots/loss-curve%s.png' % timestamp)

    def plot_value_rew():
        
        P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
        X = fmap.transform(P)
        R = world.reward_func(P) # true reward function
        
        model_val = model.get_model_value(X) 
        lstd_val = model.get_lstd_value(X)
        
        #model_rew = model.reward_func(X)
        value_list = [model_val, lstd_val]
        
        if hasattr(world, 'value_func'):
            value_list.append(world.value_func(P))

        #plot_filters(numpy.vstack([model_rew, lstd0_rew, lstd1_rew, R]).T, (1, 4), file_name='plots/rewards%s.png' % timestamp)
        plot_filters(numpy.vstack(value_list).T, (1, len(value_list)), file_name='plots/values%s.png' % timestamp)

    plot_value_rew()    
    plot_loss_curve()
    logger.info('norm of model weights: %05f' % numpy.sum(numpy.dot(model.Phi, model.u)**2))    
    logger.info('norm of lstd weights: %05f' % numpy.sum(model.w**2))  


if __name__ == '__main__':

    rsis.script(main)
