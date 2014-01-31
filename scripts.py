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
        batches = 50, # number of iterations to run cg for
        n = 1000, # number of samples per minibatch
        N = 20, # grid/basis resolution, num of fourier funcs. grid is [2Nx2N]
        k = 8,  # number of compressed features, Phi is [dxk]
        h = 20, # horizon for reward loss
        max_iter = 3, # max number of cg optimizer steps per iteration
        l1 = 1e-2, # applied to Phi
        l2 = 1e-2, # applied to params[1:]
        shift = 1e-12,
        gam = 1-1e-2,
        ):

    #os.system('rm plots/*.png')
    os.system('mv plots/*.png plots/old/')
    
    
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
        plot_filters(Z, tup, 'plots/learned_basis_%s.%05d.png' % (key, it))

    def evaluate(x_test = None, r_test = None):
        if x_test is None:
            x_test, r_test = sample_circle_world(4*n)

        model_val = model.value_func(x_test, gam)
        lstd_val0 = lstd0.get_value(x_test)
        lstd_val1 = lstd1.get_value(x_test)

        target_err = model.loss(x_test, r_test)

        model_err0 = td_error(model_val, r_test, all_steps=False)
        model_err1 = td_error(model_val, r_test, all_steps=True)
        lstd0_err0 = td_error(lstd_val0, r_test, all_steps=False)
        lstd0_err1 = td_error(lstd_val0, r_test, all_steps=True)
        lstd1_err0 = td_error(lstd_val1, r_test, all_steps=False)
        lstd1_err1 = td_error(lstd_val1, r_test, all_steps=True)

        model_rerr = model.reward_error(x_test, r_test) / n
        model_zerr = model.transition_error(x_test) / (n-1)
        
        logger.info('h/o sample target loss: %.5f', model.loss(X_test, R_test))
        logger.info('h/o sample reward error: %s' % str(model_rerr))
        logger.info('h/o sample transition error: %s' % str(model_zerr))
        logger.info('h/o sample lstd0 error; all-steps: %05f; one-step: %05f' % (lstd0_err1, lstd0_err0))
        logger.info('h/o sample lstd1 error; all-steps, %05f; one-step: %05f' % (lstd1_err1, lstd1_err0))
        logger.info('h/o sample model error; all-steps, %05f; one-step: %05f' % (model_err1, model_err0))

        return {'td-error0': model_err0,
                'td-error1': model_err1,
                'lstd0-error0': lstd0_err0,
                'lstd1-error0': lstd1_err0,
                'lstd0-error1': lstd0_err1,
                'lstd1-error1': lstd1_err1,
                'reward-error': model_rerr, 
                'transition-error': model_zerr,
                'target-error': target_err}

    def td_error(v, r, all_steps = True):
        ntot = 0
        los = 0.
        for i in xrange(1, r.shape[0] if all_steps else 2):
            los += numpy.sum((r[:-i] + gam * v[i:] - v[:-i])**2) 
            ntot += r[:-i].shape[0]
        return los / ntot

    def print_param_norms():
        phi_norms = numpy.apply_along_axis(numpy.linalg.norm, 0, model.Phi)
        T_norms = numpy.apply_along_axis(numpy.linalg.norm, 0, model.T)
        logger.info('average Phi column norm: ' + str(numpy.average(phi_norms)))
        logger.info('average T column norms: ' + str(numpy.average(T_norms)))
        logger.info('q norm: ' + str(numpy.linalg.norm(model.q)))
    
    
    logger.info('constructing world and feature map')
    world = rsis.CircleWorld(gam = gam)
    #world = rsis.TorusWorld()
    fmap = rsis.FourierFeatureMap(N)
    #fmap = rsis.TileFeatureMap(N**2)

    logger.info('dimension of raw features: ' + str(fmap.d))

 
    logger.info('constructing theano model')
    t = time.time()
    model = rsis.QR_RSIS(fmap.d, k, h, l1=l1, l2=l2, shift=shift)
    #model = rsis.Multistep_RSIS(fmap.d, k, h, l1=l1, l2=l2, shift=shift)
 
    logger.info('time to compile model: ' + str(time.time() - t))
    
    losses = collections.OrderedDict(
                    [
                    ('mulistep likelihood', (model.optimize_loss, model.optimize_grad, model.all_params, model.all_params_t)),
                    #('model', (model.optimize_model_loss, model.optimize_model_grad, model.model_params, model.model_params_t)),
                    ])
    
    X_test, R_test = sample_circle_world(4*n)

    lstd0 = rsis.LSTD(fmap.d, all_steps = False)
    lstd1 = rsis.LSTD(fmap.d, all_steps = True)
    loss_values = []
    it = 0
    try:
        for it in range(batches):
            logger.info('*** iteration ' + str(it) + '***')
            
            X, R = sample_circle_world(n)

            # collect feature statistics for use w/ lstd 
            lstd0.update_params(X, R)
            lstd1.update_params(X, R)

            for key, val in losses.items():
                
                opt_loss, opt_grad, wrt, wrt_t = val


                logger.info('descending %s loss' % key)

                #model.set_wrt(wrt, wrt_t)

                t = time.time()
                model.set_params(
                    scipy.optimize.fmin_cg(
                        opt_loss,
                        model.flat_params,
                        opt_grad,
                        args=(X, R),
                        full_output=False,
                        maxiter=max_iter)
                )
                logger.info('time for cg iteration: %f' % (time.time()-t))
                
                loss_values.append(evaluate(X_test, R_test))
                
                #model.qr_step()
                #loss_values.append(evaluate(X_test, R_test))                
                 
                if it % 10 == 0:
                    plot_learned(N)
                
    except KeyboardInterrupt:
        logger.info( '\n user stopped current training loop')
    
    timestamp = str(datetime.datetime.now()).replace(' ','.').replace(':','.')

    logger.info('final q: ' + str(model.q))
    logger.info('final w: ' + str(model.w))
    
    logger.info('norm of model weights: %05f' % numpy.sum(numpy.dot(model.Phi, model.w)**2))
    logger.info('norm of lstd0 weights: %05f' % numpy.sum(lstd0.get_weights()**2))
    logger.info('norm of lstd1 weights: %05f' % numpy.sum(lstd1.get_weights()**2))

    plot_learned(N)

    df = pandas.DataFrame(loss_values)
    df.to_csv('data/losses%s.csv' % timestamp)
    
    def plot_loss_curve():
        
        # add line for td error of the true value function 

        plt.clf()

        for c in df.columns:

            y = df[c].values

            if c in ['transition-error', 'reward-error', 'target-error']:
                plt.subplot(311)
                plt.semilogy(numpy.arange(1,len(y)+1), y, label = c.replace('-error',''))
                #plt.ylim([0,numpy.max(df['transition-error'].values)])
            if 'lstd' in c:
                plt.subplot(312)
                plt.semilogy(numpy.arange(1,len(y)+1), y, label = c.replace('-error',''))
            if 'td' in c:
                plt.subplot(313)
                plt.semilogy(numpy.arange(1,len(y)+1), y, label = c.replace('-error',''))
                #plt.ylim([0, numpy.max(df['transition-error'].values)])
    
        plt.subplot(311); plt.legend()
        plt.subplot(312); plt.legend()
        plt.subplot(313); plt.legend()
        plt.savefig('plots/loss-curve%s.png' % timestamp)

    def plot_value_rew():
        
        P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
        X = fmap.transform(P)
        R = world.reward_func(P) # true reward function
        
        lstd_val0 = lstd0.get_value(X)
        lstd_val1 = lstd1.get_value(X)        
        model_val = model.value_func(X, gam)
        #model_lstd_val = model.lstd_value_func(x, r, gam, X) # XXX replace with lstd object statistics, add closest reward fn - projected LSTD
        model_rew = model.reward_func(X)

        value_list = [model_val, lstd_val0, lstd_val1]
        if hasattr(world, 'value_func'):
            value_list.append(world.value_func(P))

        plot_filters(numpy.vstack([model_rew, R]).T, (1, 2), file_name='plots/rewards%s.png' % timestamp)
        plot_filters(numpy.vstack(value_list).T, (1, len(value_list)), file_name='plots/values%s.png' % timestamp)

    plot_value_rew()    
    plot_loss_curve()


if __name__ == '__main__':

    rsis.script(main)
    #view_fourier_basis()
