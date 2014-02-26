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

# plot reward and transition error over time
# crop beginning of plots (when weight norm is 0)
# plot current reward function in each basis
# record compute times of eig+solve and solve

# do svd matrix inv and use k components for lstd comparison
# add mean component to A matrix in model
# force reward to be in initial sample
# autoencoder loss version - recursive linear, randomized nonlinear

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


# increase sample size incrementally
# crossvalidation
# run multiple experiments and plot avg and error bars
# calculate, plot reward and transition error 
# plot resulting reward and value functions
# hypderparameter optimization

# to vary: d, p, k


class LSTD_Experiment(object):
    
    def __init__(self,
                b = 50, # number of batches / iterations to run cg for
                p = 2000, # total number of samples used for learning
                n = 100, # dimension of each image
                m = 100, # number of samples per minibatch
                d = 400, # dimension of base feature representation
                k = 16,  # number of compressed features, W is [dxk]
                max_iter = 3, # max number of cg optimizer steps per iteration
                max_h = 1000, # max horizon
                l2_lstd = 1e-15,
                l2_subs = 1e-15,
                reg_loss = ('l2',1e-3),
                gam = 1-1e-2,
                world = 'torus',
                ):

        self.b = b
        self.p = p
        self.n = n
        self.m = m
        self.d = d
        self.k = k
        self.max_iter = max_iter

        self.world = rsis.TorusWorld() if world is 'torus' else rsis.CircleWorld(gam=gam)
        self.fmap = rsis.TileFeatureMap(d)
        
        N = int(numpy.sqrt(d))
        P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
        self.Im = self.fmap.transform(P) # generate a regular grid for plotting
        self.it = 0

        self.timestamp = str(datetime.datetime.now()).replace(' ','.').replace(':','.')
        self.loss_values = []
        self.loss_df = None

        self.X_test, self.R_test = self.sample_world(p) #, seed=0)
        X_data, R_data = self.sample_world(p)
        self.model = rsis.Reconstructive_LSTD(k, d, gam, 
                                    states=X_data, 
                                    rewards=R_data, 
                                    m=m, 
                                    n=n, 
                                    reg_loss=reg_loss,
                                    l2_lstd=l2_lstd,
                                    l2_subs=l2_subs,
                                    max_h=max_h) 


    def sample_world(self, n, seed=None):
        P, R = self.world.get_samples(n, seed=seed)
        X = self.fmap.transform(P)
        return X, R

    def plot_learned(self):
        # plot a regular grid
        logger.info('plotting current features')
        Z = self.model.encode(self.Im)[:,::-1]
        l = int(numpy.sqrt(self.k))         
        tup = (l,self.k//l) if self.k > 2 else (1,self.k)
        plot_filters(Z, tup, 'plots/learned_basis_.%i.png' % self.it)

    def plot_filters(self, X, (n_sp1, n_sp2), file_name = 'basis.png', last = False):
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


    def evaluate(self, x_test=None, r_test=None):
        
        x_test = x_test if x_test else self.X_test
        r_test = r_test if r_test else self.R_test

        model_tderr = self.model.td_error(self.model.get_model_value(x_test), r_test)
        lstd_tderr = self.model.td_error(self.model.get_lstd_value(x_test), r_test) 
        
        #model_rerr = self.model.reward_error(x_test, r_test) / n
        #model_zerr = self.model.transition_error(x_test) / (n-1)
        
        logger.info('h/o sample model td error: %05f' % model_tderr)
        logger.info('h/o sample lstd td error: %05f' % lstd_tderr)

        logger.info('norm of model weights: %05f' % numpy.sum(numpy.dot(self.model.W, self.model.u)**2))    
        logger.info('norm of lstd weights: %05f' % numpy.sum(self.model.w**2)) 

        self.loss_values.append({'model-td': model_tderr, 
                                 'lstd-td': lstd_tderr,
                                 'iter': self.it })
        self.loss_df = None # reset loss df to be rebuilt

    
    def train_model(self, b = None, evaluate = False):

        b = b if b else self.b
        try:
            for i in xrange(b):
                
                logger.info('*** iteration ' + str(self.it) + '***')
                
                X, R = self.model.sample_minibatch() 
                    
                self.model.set_params(
                    scipy.optimize.fmin_cg(
                        self.model.optimize_loss,
                        self.model.W.flatten(),
                        self.model.optimize_grad,
                        args=(X, R),
                        full_output=False,
                        maxiter=self.max_iter
                        )
                    )

                if evaluate:
                    self.evaluate()

                self.it += 1
                 
        except KeyboardInterrupt:
            logger.info('\n user stopped current training loop')

    def save_csv(self):
        if self.loss_df is None:
            self.loss_df = pandas.DataFrame(self.loss_values)
        self.loss_df.to_csv('data/losses%s.csv' % self.timestamp)

    def plot_loss_curve(self):
        
        self.loss_df = pandas.DataFrame(self.loss_values)

        # XXX add line for td error of the true value function 
        plt.clf()
        n_cols = len(self.loss_df.columns)
        for i, c in enumerate(self.loss_df.columns):
            if not (c == 'iter'):
                y = self.loss_df[c].values
                #plt.subplot(n_cols, 1, i)
                plt.plot(numpy.arange(1, len(y)+1), y, label = c)

        #for i in xrange(n_cols):
            #plt.subplot(n_cols, 1, i)
        plt.legend()        
        
        plt.savefig('plots/loss-curve%s.png' % self.timestamp)


def main(
        b = 10, # number of batches / iterations to run cg for
        p = 2000, # total number of samples used for learning
        n = 100, # dimension of each image
        m = 100, # number of samples per minibatch
        d = 1000, # dimension of base feature representation
        k = 16,  # number of compressed features, W is [dxk]
        max_iter = 3, # max number of cg optimizer steps per iteration
        max_h = 1000, # max horizon
        l2_lstd = 1e-15,
        l2_subs = 1e-15,
        reg_loss = ('l2',1e-3),
        gam = 1-1e-2,
        world = 'torus',
        ):
    
    logger.info("building experiment and model objects")
    experiment = LSTD_Experiment(
                             b=b,p=p,n=n,m=m,d=d,k=k,
                             max_iter=max_iter, max_h=max_h,
                             l2_lstd=l2_lstd,l2_subs=l2_subs,reg_loss=reg_loss,
                             gam=gam, world=world
                             )

    logger.info("training model")
    for i in xrange(5):
        logger.info("batch %i" % i)
        experiment.train_model(evaluate=True)
        experiment.plot_learned()
    
    logger.info("processing results")
    experiment.save_csv()
    experiment.plot_loss_curve()


    #def plot_value_rew():
    #    
    #    P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
    #    X = fmap.transform(P)
    #    R = world.reward_func(P) # true reward function
    #    
    #    model_val = model.get_model_value(X) 
    #    lstd_val = model.get_lstd_value(X)
    #    
    #    #model_rew = model.reward_func(X)
    #    value_list = [model_val, lstd_val]
    #    
    #    if hasattr(world, 'value_func'):
    #        value_list.append(world.value_func(P))

    #    #plot_filters(numpy.vstack([model_rew, lstd0_rew, lstd1_rew, R]).T, (1, 4), file_name='plots/rewards%s.png' % timestamp)
    #    plot_filters(numpy.vstack(value_list).T, (1, len(value_list)), file_name='plots/values%s.png' % timestamp)

    #plot_value_rew()    

def test_loss_working(k = 4, d = 400, n=1000, gam=1-1e-2):

    world = rsis.CircleWorld(gam = gam)
    P, R_data = world.get_samples(n)

    fmap = rsis.TileFeatureMap(d)
    X_data = fmap.transform(P)

    model = rsis.Reconstructive_LSTD(k, d, gam, states=X_data, rewards=R_data) 
    
    X, R = model.sample_minibatch() 

    print 'optimization loss: ', model.optimize_loss(model.W.flatten(), X, R)
    print 'optimization grad norm: ', numpy.sum(model.optimize_grad(model.W.flatten(), X, R)**2)
    

if __name__ == '__main__':
    #view_fourier_basis()
    #test_loss_working()
    rsis.script(main)
