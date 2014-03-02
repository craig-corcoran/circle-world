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
import theano
import theano.tensor as TT
import rsis

logger = rsis.get_logger(__name__)

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


class LSTD_Experiment(object):
    
    def __init__(self,
                p = 2000, # total number of samples used for learning
                n = 100, # dimension of each image
                m = 100, # number of samples per minibatch
                d = 500, # dimension of base feature representation
                k = 16,  # number of compressed features, U is [dxk]
                max_iter = 3, # max number of cg optimizer steps per iteration
                max_h = 1000, # max horizon
                l2_lstd = 1e-15,
                l2_subs = 1e-15,
                reg_loss = ('l2',1e-3),
                gam = 1-1e-2,
                world = 'torus',
                seed = None,
                g = (lambda x: x,)*2,
                ):

        self.p = p
        self.n = n
        self.m = m
        self.d = d
        self.k = k
        self.max_iter = max_iter

        self.world = rsis.TorusWorld() if world is 'torus' else rsis.CircleWorld(gam=gam)
        self.fmap = rsis.TileFeatureMap(d) # XXX seed for tile features
        
        N = int(numpy.sqrt(d))
        self.grid = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
        self.Im = self.fmap.transform(self.grid) # generate a regular grid for plotting
        self.it = 0

        self.timestamp = str(datetime.datetime.now()).replace(' ','.').replace(':','.')
        self.loss_values = []
        self._loss_df = None

        self.X_test, self.R_test = self.sample_world(p, seed=seed)
        X_data, R_data = self.sample_world(p, seed+1 if seed else None)

        logger.info("average test reward: " + str(numpy.mean(self.R_test)))
        logger.info("average train reward: " + str(numpy.mean(R_data)))
    
        self.model = rsis.Reconstructive_LSTD(k, d, gam, 
                                    states=X_data, 
                                    rewards=R_data, 
                                    m=m, 
                                    n=n, 
                                    reg_loss=reg_loss,
                                    l2_lstd=l2_lstd,
                                    l2_subs=l2_subs,
                                    max_h=max_h,
                                    g=g) 

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
        plot_filters(Z, tup, 'plots/learned_basis%i.png' % self.it)

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

    def evaluate_lstd(self, x_test=None, r_test=None, k = None):
        
        x_test = x_test if x_test else self.X_test
        r_test = r_test if r_test else self.R_test

        tderr = self.model.td_error(self.model.get_lstd_value(x_test, k=k), r_test) 
        rerr = self.model.lstd_reward_error(x_test, r_test, k=k)
        zerr = self.model.lstd_transition_error(x_test, k=k)

        logger.info('h/o sample lstd td error: %05f' % tderr)
        logger.info('norm of lstd weights: %05f' % numpy.sum(self.model.w_k**2)) 
        
        pref = 'k-' if k else ''
        self.loss_values.append({pref + 'lstd-td': tderr, 
                                 pref + 'lstd-rerr': rerr,
                                 pref + 'lstd-trans': zerr,
                                 'iter': self.it})

        self._loss_df = None # reset loss df to be rebuilt


    def evaluate_model(self, x_test=None, r_test=None):
        
        x_test = x_test if x_test else self.X_test
        r_test = r_test if r_test else self.R_test

        tderr = self.model.td_error(self.model.get_model_value(x_test), r_test)  
        rerr = self.model.model_reward_error(x_test, r_test) 
        zerr = self.model.model_transition_error(x_test)
        
        logger.info('h/o sample model td error: %05f' % tderr)
        logger.info('norm of model weights: %05f' % numpy.sum(numpy.dot(self.model.U, self.model.wz)**2))    

        self.loss_values.append({'model-td': tderr, 
                                 'model-rerr': rerr,
                                 'model-trans': zerr,
                                 'iter': self.it})

        self._loss_df = None # reset loss df to be rebuilt

    
    def train_model(self, evaluate = False):
     
        logger.info('*** training iteration ' + str(self.it) + '***')
        
        X, R = self.model.sample_minibatch() 
            
        self.model.set_params(
            scipy.optimize.fmin_cg(
                self.model.optimize_loss,
                self.model.U.flatten(),
                self.model.optimize_grad,
                args=(X, R),
                full_output=False,
                maxiter=self.max_iter
                )
            )

        if evaluate:
            self.evaluate_model()

        self.it += 1
                     
    @property
    def loss_df(self): 
        if self._loss_df is None:   
            self._loss_df = pandas.DataFrame(self.loss_values)
            self._loss_df.sort('iter', inplace=True)
        
        return self._loss_df

    def save_csv(self):
        self.loss_df.to_csv('data/losses%s.csv' % self.timestamp)

    def plot_loss_curve(self):
        
        # XXX add line for td error of the true value function 
        plt.clf()
        for i, c in enumerate(self.loss_df.columns):

            y = self.loss_df[c].values
            n = len(y)
            y = y[numpy.invert(numpy.isnan(y))]
            if len(y) == 1: 
                y = y * numpy.ones(n-2)
                
            if 'rerr' in c: 
                plt.subplot(3,1,1)
                plt.plot(numpy.arange(1, len(y)+1), y, label = c)

            elif 'trans' in c:
                plt.subplot(3,1,2)
                plt.plot(numpy.arange(1, len(y)+1), y, label = c)

            elif ('-td' in c) or ('model' in c):
                plt.subplot(3,1,3)
                plt.plot(numpy.arange(1, len(y)+1), y, label = c)

        for i in xrange(3):
            plt.subplot(3, 1, i+1)
            plt.legend()        
        
        plt.savefig('plots/loss-curve%s.png' % self.timestamp)

    def plot_value_rew():
        
        model_val = model.get_model_value(self.Im) 
        lstd_val = model.get_lstd_value(self.Im)
        k_lstd_val = model.get_lstd_value(self.Im, k=self.k)
        
        world_rew = world.reward_func(self.grid) # true reward function
        model_rew = model.reward_func(X)
        value_list = [model_val, lstd_val]
        
        if hasattr(world, 'value_func'):
            value_list.append(world.value_func(P))

        #plot_filters(numpy.vstack([model_rew, lstd0_rew, lstd1_rew, R]).T, (1, 4), file_name='plots/rewards%s.png' % timestamp)
        plot_filters(numpy.vstack(value_list).T, (1, len(value_list)), file_name='plots/values%s.png' % timestamp)

# show target loss (training and holdout data)
# plot resulting reward and value functions
# save experiment/model object after construction
# td network with same architecture, similar samples / training
# do crossvalidation and plot avg and error bars
# hypderparameter optimization
# plot distribution from rolling the learned/true model out through time

def main(
        b = 100, # number of batches / iterations to run cg for
        p = 2000, # total number of samples used for learning
        n = 200, # dimension of each image
        m = 200, # number of samples per minibatch
        d = 200, # dimension of base feature representation
        k = 20,  # number of compressed features, U is [dxk]
        max_iter = 3, # max number of cg optimizer steps per iteration
        max_h = 1000, # max horizon
        l2_lstd = 1e-15, # reg used for full lstd
        l2_subs = 1e-15, # reg used for subspace lstd
        reg_loss = ('l1',1e-3),
        gam = 1-1e-2,
        world = 'torus',
        seed = 0,
        tran_func = 'sigmoid'
        ):
    
    losses = {
        'relu': (lambda x: numpy.maximum(0,x), lambda x: TT.maximum(0,x)),
        'softplus': (lambda x: numpy.log(1+numpy.exp(x)), lambda x: TT.log(1+TT.exp(x))),
        'sigmoid': (lambda x: 1/(1+numpy.exp(x)), lambda x: 1/(1+TT.exp(x)))
        }
    g = losses[tran_func]

    os.system('rm plots/*.png')
    
    logger.info("building experiment and model objects")
    experiment = LSTD_Experiment(
                             p=p,n=n,m=m,d=d,k=k,
                             max_iter=max_iter, max_h=max_h,
                             l2_lstd=l2_lstd,l2_subs=l2_subs,reg_loss=reg_loss,
                             gam=gam, world=world, seed=seed, g=g,
                             )

    #experiment.evaluate_lstd()
    experiment.evaluate_lstd(k=k)    
    experiment.evaluate_model()
    #experiment.plot_learned() # plot initial

    logger.info("training and evaluating model")
    try:
        for i in xrange(b):
            logger.info("batch %i" % i)
            experiment.train_model(evaluate=True)

    except KeyboardInterrupt:
        logger.info('\n user stopped current training loop')

    logger.info("processing results")
    experiment.save_csv()
    experiment.plot_loss_curve()
    experiment.plot_learned() # plot final
    
    # XXX plot value and reward functions

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

if __name__ == '__main__':
    rsis.script(main)
