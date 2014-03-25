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
import glob
import plac
import cPickle as pickle #cloud.serialization.cloudpickle as pickle

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


# XXX data object for storing inputs?
class RL_Output(object):

    def __init__(self, k, d, fmap, model):
        
        self.k = k
        self.d = d
        self.loss_values = []
        self.model_params = model.get_params()
        self.timestamp = str(datetime.datetime.now()).replace(' ','.').replace(':','.')

        self.model_args = [model.k, model.d, model.gam]
        self.model_kwargs = {'states':model.states,
                        'rewards':model.rewards,
                        'm':model.m,
                        'n':model.n,
                        'reg_loss':model.reg_loss,
                        'l2_lstd':model.l2_lstd,
                        'l2_subs':model.l2_subs,
                        'max_h':model.h,
                        'g':model.g_str,
                        }

        self.model_stats = {
                            'S':model.S,
                            'Sp':model.Sp,
                            'b':model.b,
                            'ntot':model.ntot,
                           } 

        N = int(numpy.sqrt(d))
        self.grid = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
        self.Im = fmap.transform(self.grid) # generate a regular grid for plotting


class LSTD_Experiment(object):
    
    def __init__(self,
                p = 2000, # total number of samples used for learning
                n = 200, # dimension of each image
                m = 200, # number of samples per minibatch
                d = 200, # dimension of base feature representation
                k = (20,),  # number of compressed features, U is [dxk]
                max_iter = 3, # max number of cg optimizer steps per iteration
                max_h = 1000, # max horizon
                l2_lstd = 1e-15,
                l2_subs = 1e-15,
                reg_loss = ('l1',1e-6),
                gam = 1-1e-2,
                world = 'torus',
                seed = None,
                g = None,
                optimizer = scipy.optimize.fmin_cg,
                model = rsis.ProjectedRLSTD,
                Uinit = None,
                ):

        self.p = p
        self.n = n
        self.m = m
        self.d = d
        self.k = k
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.td_steps = [1,2,4] # XXX hardcoded here        

        self.world = rsis.TorusWorld() if world is 'torus' else rsis.CircleWorld(gam=gam)
        self.fmap = rsis.TileFeatureMap(d) # XXX seed for tile features
        
        self.it = 0
        
        self.x_test, self.r_test = self.sample_world(p, seed=seed)
        x_data, r_data = self.sample_world(p, seed+1 if seed else None)

        logger.info("average test reward: " + str(numpy.mean(self.r_test)))
        logger.info("average train reward: " + str(numpy.mean(r_data)))
        
        if Uinit is None: Uinit = [None]*len(k)

        self.model = model(k, d, gam,   
                        states=x_data, 
                        rewards=r_data, 
                        m=m, 
                        n=n, 
                        reg_loss=reg_loss,
                        l2_lstd=l2_lstd,
                        l2_subs=l2_subs,
                        max_h=max_h,
                        g=g,
                        Uinit=Uinit) 

        self.output = RL_Output(self.k, self.d, self.fmap, self.model)

    def resample_model_data(self, seed = None):
        x_data, r_data = self.sample_world(self.p, seed)
        self.model.set_data(x_data, r_data) # set the data set subsampled for rica loss

    def sample_world(self, n, seed=None):
        P, R = self.world.get_samples(n, seed=seed)
        X = self.fmap.transform(P)
        return X, R

    def evaluate_lstd(self, x_test=None, r_test=None, k = None):
        
        x_test = x_test if x_test else self.x_test
        r_test = r_test if r_test else self.r_test
        
        tderr = [self.model.td_error(self.model.get_lstd_value(x_test, k=k), r_test, n = i) for i in self.td_steps]
        rerr = self.model.lstd_reward_error(x_test, r_test, k=k)
        zerr = self.model.lstd_transition_error(x_test, k=k)

        logger.info('h/o sample %s lstd td error (short): %05f' % ('k ' if k else '', tderr[0]))
        logger.info('h/o sample %s lstd td error (long): %05f' % ('k ' if k else '', tderr[-1]))
        logger.info('norm of lstd weights: %05f' % numpy.sum(self.model.w_k**2) if k else numpy.sum(self.model.w**2)) 
        
        pref = str(self.k[-1]) + ('svd-' if k else '')
        d = dict([(pref + 'lstd-td-' + str(self.td_steps[i]), tderr[i]) for i in xrange(len(self.td_steps))])
        d.update({pref + 'lstd-rerr': rerr,
                 pref + 'lstd-trans': zerr,
                 'iter': self.it})
        self.output.loss_values.append(d)
        self.output._loss_df = None # reset loss df to be rebuilt


    def evaluate_model(self, X, R, Y, x_test=None, r_test=None, lamb=1.):
        
        X_test, R_test, Y_test = self.model.sample_minibatch() # X here is a 3-tensor, R is a matrix

        x_test = x_test if x_test else self.x_test
        r_test = r_test if r_test else self.r_test # x here is a matrix, r is a vector

        train_loss = numpy.sum(self.model.theano_loss(*(self.model.U + [X, R, Y]))) # sum is just to make the output a scalar, not an array
        test_loss = numpy.sum(self.model.theano_loss(*(self.model.U + [X_test, R_test, Y_test]))) # test loss is 

        tderr = [self.model.td_error(self.model.get_model_value(x_test),r_test,n=i) for i in self.td_steps]  
        rerr = self.model.model_reward_error(x_test, r_test) 
        zerr = self.model.model_transition_error(x_test)
        
        logger.info('h/o sample model td error (short): %05f' % tderr[0])
        logger.info('h/o sample model td error (long): %05f' % tderr[-1])
        logger.info('norm of model weights: %05f' % numpy.sum(self.model.wz**2)) 
        
        pref = str(self.k[-1])
        d = dict([(pref + 'model-td-' + str(self.td_steps[i]), tderr[i]) for i in xrange(len(self.td_steps))])
        d.update({pref+'model-train-loss':train_loss,
                 pref+'model-test-loss':test_loss,
                 pref+'model-rerr': rerr,
                 pref+'model-trans': zerr,
                 'iter': self.it})
        self.output.loss_values.append(d)
        self.output._loss_df = None # reset loss df to be rebuilt

    
    def train_model(self,
                    eval_freq = 1, 
                    resample_freq = None, 
                    optimizer = scipy.optimize.fmin_l_bfgs_b):
     
        logger.info('*** training iteration ' + str(self.it) + '***')

        if resample_freq:
            if (self.it % resample_freq == 0):
                self.resample_model_data()
        
        X, R, Y = self.model.sample_minibatch() 

        if eval_freq:
            if (self.it % eval_freq == 0):
                self.evaluate_model(X, R, Y)
                self.evaluate_lstd()
                self.evaluate_lstd(k=self.k[-1])  

        if optimizer is scipy.optimize.fmin_l_bfgs_b:
            
            self.model.set_params(
                optimizer( 
                    self.model.optimize_loss,
                    self.model.get_flat_params(), 
                    self.model.optimize_grad,
                    args=(X, R, Y),
                    maxiter=self.max_iter
                    )[0]
                )
            
        else: 

            self.model.set_params(
                optimizer( 
                    self.model.optimize_loss,
                    self.model.get_flat_params(), 
                    self.model.optimize_grad,
                    args=(X, R, Y),
                    full_output=False,
                    maxiter=self.max_iter
                    )
                )

        self.it += 1
        self.output.model_params = self.model.get_params()



class PostProcess(object):

    def __init__(self, rl_output, model, world):
        self.output = rl_output
        self.model = model
        self.world = world
        self._loss_df = None

    @property
    def loss_df(self): 
        if self._loss_df is None:   
            self._loss_df = pandas.DataFrame(self.output.loss_values)
            self._loss_df.sort('iter', inplace=True)
        
        return self._loss_df

    @property
    def timestamp(self):
        return self.output.timestamp

    @property
    def Im(self):
        return self.output.Im

    @property
    def grid(self):
        return self.output.grid

    @property
    def k(self):
        return self.output.k[-1]

    def save_csv(self):
        self.loss_df.to_csv('data/losses%s.csv' % self.output.timestamp)

    def plot_loss_curve(self):
        
        # XXX add line for td error of the true value function 
        plt.clf()
        for i, c in enumerate(self.loss_df.columns):
            
            # remove all nans
            y = self.loss_df[c].dropna().values
            n = len(y)
            #non_nan = numpy.invert(numpy.isnan(y))
            #ids = non_nan.nonzero()[0] # corresponding iteration numbers
            
            if len(y) == 1: 
                assert False
                #y = y * numpy.ones(n-2)
            
            ids = numpy.arange(n)
                
            if 'rerr' in c: 
                plt.subplot(4,1,1)
                plt.plot(ids, y, label = c)

            elif 'trans' in c:
                plt.subplot(4,1,2)
                plt.plot(ids, y, label = c)

            elif '-td' in c:
                plt.subplot(4,1,3)

                plt.plot(ids, y, label = c)
            
            elif 'model' in c:
                plt.subplot(4,1,4)
                plt.plot(ids, y, label = c)

        for i in xrange(4):
            plt.subplot(4, 1, i+1)
            plt.legend()        
        
        plt.savefig('plots/loss-curve%s.png' % self.timestamp)

    def plot_learned(self):
        # plot a regular grid
        logger.info('plotting current features')
        Z = self.model.encode(self.Im)[:,::-1]
        l = int(numpy.sqrt(self.k))         
        tup = (l, self.k//l) if self.k > 2 else (1, self.k)
        plot_filters(Z, tup, 'plots/learned_basis.png')

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


    def plot_value_rew(self):
        
        world_rew = self.world.reward_func(self.grid) # true reward function
        model_rew = self.model.get_model_reward(self.Im)
        lstd_rew = self.model.get_lstd_reward(self.Im)
        k_lstd_rew = self.model.get_lstd_reward(self.Im)
        
        model_val = self.model.get_model_value(self.Im) 
        lstd_val = self.model.get_lstd_value(self.Im)
        k_lstd_val = self.model.get_lstd_value(self.Im, k=self.k)
        
        value_list = [model_val, lstd_val, k_lstd_val]
        if hasattr(self.world, 'value_func'):
            value_list.append(self.world.value_func(self.grid))

        self.plot_filters(numpy.vstack([model_rew, lstd_rew, k_lstd_rew, world_rew]).T, (1, 4), file_name='plots/rewards%s.png' % self.output.timestamp)
        self.plot_filters(numpy.vstack(value_list).T, (1, len(value_list)), file_name='plots/values%s.png' % self.output.timestamp)

# add convergence check and delta theta printout
# gold standard value func
# td lambda error
# freeze first layer while converging with second, etc? (train layers: 1,2,1+2)
# catch timeout on ncg / bfgs
# smooth horizon diversity in minibatches
# seed for features (other random elements?)
# try contractive regularizer
# td network with same architecture, similar samples / training
# do crossvalidation and plot avg and error bars
# hypderparameter optimization
# qualitatively view reward and transition models
# plot reward image(s) and reconstruction 

# experiments:
# convergence on batch with increasing sample size
# - lstd, k-lstd, 1rsis, 1td0, 2rsis, 2td0 (tdlambda?)
# - run to convergence on hold out loss + delta param
# - [average, max, var] loss across initializations

# minibatch online training
# - 1rsis, 1td0, 2rsis, 2td0 (tdlambda?)
# - run for some max no. episodes
# - plot [average, max, var] loss across initializations no. reward samples seen, no. of resamples
# - how to choose minibatch size and no. of samples/images from it

tran_funcs = {
            'linear': (lambda x: x, lambda x: x, 'linear'),
            'relu': (lambda x: numpy.maximum(0,x), lambda x: TT.maximum(0,x), 'relu'),
            'softplus': (lambda x: numpy.log(1+numpy.exp(x)), TT.nnet.softplus, 'softplus'),
            'sigmoid': (lambda x: 1/(1+numpy.exp(x)), TT.nnet.sigmoid, 'sigmoid'),
            'tanh': (lambda x: numpy.tanh(x), TT.tanh, 'tanh'),
            }
 
def train_model_experiment(
        b = 20, # number of batches / iterations to run cg for
        p = 5000, # total number of samples used for learning
        n = 200, # dimension of each image
        m = 200, # number of samples per minibatch
        d = 100, # dimension of base feature representation
        k = (64,16),  # number of compressed features, U is [dxk]
        max_iter = 3, # max number of cg optimizer steps per iteration
        max_h = 1000, # max horizon
        l2_lstd = 1e-15, # reg used for full lstd
        l2_subs = 1e-6, # reg used for subspace lstd
        reg_loss = ('l1', 0.),
        gam = 1-1e-2,
        world = 'torus',
        seed = 0,
        tran_func = 'sigmoid',
        optimizer = scipy.optimize.fmin_l_bfgs_b, # fmin_l_bfgs_b, fmin_bfgs, fmin_ncg, fmin_cg
        model = rsis.StatespaceRLSTD,
        resample_freq = None,
        eval_freq = 1,
        ):
    
    g = tran_funcs[tran_func]

    os.system('rm plots/*.png')
    
    n_layers = len(k)
    output = None
    
    for jj in xrange(n_layers):
        
        logger.info("training layer %i" % (jj+1))
        
        logger.info("building experiment and model objects")
            
        # initialize weights to previous shallower model if jj > 0
        Uinit = experiment.model.U if jj else []
        Uinit += [None]*(n_layers-jj)
        
        experiment = LSTD_Experiment(
                                 p=p,n=n,m=m,d=d,k=k[:jj+1],
                                 max_iter=max_iter, max_h=max_h,
                                 l2_lstd=l2_lstd,l2_subs=l2_subs,reg_loss=reg_loss,
                                 gam=gam, world=world, seed=seed, g=g,
                                 model=model,
                                 Uinit=Uinit,
                                 )

        if output: # add lstd statistics and losses from the previous iterations
            experiment.model.__dict__.update(output.model_stats)
            experiment.output.loss_values = output.loss_values

        logger.info("training and evaluating model")
        try:
            for i in xrange(b):
                logger.info("batch %i" % i)
                experiment.train_model(eval_freq = eval_freq,
                                       resample_freq = resample_freq,
                                       optimizer=optimizer)

        except KeyboardInterrupt:
            logger.info('\n user stopped current training loop')

        output = experiment.output

    logger.info("pickling results")
    with open('data/experiment%s.pkl'%experiment.output.timestamp, 'wb') as pfile:
        pickle.dump(experiment.output, pfile)


def post_process(world = 'circle', model = rsis.StatespaceRLSTD):        
        
    paths = glob.glob('data/*.pkl')
    paths.sort()
    exp_path = paths[-1]

    logger.info("using the pickled output file at: %s" % exp_path)
    with open(exp_path, 'rb') as pfile:
        output = pickle.load(pfile)
    

    output.model_kwargs.update({'g': tran_funcs[output.model_kwargs['g']]}) # swap output.g (string) for the tuple including functions
    model = model(*output.model_args, **output.model_kwargs)
    model.set_params(output.model_params) # XXX learned params currently not taken by constructor

    world = rsis.TorusWorld() if world is 'torus' else rsis.CircleWorld(gam=output.model_args[2])

    postproc = PostProcess(output, model, world)
    postproc.save_csv()

    postproc.plot_loss_curve()
    postproc.plot_learned() # plot final
    postproc.plot_value_rew()

# (help, kind, abbrev, type, choices, metavar)
@plac.annotations(
trainmodel=("boolean for whether training script should be run", 'option', None, int),
postprocess=("boolean for whether postprocessing should be run", 'option', None, int)
)
def main(trainmodel = 1, postprocess = 1, world = 'torus',  model = rsis.ProjectedRLSTD):
    
    if trainmodel:
        logger.info("training new model")
        train_model_experiment(world=world, model=model)

    if postprocess:
        logger.info("postprocessing / plotting")
        post_process(world=world, model=model)
           

if __name__ == '__main__':
    rsis.script(main)
