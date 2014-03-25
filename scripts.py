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
from rsis.experiments import LSTD_Experiment, PostProcess, tran_funcs

logger = rsis.get_logger(__name__)

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

 
def train_model_experiment(
        b = 20, # number of batches / iterations to run cg for
        p = 2000, # total number of samples used for learning
        n = 200, # dimension of each image
        m = 200, # number of samples per minibatch
        d = 100, # dimension of base feature representation
        k = (64,16),  # number of compressed features, U is [dxk]
        max_iter = 3, # max number of cg optimizer steps per iteration
        max_h = 1000, # max horizon
        l2_lstd = 1e-15, # reg used for full lstd
        l2_subs = 1e-4, # reg used for subspace lstd
        reg_loss = ('l1', 1e-4),
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
