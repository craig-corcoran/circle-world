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

# grid features
# XXX learned features broken again - regularization?
# td network with same architecture, similar samples / training
# gold standard value func
# td lambda error
# freeze first layer while converging with second, etc? (train layers: 1,2,1+2)
# catch timeout on ncg / bfgs
# smooth horizon diversity in minibatches
# seed for features (other random elements?)
# try contractive regularizer
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

def batch_experiment(
                    p = 10000, # total number of samples used for learning
                    n = 200, # dimension of each image
                    m = 200, # number of samples per minibatch
                    d = 200, # dimension of base feature representation
                    k = (24,8),  # number of compressed features, U is [dxk]
                    max_iter = 3, # max number of cg optimizer steps per iteration
                    max_h = 1000, # max horizon
                    l2_lstd = 1e-12, # reg used for full lstd
                    l2_subs = 1e-12, # reg used for subspace lstd
                    reg_loss = ('l2', 1e-12),
                    gam = 1-1e-2,
                    world = 'torus',
                    seed = 0,
                    tran_func = 'sigmoid',
                    optimizer = scipy.optimize.fmin_l_bfgs_b, # fmin_l_bfgs_b, fmin_bfgs, fmin_ncg, fmin_cg
                    model = rsis.StatespaceRLSTD,
                    eval_freq = 1,
                    rel_imp = 1e-7,
                    patience = 5,
                    ):

    g = tran_funcs[tran_func]

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

        if output: # add losses from the previous iterations
            experiment.output.loss_values = output.loss_values

        logger.info("training and evaluating model")

        delta_loss_frac = 1e16
        delta_weig_frac = 1e16
        loss_best = 1e16
        Ubest = 1e16
        it = 0
        wait = 0

        try:
            while wait < patience:

                logger.info("batch %i" % it)

                # measure parameters before learning step
                Uold = experiment.model.get_flat_params()

                # optimize the model
                experiment.train_model(eval_freq=eval_freq,
                                       optimizer=optimizer)

                # measure loss and parameters after learning step
                Unew = experiment.model.get_flat_params()
                loss_new = experiment.model.td_error(experiment.model.get_model_value(experiment.x_test), experiment.r_test)

                # measure change in parameters and hold out loss
                dweights = numpy.linalg.norm(Ubest-Uold)
                delta_loss_frac = (loss_best-loss_new) / loss_best
                delta_weig_frac = dweights / numpy.linalg.norm(Ubest)

                logger.info('fraction change in loss: %04f' % delta_loss_frac)
                logger.info('fraction change in weights: %04f' % delta_weig_frac)
                logger.info('absolute change in weights: %04f' % dweights)

                it += 1
                if it == 1 or (delta_loss_frac > rel_imp): # or (delta_weig_frac > rel_imp)):
                    wait = 0
                    Ubest = Unew
                    loss_best = loss_new
                    logger.info("new best loss: %04f" % loss_best)
                else:
                    wait += 1
                    logger.info('converging (wait, patience): (%i, %i)' % (wait, patience))

        except KeyboardInterrupt:
            logger.info('\n user stopped current training loop')

        experiment.set_model_params(Ubest)
        output = experiment.output # to pass output onto next layer model

    logger.info("pickling results")
    with open('data/experiment%s.pkl' % experiment.output.timestamp, 'wb') as pfile:
        pickle.dump(experiment.output, pfile)

    return experiment


def post_process(experiment = None, world = 'torus', model = rsis.StatespaceRLSTD):

    if experiment is None:
        paths = glob.glob('data/*.pkl')
        paths.sort()
        exp_path = paths[-1]

        logger.info("using the pickled output file at: %s" % exp_path)
        with open(exp_path, 'rb') as pfile:
            output = pickle.load(pfile)


        logger.info("updating the output and model objects")
        output.model_kwargs.update({'g': tran_funcs[output.model_kwargs['g']]}) # swap output.g (string) for the tuple including functions
        model = model(*output.model_args, **output.model_kwargs)
        model.set_params(output.model_params) # XXX learned params currently not taken by constructor

        world = rsis.TorusWorld() if world is 'torus' else rsis.CircleWorld(gam=output.model_args[2])

    else:

        logger.info("using the output from given training run")
        model = experiment.model
        output = experiment.output
        world = experiment.world

    logger.info("model gamma: %04f" % model.gam)

    postproc = PostProcess(output, model, world)
    postproc.save_csv()

    # remove old plots
    os.system('rm plots/*.png')

    # plot learning curves for the transition, td, and reward error
    postproc.plot_reward_err()
    postproc.plot_transition_err()

    for stepsize in LSTD_Experiment.td_steps:
        postproc.plot_td_err(n=stepsize)

    postproc.plot_learned() # plot final learned features
    postproc.plot_value_rew() # plot value and reward functions

# (help, kind, abbrev, type, choices, metavar)
@plac.annotations(
trainmodel=("boolean for whether training script should be run", 'option', None, int),
postprocess=("boolean for whether postprocessing should be run", 'option', None, int)
)
def main(trainmodel = 1, postprocess = 1, world = 'torus',  model = rsis.ProjectedRLSTD):

    if trainmodel:
        logger.info("training new model")
        experiment = batch_experiment(world=world, model=model)
    else:
        experiment = None

    if postprocess:
        logger.info("postprocessing / plotting")
        post_process(experiment=experiment, world=world, model=model)


if __name__ == '__main__':
    rsis.script(main)
