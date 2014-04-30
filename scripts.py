import os
import glob
import cPickle as pickle
import numpy
import scipy.optimize
import plac
import rsis
from rsis.experiments import LSTD_Experiment, PostProcess, tran_funcs


logger = rsis.get_logger(__name__)

# plot both last and best features
# plot iteration used (Ubest);
# untied weights
# autoencoder as score matching - what distribution?
# profile performance
# add exponentially decaying weights on horizon losses
# freeze weights while learning initial recon bias?
# plot learned bias
# try giving larger dataset to fmin (no loop around fmin)

# create function to map for random hyperparameter search
# grid features
# td network with same architecture, similar samples / training
# gold standard value func
# td lambda error
# normalize transition error by feature size
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
        p=5000,  # total number of samples used for learning
        n=300,  # dimension of each image
        m=200,  # number of samples per minibatch
        d=100,  # dimension of base feature representation
        k=(24,16),  # number of compressed features, U is [dxk]
        max_iter=3,  # max number of cg optimizer steps per iteration
        max_h=2000,  # max horizon
        l2_lstd=1e-12,  # reg used for full lstd
        l2_subs=1e-12,  # reg used for subspace lstd
        reg_loss=('l2', 1e-8),
        init_scale=1e-3,
        gam=1-1e-2,
        world='torus',
        seed=0,
        tran_func='sigmoid',
        optimizer=scipy.optimize.fmin_cg,  # fmin_l_bfgs_b, fmin_bfgs, fmin_ncg, fmin_cg
        model=rsis.StatespaceRLSTD,
        eval_freq=1,
        samp_dist= 'unif',  # sample distribution for images (geom or unif)
        patience=100,
        max_training_steps = 100,
        tied_weights = False,
):

    os.system('rm plots/*.png')
    g = tran_funcs[tran_func]

    n_layers = len(k)
    output = None

    for jj in xrange(n_layers):

        logger.info("training layer %i" % (jj + 1))

        logger.info("building experiment and model objects")

        # TODO add initial bias values
        # initialize weights to previous shallower model if jj > 0
        Uinit = experiment.model.U if jj else []
        bias_layer = experiment.model.bias_layer if jj else []
        bias_recon = experiment.model.bias_recon if jj else None

        Uinit += [None] * (n_layers - jj)
        bias_layer += [None] * (n_layers - jj)

        experiment = LSTD_Experiment(
            p=p, n=n, m=m, d=d, k=k[:jj + 1],
            max_iter=max_iter, max_h=max_h,
            l2_lstd=l2_lstd, l2_subs=l2_subs, reg_loss=reg_loss,
            gam=gam, world=world, seed=seed, g=g,
            model=model,
            Uinit=Uinit,
            bias_layer=bias_layer,
            bias_recon=bias_recon,
            init_scale=init_scale,
            tied_weights=tied_weights,
        )

        if output:  # add losses from the previous iterations
            experiment.output.loss_values = output.loss_values

        logger.info("training and evaluating model")

        loss_best = 1e16
        it = 0
        wait = 0

        try:
            while (wait < patience) and (it < max_training_steps):

                logger.info("batch %i" % it)
                # optimize the model
                experiment.train_model(eval_freq=eval_freq,
                                       optimizer=optimizer,
                                       samp_dist=samp_dist)

                # measure rsis loss after learning step
                loss_new = experiment.model.theano_loss(*experiment.model.get_params() +
                                                        list(experiment.model.sample_minibatch(samp_dist, seed)))
                it += 1

                if loss_new < loss_best:
                    wait = 0
                    Ubest = experiment.model.get_flat_params()
                    loss_best = loss_new
                    logger.info("new best loss: %04f" % loss_best)
                else:
                    wait += 1
                    logger.info('converging (wait, patience): (%i, %i)' % (wait, patience))

                logger.info("best loss: %04f" % loss_best)
                logger.info("current loss: %04f" % loss_new)

        except KeyboardInterrupt:
            logger.info('\n user stopped current training loop')

        experiment.set_model_params(Ubest)
        output = experiment.output  # to pass output onto next layer model

    logger.info("pickling results")
    with open('data/experiment%s.pkl' % experiment.output.timestamp, 'wb') as pfile:
        pickle.dump(experiment.output, pfile)

    return experiment, model


def post_process(experiment=None, world='torus', model=rsis.StatespaceRLSTD):
    if experiment is None:
        paths = glob.glob('data/*.pkl')
        paths.sort()
        exp_path = paths[-1]

        logger.info("using the pickled output file at: %s" % exp_path)
        with open(exp_path, 'rb') as pfile:
            output = pickle.load(pfile)

        logger.info("updating the output and model objects")
        output.model_kwargs.update(
            {'g': tran_funcs[output.model_kwargs['g']]})  # swap output.g (string) for the tuple including functions
        model = model(*output.model_args, **output.model_kwargs)
        model.set_params(output.model_params)  # XXX learned params currently not taken by constructor

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
    # os.system('rm plots/*.png')

    postproc.plot_learned()  # plot final learned features

    # plot learning curves for the transition, td, and reward error
    postproc.plot_reward_err()
    postproc.plot_transition_err()
    postproc.plot_cosine_err()
    for stepsize in LSTD_Experiment.td_steps:
        postproc.plot_td_err(n=stepsize)

    postproc.plot_value_rew()  # plot value and reward functions


# (help, kind, abbrev, type, choices, metavar)
@plac.annotations(
    trainmodel=("boolean for whether training script should be run", 'option', None, int),
    postprocess=("boolean for whether postprocessing should be run", 'option', None, int)
)
def main(trainmodel=1, postprocess=1, world='torus'):
    if trainmodel:
        logger.info("training new model")
        experiment, model = batch_experiment(world=world)
    else:
        experiment = None

    if postprocess:
        logger.info("postprocessing / plotting")
        post_process(experiment=experiment, world=world, model=model)


if __name__ == '__main__':
    rsis.script(main)
