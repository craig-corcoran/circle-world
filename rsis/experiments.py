import os
import time
import datetime
import copy
import collections
import glob
import cPickle as pickle  #cloud.serialization.cloudpickle as pickle

import numpy
import pandas
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import theano
import theano.tensor as TT
import plac

import rsis


logger = rsis.get_logger(__name__)

tran_funcs = {
    'linear': (lambda x: x, lambda x: x, 'linear'),
    'relu': (lambda x: numpy.maximum(0, x), lambda x: TT.maximum(0, x), 'relu'),
    'softplus': (lambda x: numpy.log(1 + numpy.exp(x)), TT.nnet.softplus, 'softplus'),
    'sigmoid': (lambda x: 1 / (1 + numpy.exp(x)), TT.nnet.sigmoid, 'sigmoid'),
    'tanh': (lambda x: numpy.tanh(x), TT.tanh, 'tanh'),
}


def plot_filters(X, (n_sp1, n_sp2), file_name='basis.png', last=False, scale=True):
    plt.clf()
    side = numpy.sqrt(X.shape[0])
    gs = gridspec.GridSpec(n_sp1, n_sp2)
    gs.update(wspace=0., hspace=0.)

    for i in xrange(min(n_sp1 * n_sp2, X.shape[1])):
        ax = plt.subplot(gs[i])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = X[:, -(i + 1)] if last else X[:, i]
        plt.imshow(
            numpy.reshape(im, (side, side)),
            cmap='gray',
            vmin=X.min() if scale else None,
            vmax=X.max() if scale else None,
            interpolation='nearest')
    plt.savefig(file_name)


# XXX data object for storing inputs?
class RL_Output(object):
    def __init__(self, k, d, fmap, model):
        self.k = k
        self.d = d
        self.loss_values = []
        self.model_params = model.get_params()
        self.timestamp = str(datetime.datetime.now()).replace(' ', '.').replace(':', '.')

        self.model_args = [model.k, model.d, model.gam]
        self.model_kwargs = {'states': model.states,
                             'rewards': model.rewards,
                             'm': model.m,
                             'n': model.n,
                             'reg_loss': model.reg_loss,
                             'l2_lstd': model.l2_lstd,
                             'l2_subs': model.l2_subs,
                             'max_h': model.h,
                             'g': model.g_str,
        }

        self.model_stats = {
            'S': model.S,
            'Sp': model.Sp,
            'b': model.b,
            'ntot': model.ntot,
        }

        N = int(numpy.sqrt(d))
        self.grid = numpy.reshape(numpy.mgrid[-1:1:N * 1j, -1:1:N * 1j], (2, N * N)).T
        self.Im = fmap.transform(self.grid)  # generate a regular grid for plotting


class LSTD_Experiment(object):
    td_steps = [1, 4, 8, 64]  # XXX hardcoded here

    def __init__(self,
                 p=2000,  # total number of samples used for learning
                 n=200,  # dimension of each image
                 m=200,  # number of samples per minibatch
                 d=200,  # dimension of base feature representation
                 k=(20,),  # number of compressed features, U is [dxk]
                 max_iter=3,  # max number of cg optimizer steps per iteration
                 max_h=1000,  # max horizon
                 l2_lstd=1e-15,
                 l2_subs=1e-15,
                 reg_loss=('l1', 1e-6),
                 gam=1 - 1e-2,
                 world='torus',
                 seed=None,
                 g=None,
                 optimizer=scipy.optimize.fmin_cg,
                 model=rsis.ProjectedRLSTD,
                 Uinit=None,
                 init_scale=1e-4,
    ):

        self.p = p
        self.n = n
        self.m = m
        self.d = d
        self.k = k
        self.max_iter = max_iter
        self.optimizer = optimizer

        self.world = rsis.TorusWorld() if world is 'torus' else rsis.CircleWorld(gam=gam)
        self.fmap = rsis.TileFeatureMap(d)  # XXX seed for tile features

        self.it = 0

        self.x_test, self.r_test = self.sample_world(p, seed=seed)
        x_data, r_data = self.sample_world(p, seed + 1 if seed else None)

        logger.info("average test reward: " + str(numpy.mean(self.r_test)))
        logger.info("average train reward: " + str(numpy.mean(r_data)))

        if Uinit is None: Uinit = [None] * len(k)

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
                           Uinit=Uinit,
                           init_scale=init_scale)

        self.output = RL_Output(self.k, self.d, self.fmap, self.model)
        self.d_tot = len(self.model.get_flat_params())

    # XXX revisit output object and serialization design 

    def set_model_params(self, U):
        self.model.set_params(U)
        self.output.model_params = self.model.get_params()

    def resample_model_data(self, seed=None):
        x_data, r_data = self.sample_world(self.p, seed)
        self.model.set_data(x_data, r_data)  # set the data set subsampled for rica loss

    def sample_world(self, n, seed=None):
        P, R = self.world.get_samples(n, seed=seed)
        X = self.fmap.transform(P)
        return X, R

    def evaluate_lstd(self, x_test=None, r_test=None, k=None):

        x_test = x_test if x_test else self.x_test
        r_test = r_test if r_test else self.r_test

        tderr = [self.model.td_error(self.model.get_lstd_value(x_test, k=k), r_test, n=i) for i in self.td_steps]
        rerr = self.model.lstd_reward_error(x_test, r_test, k=k)
        zerr = self.model.lstd_transition_error(x_test, k=k)

        logger.info('h/o sample %s lstd td error (short): %05f' % ('k ' if k else '', tderr[0]))
        logger.info('h/o sample %s lstd td error (long): %05f' % ('k ' if k else '', tderr[-1]))
        logger.info('norm of lstd weights: %05f' % numpy.sqrt(numpy.sum(self.model.w_k**2)) if k else numpy.sum(self.model.w**2))

        pref = str(self.k[-1]) + ('svd-' if k else '')
        d = dict([(pref + 'lstd-td-%02i' % self.td_steps[i], tderr[i]) for i in xrange(len(self.td_steps))])
        d.update({pref + 'lstd-rerr': rerr,
                  pref + 'lstd-trans': zerr,
                  'iter': self.it})
        self.output.loss_values.append(d)
        self.output._loss_df = None  # reset loss df to be rebuilt


    def evaluate_model(self, X, R, Y, x_test=None, r_test=None):

        x_test = x_test if x_test else self.x_test
        r_test = r_test if r_test else self.r_test  # x here is a matrix, r is a vector

        # sample input to theano functions
        X_test, R_test, Y_test = self.model.sample_minibatch()  # X here is a 3-tensor, R is a matrix

        train_loss = numpy.sum(self.model.theano_loss(*(self.model.get_params() + [X, R, Y])))  # sum is just to make the output a scalar, not an array
        test_loss = numpy.sum(self.model.theano_loss(*(self.model.get_params() + [X_test, R_test, Y_test])))  # test loss is

        train_data_loss = numpy.sum(self.model.data_loss_fn(*(self.model.get_params() + [X, R, Y])))  # sum is just to make the output a scalar, not an array
        train_reg_loss = numpy.sum(self.model.reg_loss_fn(*(self.model.get_params() + [X, R, Y])))  # sum is just to make the output a scalar, not an array

        rerr = self.model.model_reward_error(x_test, r_test)
        zerr = self.model.model_transition_error(x_test)
        tderr = [self.model.td_error(self.model.get_model_value(x_test), r_test, n=i) for i in self.td_steps]
        r_mag = numpy.linalg.norm(r_test) / len(r_test)

        # measure sum of cosine distances
        z = self.model.encode(x_test) # [n,k]
        inprod = numpy.dot(z.T, z)
        norms = numpy.linalg.norm(z, axis=0)
        cos_dist_loss = numpy.sum(abs(inprod / numpy.outer(norms, norms))) / numpy.prod(z.shape)

        logger.info('cosine distance norm: %04f' % cos_dist_loss)
        logger.info('h/o sample model td error (short): %05f' % tderr[0])
        logger.info('h/o sample model td error (long): %05f' % tderr[-1])
        logger.info('h/o sample reward magnitude : %05f' % r_mag)
        logger.info('training loss: %04f' % train_loss)
        logger.info('test loss: %04f' % test_loss)
        logger.info('data loss: %04f' % train_data_loss)
        logger.info('reg loss : %04f, (%s, %04f)' % ((train_reg_loss,) + self.model.reg_loss))
        logger.info('weight norm: %04f' % numpy.linalg.norm(self.model._flatten(self.model.U)))
        logger.info('bias norm: %04f' % numpy.linalg.norm(self.model.get_params()[-1]))
        logger.info('model value function weights norm: %05f' % numpy.sqrt(numpy.sum(self.model.wz**2)))

        pref = str(self.k[-1])
        d = dict([(pref + 'model-td-%02i' % self.td_steps[i], tderr[i]) for i in xrange(len(self.td_steps))])
        d.update({pref + 'model-train-loss': train_loss,
                  pref + 'model-test-loss': test_loss,
                  pref + 'model-rerr': rerr,
                  pref + 'model-trans': zerr,
                  pref + 'model-data-loss': train_data_loss,
                  pref + 'model-reg-loss': train_reg_loss,
                  pref + 'model-cos-loss': cos_dist_loss,
                  'r-mag': r_mag,
                  'iter': self.it})
        self.output.loss_values.append(d)
        self.output._loss_df = None  # reset loss df to be rebuilt

    def train_model(self,
                    eval_freq=1,
                    resample_freq=None,
                    optimizer=scipy.optimize.fmin_l_bfgs_b,
                    samp_dist='geom',
                    gtol=1e-8,
                    ):

        logger.info('*** training iteration ' + str(self.it) + '***')

        params_before = self.model.get_flat_params()

        # TODO remove
        #noise_scal = 1e-2
        if self.it in xrange(3):
            postproc = PostProcess(self.output, self.model, self.world)
            postproc.plot_learned('%02i' % self.it)

            # add noise to parameters
            #self.model.set_params(self.model.get_flat_params() + noise_scal * numpy.random.randn(self.d_tot))



        if resample_freq:
            if self.it % resample_freq == 0:
                self.resample_model_data()

        X, R, Y = self.model.sample_minibatch(samp_dist=samp_dist)

        logger.info("magnitude of gradient at current param values (before): %04f" %
                    numpy.linalg.norm(self.model.optimize_grad(self.model.get_flat_params(), X, R, Y)))

        if eval_freq:
            if (self.it % eval_freq == 0):
                self.evaluate_lstd()
                self.evaluate_lstd(k=self.k[-1])
                self.evaluate_model(X, R, Y)

        if optimizer is scipy.optimize.fmin_l_bfgs_b:

            self.model.set_params(
                optimizer(
                    self.model.optimize_loss,
                    self.model.get_flat_params(),
                    self.model.optimize_grad,
                    args=(X, R, Y),
                    maxiter=self.max_iter,
                    pgtol=gtol,
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
                    maxiter=self.max_iter,
                    gtol=gtol,
                )
            )

        self.it += 1
        self.output.model_params = self.model.get_params()

        params_after = self.model.get_flat_params()

        # # TODO remove: only move fraction of the direction suggested by the optimizer
        # frac = 1
        # self.model.set_params(params_before + frac * (params_after - params_before))

        logger.info("change in weight norm: %04f" % numpy.linalg.norm(params_after-params_before))
        logger.info("magnitude of gradient at current param values (after): %04f" %
                    numpy.linalg.norm(self.model.optimize_grad(self.model.get_flat_params(), X, R, Y)))

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

    def plot_loss_curve(self, filt_str, file_str):

        plt.clf()
        for c in [a for a in self.loss_df.columns if filt_str in a]:
            for j, h in enumerate(self.model.k):
                if str(h) in c:
                    y = self.loss_df[c].dropna().values
                    ids = numpy.arange(len(y))
                    plt.subplot(len(self.model.k), 1, j + 1)
                    plt.plot(ids, y, label=c)

        for j in xrange(len(self.model.k)):
            plt.subplot(len(self.model.k), 1, j + 1)
            plt.legend()

        plt.savefig('plots/%s.%s.png' % (file_str, self.timestamp))

    def plot_reward_err(self):
        self.plot_loss_curve('rerr', 'reward-loss')

    def plot_transition_err(self):
        self.plot_loss_curve('trans', 'transition-loss')

    def plot_cosine_err(self):
        self.plot_loss_curve('cos', 'cosine-loss')

    def plot_td_err(self, n=1):
        self.plot_loss_curve('-td-%02i' % n, 'td-%02i-loss' % n)

    def plot_learned(self, string=''):
        # plot a regular grid
        logger.info('plotting current features')
        Z = self.model.encode(self.Im)[:, ::-1]
        l = int(numpy.sqrt(self.k))
        tup = (l, self.k // l) if self.k > 2 else (1, self.k)
        plot_filters(Z, tup, 'plots/learned_basis%s.png' % string)

    @staticmethod
    def plot_filters(X, (n_sp1, n_sp2), file_name='basis.png', last=False, scale=False):
        plot_filters(X, (n_sp1, n_sp2), file_name, last, scale)

    def plot_value_rew(self):

        world_rew = self.world.reward_func(self.grid)  # true reward function
        model_rew = self.model.get_model_reward(self.Im)
        lstd_rew = self.model.get_lstd_reward(self.Im)
        k_lstd_rew = self.model.get_lstd_reward(self.Im)

        model_val = self.model.get_model_value(self.Im)
        lstd_val = self.model.get_lstd_value(self.Im)
        k_lstd_val = self.model.get_lstd_value(self.Im, k=self.k)

        value_list = [model_val, lstd_val, k_lstd_val]
        if hasattr(self.world, 'value_func'):
            value_list.append(self.world.value_func(self.grid))

        self.plot_filters(numpy.vstack([model_rew, lstd_rew, k_lstd_rew, world_rew]).T, (1, 4),
                          file_name='plots/rewards%s.png' % self.output.timestamp)
        self.plot_filters(numpy.vstack(value_list).T, (1, len(value_list)),
                          file_name='plots/values%s.png' % self.output.timestamp)

