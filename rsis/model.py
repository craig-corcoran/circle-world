import numpy
import scipy
import theano
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA
import matplotlib.pyplot as plt

theano.config.warn.subtensor_merge_bug = False


class BaseLSTD(object):
    ''' base LSTD with l2 regularization '''

    def __init__(self,
                 d,
                 gam,
                 l2_lstd=1e-4,
                 S=None,
                 Sp=None,
                 b=None,
                 ntot=0,
                 ):

        self.d = d
        self.gam = gam
        self.l2_lstd = l2_lstd

        self.S = S if S else numpy.zeros((self.d, self.d))
        self.Sp = Sp if Sp else numpy.zeros((self.d, self.d))
        self.b = b if b else numpy.zeros(self.d)
        self.lstd_shift = numpy.identity(self.d) * self.l2_lstd
        self.w = None

        self.updated = {}
        self.reset_updated()
        self.ntot = ntot

    @staticmethod
    def running_avg(A, x, n, np):
        return (A * n + x) / np

    @property
    def C(self):
        return self.S - self.gam * self.Sp

    def reset_updated(self):
        self.updated.update({'w': False, 'w_k': False,
                             'q': False, 'q_k': False,
                             'T': False, 'T_k': False})

    def update_statistics(self, X, R, reset=False):

        if reset:
            self.S = numpy.zeros(self.S.shape)
            self.Sp = numpy.zeros(self.Sp.shape)
            self.b = numpy.zeros(self.n.shape)
            self.ntot = 0

        nadd = len(R) - 1  # XXX is this right

        self.S = self.running_avg(self.S,
                                  numpy.dot(X[:-1].T, X[:-1]),
                                  self.ntot,
                                  self.ntot + nadd)

        self.Sp = self.running_avg(self.Sp,
                                   numpy.dot(X[:-1].T, self.gam * X[1:]),
                                   self.ntot,
                                   self.ntot + nadd)

        self.b = self.running_avg(self.b,
                                  numpy.dot(X[:-1].T, R[:-1]),
                                  self.ntot,
                                  self.ntot + nadd)

        self.ntot += nadd
        self.reset_updated()

    def get_lstd_value(self, X, k=None):
        if k:
            if not self.updated['w_k']:
                u, d, v = scipy.sparse.linalg.svds(self.C + self.lstd_shift, k=k, which='LM')
                c_inv = numpy.dot(numpy.dot(v.T, numpy.diag(1. / d)), u.T)  # XXX plot svd features
                self.w_k = numpy.dot(c_inv, self.b)
                self.updated['w_k'] = True

            return numpy.dot(X, self.w_k)

        else:
            if not self.updated['w']:
                self.w = scipy.linalg.solve(self.C + self.lstd_shift, self.b)
                self.updated['w'] = True

            return numpy.dot(X, self.w)

    def get_lstd_reward(self, X, k=None):
        if k:
            if not self.updated['q_k']:  # use k-svd for inverse
                u, d, v = scipy.sparse.linalg.svds(self.S + self.lstd_shift, k=k, which='LM')
                s_inv = numpy.dot(numpy.dot(v.T, numpy.diag(1. / d)), u.T)  # XXX store s_inv
                self.q_k = numpy.dot(s_inv, self.b)
                self.updated['q_k'] = True

            return numpy.dot(X, self.q_k)

        else:
            if not self.updated['q']:
                self.q = scipy.linalg.solve(self.S + self.lstd_shift, self.b)
                self.updated['q'] = True

            return numpy.dot(X, self.q)

    def get_lstd_transition(self, X, k=None):
        if k:
            if not self.updated['T_k']:
                u, d, v = scipy.sparse.linalg.svds(self.S + self.lstd_shift, k=k, which='LM')
                s_inv = numpy.dot(numpy.dot(v.T, numpy.diag(1. / d)), u.T)
                self.T_k = numpy.dot(s_inv, self.Sp)
                self.updated['T_k'] = True

            return numpy.dot(X, self.T_k)

        else:
            if not self.updated['T']:
                self.T = scipy.linalg.solve(self.S + self.lstd_shift, self.Sp)
                self.updated['T'] = True

            return numpy.dot(X, self.T)

    def td_error(self, v, r, n=1):
        return numpy.sqrt(numpy.sum((r[:-n] + self.gam * v[n:] - v[:-n]) ** 2) / (len(r) - n))

    def lstd_reward_error(self, x, r, k=None):
        rhat = self.get_lstd_reward(x, k=k)
        return numpy.sqrt(numpy.sum((rhat - r) ** 2)) / len(r)

    def lstd_transition_error(self, x, k=None):
        xhat = self.get_lstd_transition(x[:-1], k=k)
        return numpy.sqrt(numpy.sum((xhat - x[1:]) ** 2)) / len(xhat)


class NN_Model(BaseLSTD):
    ''' uses a neural network trained on the bellman/temporal difference error.
    the online weights for the value function can be used, or the last layer can
    be solved in batch using lstd. '''

    def __init__(self, *args, **kwargs):

        self.k = args[0]  # number of columns of U
        self.n_layers = len(self.k)

        self.states = kwargs.pop('states')  # full state/reward sequence, subsampled during learning
        self.rewards = kwargs.pop('rewards')
        self.n_samples = len(self.rewards) - 1

        self.freeze_layers = kwargs.pop('freeze_layers', [False] * self.n_layers)
        self.n = kwargs.pop('n', 100)  # size of 'image' (subsample from statespace) input into encoding
        self.m = kwargs.pop('m', 100)  # size of minibatch: number of images used for estimating gradient
        Uinit = kwargs.pop('Uinit', [None] * self.n_layers)

        bias_layer_init = kwargs.pop('bias_layer', [None] * self.n_layers)

        self.reg_loss = kwargs.pop('reg_loss', ('l2', 1e-4))  # l2 regularization of U with reconstructive loss
        self.l2_subs = kwargs.pop('l2_subs', 1e-4)  # l2 regularization when doing lstd on the subspace of U
        self.subs_shift = numpy.identity(self.k[-1]) * self.l2_subs

        self.g, self.g_t, self.g_str = kwargs.pop('g', None)  # encoding transformation (no default)
        init_scale = kwargs.pop('init_scale', 1e-1)

        super(NN_Model, self).__init__(*args[1:], **kwargs)

        self.sizes = (self.d,) + self.k


        #self.bias_recon = bias_recon_init if bias_recon_init else numpy.zeros(self.n)
        self.bias_layer = [bias_layer_init[i] if not bias_layer_init[i] is None else
                           numpy.zeros(self.sizes[i + 1])
                           for i in xrange(self.n_layers)]
        # initialize the parameters with the given ones if present, or random ones if Uinit is none
        self.U = [Uinit[i] if not Uinit[i] is None else
                  init_scale * numpy.random.randn(self.sizes[i], self.sizes[i + 1])
                  for i in xrange(self.n_layers)]

        self.bias_layer_t = [TT.dvector('bias_layer%i' % i) for i in xrange(self.n_layers)]
        self.U_t = [TT.dmatrix('U%i' % i) for i in xrange(self.n_layers)]
        self.X_t = TT.dtensor3('X')
        self.R_t = TT.dmatrix('R')
        self.Y_t = TT.dmatrix('Y')
        self.Z_t = self.encode_t()

        self.compile_theano_funcs()
        self.update_statistics(self.states, self.rewards)
        self.reset_model_updated()

    @staticmethod
    def _flatten(U):
        return numpy.concatenate([x.flatten() for x in U])

    def _unpack(self, u):
        pt = 0
        weights = []
        for i, s in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            weights.append(numpy.reshape(u[pt:pt + s[0] * s[1]], s))
            pt += s[0] * s[1]

        biases = []
        for i, s in enumerate(self.sizes):
            biases.append(u[pt:pt + s])
            pt += s

        if len(u) > pt: # if model has a reconstruction bias (not nntd)
            biases.append(u[pt:])  # append bias_recon

        return weights, biases

    def set_data(self, x_data, r_data):

        self.states = x_data
        self.rewards = r_data
        self.update_statistics(x_data, r_data)

    def reset_model_updated(self):
        self.updated.update(
            {'model-stats': False, 'wz': False, 'qz': False, 'Tz': False})

    def compile_theano_funcs(self):

        reg_str, reg_val = self.reg_loss

        if reg_str == 'l2':
            reg = 0
            for i in xrange(self.n_layers):
                reg += TT.sum(self.U_t[i] ** 2)

            reg = TT.sqrt(reg)

        elif reg_str == 'l1':
            reg = TT.sum(abs(self.Z_t[-1]))

        else:
            assert False # TODO add contractive loss
        loss = self.loss_t() + reg_val * reg
        loss_vars = self.get_theano_params() + [self.X_t, self.R_t, self.Y_t]
        self.theano_loss = theano.function(loss_vars, loss, on_unused_input='ignore')
        self.data_loss_fn = theano.function(loss_vars, self.loss_t(), on_unused_input='ignore')
        self.reg_loss_fn = theano.function(loss_vars, reg, on_unused_input='ignore')

        grad = theano.grad(loss, self.get_theano_params(), disconnected_inputs='ignore')
        self.theano_grad = theano.function(loss_vars, grad, on_unused_input='ignore')

    def optimize_loss(self, u, X, R, Y):
        self.set_params(u)
        loss = self.theano_loss(*(self.get_params() + [X, R, Y]))
        return loss

    def optimize_grad(self, u, X, R, Y):
        self.set_params(u)
        grad = self.theano_grad(*(self.get_params() + [X, R, Y]))

        # # freeze layers
        # for i in xrange(self.n_layers):
        #     if self.freeze_layers[i]:
        #         grad[i] = numpy.zeros_like(grad[i])

        return self._flatten(grad)

    def set_params(self, u):

        if (type(u) == list) or (type(u) == tuple):
            self.U = u[0]
            self.bias_layer = u[1]
        else:
            self.U, self.bias_layer = self._unpack(u)

        self.reset_model_updated()

    def get_params(self):
        return self.U + self.bias_layer

    def get_theano_params(self):
        return self.U_t + self.bias_layer_t

    def get_flat_params(self):
        return self._flatten(self.get_params())

    def update_model_statistics(self):
        if not self.updated['model-stats']:
            Z = self.encode(self.states)
            self.S_z = numpy.dot(Z[:-1].T, Z[:-1]) / self.n_samples
            self.Sp_z = numpy.dot(Z[:-1].T, Z[1:]) / self.n_samples
            self.b_z = numpy.dot(Z.T, self.rewards) / self.n_samples
            self.updated['model-stats'] = True

    def get_model_value(self, X):

        self.update_model_statistics()

        if not self.updated['wz']:
            Z = self.encode(self.states)
            C = numpy.dot(Z[:-1].T, (Z[:-1] - self.gam * Z[1:]))
            b = numpy.dot(Z.T, self.rewards)
            self.wz = scipy.linalg.solve(C + self.subs_shift, b)
            self.updated['wz'] = True

        return numpy.dot(self.encode(X), self.wz)

    def get_model_reward(self, X):

        self.update_model_statistics()

        if not self.updated['qz']:
            #self.q = scipy.linalg.lstsq(self.states, self.rewards)[0]
            self.qz = scipy.linalg.solve(self.S_z + self.subs_shift, self.b_z)
            self.updated['qz'] = True
        return numpy.dot(self.encode(X), self.qz)

    def model_reward_error(self, x, r):

        rhat = self.get_model_reward(x)
        return numpy.sqrt(numpy.sum((rhat - r) ** 2)) / len(r)

    def model_transition_error(self, x, Tz=None):

        self.update_model_statistics()

        if not self.updated['Tz']:
            # self.T = scipy.linalg.lstsq(self.states[:-1], self.states[1:])[0]
            self.Tz = scipy.linalg.solve(self.S_z + self.subs_shift, self.Sp_z)
            self.updated['Tz'] = True

        xhat = self.decode(numpy.dot(self.encode(x[:-1]), self.Tz))
        return numpy.sqrt(numpy.sum((xhat - x[1:]) ** 2)) / len(x)

    def encode(self, y):
        y = y
        for i in xrange(self.n_layers):
            y = self.g(numpy.dot(y, self.U[i]) - self.bias_layer[i])

        return y

    def encode_t(self):
        Z = [self.g_t(TT.dot(self.Y_t, self.U_t[0]) - self.bias_layer_t[0])]
        for i in xrange(self.n_layers - 1):
            Z.append(self.g_t(TT.dot(Z[i] - self.bias_layer_t[i+1], self.U_t[i + 1])))
        return Z


class NNTD(NN_Model):
    pass

class ReconstructiveLSTD(NN_Model):
    ''' uses reconstruction loss to learn a representation that captures the subspace
    spanned by r_t, r_t+1, ... '''

    def __init__(self, *args, **kwargs):

        self.eps = kwargs.pop('eps', 1e-4)  # used to determine the horizon h w/ gam
        max_h = kwargs.pop('max_h', None)
        alpha = kwargs.pop('alpha', None)
        self.alpha_t = TT.dvector('alpha')

        super(ReconstructiveLSTD, self).__init__(*args, **kwargs)

        self.alpha = alpha if alpha is not None else numpy.ones(self.sizes[-1])
        horiz = int(numpy.log(self.eps) / numpy.log(self.gam))  # determine horizon where gam^h < eps
        self.h = min(horiz, max_h) if max_h else horiz

    def get_params(self):
        return self.U + self.bias_layer + [self.bias_recon, self.alpha]

    def get_theano_params(self):
        return self.U_t + self.bias_layer_t + [self.bias_recon_t, self.alpha_t]

    def sample_image(self, n, samp_dist='geom', seed=None):
        ''' picks a random time step difference, then randomly subsamples from
        the available samples (with replacement) '''

        # sample horizon lengths
        numpy.random.seed(seed)
        if samp_dist == 'geom':
            i = min(numpy.random.geometric(1 - self.gam), self.h)  # geometrically distributed
        else:
            i = numpy.random.randint(self.h + 1)  # uniform dist

        # sampling with replacement until nonzero reward signal
        t = numpy.random.randint(self.n_samples - i, size=n)
        ii = 0
        while len(self.rewards[t + i].nonzero()[0]) == 0:
            ii += 1
            numpy.random.seed(seed+10*ii)
            t = numpy.random.randint(self.n_samples - i, size=n)

        x = self.states[t]
        r = self.rewards[t + i]
        y = numpy.dot(x.T, r)

        return x, r, y

    def sample_minibatch(self, samp_dist='geom', seed=None):
        ''' sample a minibatch of reward images. each image is a random
        subsample of the state space with a constant temporal offset between
        the reward r and the corresponding state x '''
        x = numpy.empty((self.m, self.n, self.d))
        r = numpy.empty((self.m, self.n))
        y = numpy.empty((self.m, self.d))
        for i in xrange(self.m):
            x[i], r[i], y[i] = self.sample_image(self.n, samp_dist=samp_dist, seed=seed+i if seed else seed)

        assert numpy.sum(r**2) > 0

        x = numpy.rollaxis(x, 2)  # makes x [d,m,n]
        return x, r, y

    def decode(self, z):
        for i in xrange(self.n_layers):
            z = numpy.dot(z * self.alpha if not i else z, self.U[-(i + 1)].T)
        return z

    def decode_t(self):
        I = TT.dot(self.Z_t[-1] * self.alpha_t, self.U_t[-1].T)
        for U in self.U_t[:-1][::-1]:
            I = TT.dot(I, U.T)
        return I

    def set_params(self, u):

        if (type(u) == list) or (type(u) == tuple):
            out = u
        else:
            out = self._unpack(u)

        self.U = out[0]
        self.bias_layer = out[1][:-1]
        self.bias_recon = out[1][-1]
        self.alpha = out[2]
        self.reset_model_updated()

    def _unpack(self, u):
        pt = 0
        weights = []
        for i, s in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            weights.append(numpy.reshape(u[pt:pt + s[0] * s[1]], s))
            pt += s[0] * s[1]

        biases = []
        for i, s in enumerate(self.sizes[1:]):
            biases.append(u[pt:pt + s])
            pt += s

        biases.append(u[pt:pt + self.bias_recon_len])  # append bias_recon
        pt += self.bias_recon_len

        alpha = u[pt:]
        assert len(alpha) == self.sizes[-1]

        return weights, biases, alpha


class StatespaceRLSTD(ReconstructiveLSTD):

    def __init__(self, *args, **kwargs):
        bias_recon = kwargs.pop('bias_recon', None)
        self.bias_recon_t = TT.dvector('bias_recon')
        super(StatespaceRLSTD, self).__init__(*args, **kwargs)

        # if no initial bias given, set to the mean reward image of a minibatch or zero
        # self.bias_recon = bias_recon if bias_recon else numpy.mean(self.sample_minibatch()[1], axis=0)
        self.bias_recon = bias_recon if bias_recon is not None else numpy.zeros(self.n)
        self.bias_recon_len = self.n

    def encode_t(self):
        # X is [dxmxn]
        # R is [m,n]
        # U is [d,k]
        Y = TT.sum(TT.mul(self.X_t, self.R_t), axis=2).T  # Y is [m,d] // TODO bias here?
        #H = [self.g_t(TT.sum(TT.tensordot(self.U_t[0].T, self.X_t, 1) * self.R_t, axis=2)).T] # mxk matrix of first layer hidden unit activations
        H = [self.g_t(TT.dot(Y, self.U_t[0]) - self.bias_layer_t[0])]  # [m,k]
        for i in xrange(self.n_layers - 1):
            H.append(self.g_t(TT.dot(H[i], self.U_t[i + 1]) - self.bias_layer_t[i+1]))
        return H

    def loss_t(self):
        # equiv to sum_i || Xi^T U g( U^T Xi r_i) - r_i ||^2 
        # X is [d,m,n]
        # I is [m,d]
        I = self.decode_t()
        Rhat = TT.sum(TT.mul(self.X_t.T, I), axis=2).T
        return TT.sum(( Rhat - self.R_t - self.bias_recon_t) ** 2) / (self.m * self.n)


class ProjectedRLSTD(ReconstructiveLSTD):

    def __init__(self, *args, **kwargs):
        bias_recon = kwargs.pop('bias_recon', None)
        self.bias_recon_t = TT.dvector('bias_recon')
        super(ProjectedRLSTD, self).__init__(*args, **kwargs)

        # if no initial bias given, set to the mean reward image in x-space of a minibatch
        # self.bias_recon = bias_recon if bias_recon else numpy.mean(self.sample_minibatch()[2], axis=0)
        self.bias_recon = bias_recon if bias_recon is not None else numpy.zeros(self.d)
        self.bias_recon_len = self.d
    # Y is [m,d]
    # U is [d,k]
    # Z is [m,k]

    def encode_t(self):
        Z = [self.g_t(TT.dot(self.Y_t, self.U_t[0]) - self.bias_layer_t[0])]
        for i in xrange(self.n_layers - 1):
            Z.append(self.g_t(TT.dot(Z[i], self.U_t[i+1]) - self.bias_layer_t[i+1]))
        return Z

    def loss_t(self):
        # equiv to sum_i || g(y_i.T U) U.T - y_i ||^2
        return TT.sum((self.decode_t() - self.Y_t - self.bias_recon_t) ** 2) / (self.m * self.n)


