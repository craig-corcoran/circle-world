import numpy
import theano
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA

class RSIS(object):

    def __init__(self,
                d, # dim of state rep (initial features)
                k, # dim after linear transorm
                Phi = None, # feature transform matrix f(x) = Phi x; [dxk]
                T = None, # transition matrix; Tz_t = z_t+1
                q = None, # reward function weights; r(x) = q' Phi x
                w = None, # value function weights; v(x) = w' Phi x
                Mz = None, # transition noise, kxl where l <= k
                sr = None, # reward noise
                shift = 1e-12,
                l2 = 1e-4, # l2 regularization constant
                ):
        self.Phi = 0.1*numpy.random.standard_normal((d,k)) if Phi is None else Phi
        #self.Phi = numpy.zeros((d,k)) if Phi is None else Phi
        self.T = numpy.identity(k) if T is None else T
        self.q = numpy.random.standard_normal(k) if q is None else q
        #self.q = numpy.zeros(k) if q is None else q
        self.w = numpy.random.standard_normal(k) if w is None else w
        self.Mz = numpy.identity(k) if Mz is None else Mz
        self.Sz = numpy.dot(self.Mz,self.Mz.T)
        self.sr = 1. if sr  is None else sr
        self.inv_shift = shift*numpy.identity(k)

        self.Phi_t = TT.dmatrix('Phi')
        self.T_t = TT.dmatrix('T')
        self.q_t = TT.dvector('q')
        self.Mz_t = TT.dmatrix('Mz')
        self.Sz_t = TT.dot(self.Mz_t,self.Mz_t.T) # forces Sz to be positive semidefinite
        self.sr_t = TT.dscalar('sr')
        self.X_t = TT.dmatrix('X')
        self.R_t = TT.dvector('R')

        self.Z_t = self._encode_t(self.X_t) # encode X into low-d state Z
        self.inv_shift_t = TT.sharedvar.scalar_constructor(shift) * TT.identity_like(self.T_t)

        loss_t = self._loss_t()
        self.theano_loss = theano.function(self.theano_vars, loss_t, on_unused_input='ignore')

        grad = theano.grad(TT.sum(loss_t) + l2 * sum(
            TT.sum(p ** 2) for p in self.theano_params), self.theano_params)
        self.theano_grad = theano.function(self.theano_vars, grad)

    @property
    def theano_vars(self):
        return [self.X_t, self.R_t, self.Phi_t, self.T_t, self.q_t, self.Mz_t, self.sr_t]

    @property
    def theano_params(self):
        return [self.Phi_t, self.T_t, self.q_t, self.Mz_t, self.sr_t]

    @property
    def params(self):
        return [self.Phi, self.T, self.q, self.Mz, self.sr]

    def value(self, z):
        return numpy.dot(z, self.w)

    def reward(self, z):
        return numpy.dot(z, self.q)

    def _reward_t(self, z):
        return TT.dot(z, self.q_t)

    def transition(self, z):
        return numpy.dot(z, self.T)

    def _transition_t(self, z):
        return TT.dot(z, self.T_t)

    def encode(self, x):
        return numpy.dot(x, self.Phi)

    def _encode_t(self, x):
        return TT.dot(x, self.Phi_t)

    def loss(self, X, R):
        ''' callable, takes array of features X and rewards R and returns the
        loss given the current set of parameters. Examples through time are
        indexed by row '''

        Z = self.encode(X)
        rerr = numpy.sum((R - self.reward(Z))**2)/self.sr**2
        zerr_v = (Z[1:] - self.transition(Z[:-1])) #n-1 by k
        zerr_vp = numpy.dot(zerr_v, numpy.linalg.inv(self.Sz)) #n-1 by k
        zerr = numpy.sum(numpy.multiply(zerr_vp, zerr_v))

        n = Z.shape[0]
        norm = (n-1)*numpy.log(numpy.linalg.det(self.Sz)) + 2*n*numpy.log(self.sr)

        return zerr, rerr, norm

    def _loss_t(self):
        ''' Generates the theano loss variable '''
        #return self.R_t - TT.dot(self.Z_t, self.q_t)
        rerr = TT.sum(TT.sqr(self.R_t - self._reward_t(self.Z_t)))/self.sr_t**2
        zerr_v = self.Z_t[1:] - self._transition_t(self.Z_t[:-1])
        zerr_vp = TT.dot(zerr_v, LA.matrix_inverse(self.Sz_t))
        zerr = TT.sum(TT.mul(zerr_v, zerr_vp))

        n = TT.sum(TT.ones_like(self.R_t))
        norm = (n-1)*TT.log(LA.det(self.Sz_t)) + 2*n*TT.log(self.sr_t)

        #return TT.sum(zerr + rerr + norm)
        return zerr, rerr, norm

    def _loss(self, X, R):
        ''' slightly slower theano version of loss function '''

        return self.theano_loss(X, R, self.Phi, self.T, self.q, self.Mz, self.sr)

    def grad(self, X, R):
        ''' returns gradient at the current parameters with the given inputs X
        and R. '''

        return self.theano_grad(X, R, self.Phi, self.T, self.q, self.Mz, self.sr)

    def grad_step(self, X, R, rate):
        grad = self.grad(X,R)
        self.update_params(rate, grad)

    def update_params(self, rate, grad):
        for i, p in enumerate(self.params):
            p -= rate * grad[i]

        self.Sz = numpy.dot(self.Mz,self.Mz.T)
        self.last_delta = (rate, grad)

    def revert_last_delta(self):
        self.update_params(-self.last_delta[0], self.last_delta[1])


