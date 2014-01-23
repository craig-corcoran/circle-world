import numpy
import theano
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA

class BASE(object):
    ''' base model class which defines the central model variables and
    optimization infrastructure, but leaves the theano loss variables _loss_t
    to be defined'''

    def __init__(self,
                d, # dim of input features
                k, # dim of learned representation
                h, # time horizon 
                l1 = 1e-3, # l1 regularization constant
                l2 = 1e-6, # l2 regularization constant
                shift = 1e-12,
                init_scale = 1e-1,
                Phi = None, # feature transform matrix f(x) = Phi x; [dxk]
                T = None, # transition matrix; Tz_t = z_t+1
                q = None, # reward function weights; r(x) = q' Phi x
                w = None, # value function weights; v(x) = w' Phi x
                Sz = None, # transition noise, kxl where l <= k
                sr = None, # reward noise
                ):

        self.d = d
        self.k = k
        self.h = h
        self.Phi = init_scale*numpy.random.standard_normal((d,k)) if Phi is None else Phi
        #self.Phi = numpy.zeros((d,k)) if Phi is None else Phi
        #self.T = init_scale*numpy.random.standard_normal((k,k)) if T is None else T
        self.T = numpy.identity(k) if T is None else T
        self.q = init_scale*numpy.random.standard_normal(k) if q is None else q
        #self.q = numpy.zeros(k) if q is None else q
        self.Sz = numpy.identity(k) if Sz is None else Sz
        self.sr = numpy.array(1.) if sr  is None else sr
        self.l1 = l1
        self.l2 = l2
        self.shift = shift
        self.inv_shift = shift*numpy.identity(k)
        self.init_scale = init_scale
        self.w = None # can solve for this upon value funct query

        self.Phi_t = TT.dmatrix('Phi')
        self.T_t = TT.dmatrix('T')
        self.q_t = TT.dvector('q')
        self.Sz_t = TT.dmatrix('Sz')
        self.sr_t = TT.dscalar('sr')
        self.X_t = TT.dmatrix('X')
        self.R_t = TT.dvector('R')
        
        self.n_t = TT.sum(TT.ones_like(self.R_t))
        self.Z_t = self._encode_t(self.X_t) # encode X into low-d state Z
        self.inv_shift_t = TT.sharedvar.scalar_constructor(shift) * TT.identity_like(self.T_t)
        self.Sz_inv_t = LA.matrix_inverse(self.Sz_t + self.inv_shift_t)

        self.param_names = [x.name for x in self.theano_params]
    
        # compute theano loss function 
        loss_t = self._loss_t() 
        self.theano_loss = theano.function(self.theano_vars, loss_t, on_unused_input='ignore')

        grad = theano.grad(loss_t, self.theano_params) # , disconnected_inputs='ignore'
        self.theano_grad = theano.function(self.theano_vars, grad, on_unused_input='ignore')

    @property
    def theano_vars(self):
        return [self.Phi_t, self.T_t, self.q_t, self.Sz_t, self.sr_t, self.X_t, self.R_t]

    @property
    def theano_params(self):
        return [self.Phi_t, self.T_t, self.q_t, self.Sz_t, self.sr_t] 
    
    @property
    def params(self):
        return [self.Phi, self.T, self.q, self.Sz, self.sr]

    @property
    def args(self):
        return []

    @property
    def flat_params(self):
        return self._flatten(self.params)
    
    @property
    def shapes(self):
        return map(numpy.shape, self.params)

    @staticmethod
    def _flatten(params):
        z = numpy.array([])
        for p in params:
            z = numpy.append(z, p.flatten())
        return z

    def set_params(self, params, flat = True):
        if flat:
            assert params.ndim == 1
            self.set_params(self._unpack_params(params), flat = False)
        else:
            for i, name in enumerate(self.param_names): 
                self.__setattr__(name, params[i])

    def _unpack_params(self, vec):
        i = 0
        params = []
        for s in self.shapes:
            j = i + numpy.product(s)
            params.append(vec[i:j].reshape(s))
            i = j
        return params

    def reward_func(self, x):
        return self._reward(self.encode(x))

    def value_func(self, x, gam):
        # w = q + gam * T * w  ->  w = (I - gam * T)^-1 q
        self.w = numpy.linalg.solve(numpy.identity(self.T.shape[0]) - gam * self.T, self.q)
        return numpy.dot(self.encode(x), self.w)

    def lstd_value_func(self, X, R, gam, X_eval):
        phi = self.encode(X)
        A = phi[:-1] - gam * phi[1:]
        b = R[:-1]
        w = numpy.linalg.lstsq(A,b)[0]
        return numpy.dot(self.encode(X_eval), w)

    def _reward(self, z):
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

    def regularization(self):
        reg = self.l1 * numpy.sum(self.Phi**2) # numpy.sum(abs(self.Phi))
        reg += self.l2 * sum(numpy.sum(p**2) for p in self.params[1:])
        return reg

    def _regularization_t(self):
        reg = self.l1 * TT.sum(self.Phi_t**2) # TT.sum(abs(self.Phi_t))
        reg += self.l2 * sum(TT.sum(p * p) for p in self.theano_params[1:])
        return reg
    
    def loss(self, X, R):
        ''' callable, takes array of features X and rewards R and returns the
        loss given the current set of parameters. Examples through time are
        indexed by row '''

        return self.theano_loss(*(self.params + self.args + [X, R]))

    def grad(self, X, R):
        ''' returns gradient at the current parameters with the given inputs X
        and R. '''

        return self.theano_loss(*(self.params + self.args + [X, R]))

    def optimize_loss(self, params, X, R):
        unpacked = self._unpack_params(params)
        return self.theano_loss(*(unpacked + self.args + [X, R]))

    def optimize_grad(self, params, X, R):
        unpacked = self._unpack_params(params)
        grad = self.theano_grad(*(unpacked + self.args + [X, R]))
        return self._flatten(grad)



class Horizon_RSIS(BASE):

    def __init__(self, *args, **kwargs):
        super(Horizon_RSIS, self).__init__(*args, **kwargs)

        err_t = self._smooth_loss() 
        self.smooth_loss = theano.function(self.theano_vars, err_t, on_unused_input='ignore')

        rerr_grad_t = theano.grad(err_t, self.theano_params, disconnected_inputs='ignore')
        self.smooth_grad = theano.function(self.theano_vars, rerr_grad_t, on_unused_input='ignore')

    def _loss_t(self):
        
        rerr = 0.
        ntot = 0.
        A = TT.identity_like(self.T_t)
        for i in xrange(self.h+1):

            z = self.Z_t if i is 0 else self.Z_t[:-i]
            r = self.R_t if i is 0 else self.R_t[i:]
            rerr += TT.sum((self._reward_t(TT.dot(z, A)) - r)**2)
            ntot += r.shape[0]

            A = TT.dot(A, self.T_t)

        return rerr / ntot + self._regularization_t()

    @property
    def theano_params(self):
        return [self.Phi_t, self.T_t, self.q_t]
    
    @property
    def params(self):
        return [self.Phi, self.T, self.q]

    @property
    def args(self):
        return [self.Sz, self.sr]

    def _reward_error_t(self):
        return TT.sum((self.R_t - self._reward_t(self.Z_t))**2)/self.sr_t**2

    def _transition_error_t(self):
        zerr_v = self.Z_t[1:] - self._transition_t(self.Z_t[:-1])
        zerr_vp = TT.dot(zerr_v, self.Sz_inv_t)
        return TT.sum(TT.mul(zerr_v, zerr_vp))

    def _smooth_loss(self):
        return self._reward_error_t() + self._transition_error_t() 

    def optimize_smooth_loss(self, params, X, R):
        unpacked = self._unpack_params(params)
        return self.smooth_loss(*(unpacked + self.args + [X, R]))

    def optimize_smooth_grad(self, params, X, R):
        unpacked = self._unpack_params(params)
        grad = self.smooth_grad(*(unpacked + self.args + [X, R]))
        return self._flatten(grad)


        



    

