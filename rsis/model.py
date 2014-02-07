import numpy
import scipy
import theano
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA

theano.config.warn.subtensor_merge_bug = False

class LSTD(object):
    
    def __init__(self, d, gam = 1-1e-2, shift = 2., all_steps = False):
            
        # C = S - gam * Sp
        self.S = numpy.zeros((d, d))
        self.Sp = numpy.zeros((d, d))
        self.b = numpy.zeros(d)
        self.d = d
        self.gam = gam
        self.shift = shift
        self.all_steps = all_steps
        self.ntot = 0
        self.changed = False

    @property
    def C(self):
        return self.S - self.gam * self.Sp

    def update_params(self, X, R):
        nsteps = X.shape[0] if self.all_steps else 2
        for i in xrange(1,nsteps):
            
            h = X[i:].shape[0]
            self.ntot += h

            self.S += numpy.dot(X[:-i].T, X[:-i])
            self.Sp += numpy.dot(X[:-i].T, self.gam**i * X[i:])
            
            M = numpy.zeros((X.shape[0], h))
            for j in xrange(h):
                M[j:j+i, j] = self.gam**numpy.arange(i)
                
            r = numpy.dot(R, M)
            self.b += numpy.dot(X[:-i].T, r)

        self.changed = True

    def get_reward(self, X):
        try:
            q = scipy.linalg.solve(self.S, self.b)
        except numpy.linalg.linalg.LinAlgError as e:
            #logger.info('singular matrix error, using offset')
            print 'singular matrix error, using offset'
            q = scipy.linalg.solve(self.S + self.shift * numpy.identity(self.d), self.b)
        
        return numpy.dot(X, q)

    def get_weights(self):
        C = self.C / self.ntot
        b = self.b / self.ntot
        try:
            return scipy.linalg.solve(C, b) 
        except numpy.linalg.linalg.LinAlgError as e:
            #logger.info('singular matrix error, using offset')
            print 'singular matrix error, using offset'
            return scipy.linalg.solve(C + self.shift * numpy.identity(self.d), b)

    def get_value(self, X):
        if self.changed:
            self.lstd_w = self.get_weights()
        return numpy.dot(X, self.lstd_w)

class LowRankLSTD(object):
    
    ''' low rank lstd maintains full base feature statistics X^T X, but projects
    onto smaller basis Phi before solving for weights. Phi is learned iteratively.'''

    def __init__(self,
                d, # dim of input features
                k, # dim of learned representation
                l1 = 1e-3, # l1 regularization constant
                l2 = 1e-6, # l2 regularization constant
                shift = 1e-12,
                init_scale = 1e-1,
                Phi = None, # feature transform matrix f(x) = Phi x; [dxk]
                T = None, # transition matrix; Tz_t = z_t+1
                q = None, # reward function weights; r(x) = q' Phi x
                w = None, # value function weights; v(x) = w' Phi x
                ):
        
        self.d = d
        self.k = k
        self.Phi = init_scale*numpy.random.standard_normal((d,k)) if Phi is None else Phi
        self.T = T if T else numpy.identity(k) 
        self.q = q if q else init_scale*numpy.random.standard_normal(k) 
        self.l1 = l1
        self.l2 = l2
        self.shift = shift
        self.inv_shift = shift*numpy.identity(k)
        self.w = None # can solve for this upon value funct query

        self.Phi_t = TT.dmatrix('Phi')
        self.T_t = TT.dmatrix('T')
        self.q_t = TT.dvector('q')
        self.X_t = TT.dmatrix('X')
        self.R_t = TT.dvector('R')
       
        self.n_t = TT.sum(TT.ones_like(self.R_t))
        self.Z_t = self._encode_t(self.X_t) # encode X into low-d state Z
        self.inv_shift_t = TT.sharedvar.scalar_constructor(shift) * TT.identity_like(self.T_t)
    

         


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
        self.Tpows_t, _ = theano.scan(
                            fn=lambda prior_result, T: TT.dot(prior_result, T),
                            outputs_info=TT.identity_like(self.T_t),
                            non_sequences=self.T_t,
                            n_steps=self.h)

        
        self.params = self.all_params
        self.params_t = self.all_params_t

    
    def compile_theano_funcs(self): 
        # compute theano horizon loss function 
        loss_t = self._loss_t() 
        self.theano_loss = theano.function(self.all_vars_t, loss_t, on_unused_input='ignore')

        grad = theano.grad(loss_t, self.all_params_t) # , disconnected_inputs='ignore'
        self.theano_grad = theano.function(self.all_vars_t, grad, on_unused_input='ignore')
    
    def set_wrt(self, params, params_t):
        self.params = params
        self.params_t = params_t

    @property
    def all_vars_t(self):
        return [self.Phi_t, self.T_t, self.q_t, self.Sz_t, self.sr_t, self.X_t, self.R_t]
    
    @property
    def all_vars(self):
        return [self.Phi, self.T, self.q, self.Sz, self.sr]

    @property
    def all_params_t(self):
        return [self.Phi_t, self.T_t, self.q_t] 
    
    @property
    def all_params(self):
        return [self.Phi, self.T, self.q]

    @property
    def _args(self):
        return [self.Sz, self.sr]

    @property
    def flat_params(self, params=None):
        p = params if params else self.params
        return self._flatten(p)
    
    @property
    def shapes(self):
        return map(numpy.shape, self.params)

    @property
    def param_names(self):
        return [x.name for x in self.params_t]

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
        reg += self.l2 * sum(TT.sum(p * p) for p in self.params_t[1:])
        return reg

    def _reward_error_t(self):
        return TT.sum((self.R_t - self._reward_t(self.Z_t))**2)/self.sr_t**2

    def reward_error(self, X, R):
        # alternatively call theano function reward_loss(
        return numpy.sum((R - self._reward(self.encode(X)))**2)/self.sr**2

    def _transition_error_t(self):
        zerr_v = self.Z_t[1:] - self._transition_t(self.Z_t[:-1])
        zerr_vp = TT.dot(zerr_v, self.Sz_inv_t)
        return TT.sum(TT.mul(zerr_v, zerr_vp))

    def transition_error(self, X):
        zerr_v =  self.encode(X[1:]) - self.transition(self.encode(X[:-1]))
        zerr_vp = numpy.dot(zerr_v, numpy.linalg.inv(self.Sz))
        return numpy.sum(numpy.multiply(zerr_v, zerr_vp))

    def _loss_t(self):
        ''' horizon loss '''
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

    def loss(self, X, R):
        ''' callable, takes array of features X and rewards R and returns the
        loss given the current set of parameters. Examples through time are
        indexed by row '''

        return self.theano_loss(*[self.Phi, self.T, self.q, self.Sz, self.sr, X, R])

    def grad(self, X, R):
        ''' returns gradient at the current parameters with the given inputs X
        and R. '''

        return self.theano_grad(*[self.Phi, self.T, self.q, self.Sz, self.sr, X, R])

    def get_args(self, params, X, R):
        params = params if (type(params) == list) else self._unpack_params(params)        
        return params + self._args + [X, R]

    def optimize_loss(self, params, X, R):
        self.params, self.params_t = self.all_params, self.all_params_t
        return self.theano_loss(*self.get_args(params, X, R))

    def optimize_grad(self, params, X, R):
        self.params, self.params_t = self.all_params, self.all_params_t
        grad = self.theano_grad(*self.get_args(params, X, R))
        return self._flatten(grad)


class Multistep_RSIS(BASE):

    def __init__(self, *args, **kwargs):
        super(Multistep_RSIS, self).__init__(*args, **kwargs)
        self.compile_theano_funcs()
    
    def _loss_t(self):
        zerr = 0.
        rerr = TT.sum((self.R_t - self._reward_t(self.Z_t))**2)
        ntot = 0.
        A = TT.identity_like(self.T_t)
        for i in xrange(1, self.h+1):

            A = TT.dot(A, self.T_t)
            z = self.Z_t[:-i]
            zp = self.Z_t[i:] #  TODO add Sz_inv
            r = self.R_t[i:]
            zerr += TT.sum((TT.dot(z, A) - zp)**2)
            rerr += TT.sum((self._reward_t(TT.dot(z, A)) - r)**2)
            ntot += z.shape[0]

        return zerr / ntot + rerr / ntot + self._regularization_t()

    def _transition_error_t(self):
        ''' overwrites base method adds zerr for all steps in horizon '''

        zerr = 0.
        ntot = 0.
        A = TT.identity_like(self.T_t)
        for i in xrange(1, self.h+1):

            A = TT.dot(A, self.T_t)
            z = self.Z_t[:-i]
            zp = self.Z_t[i:] #  TODO add Sz_inv
            zerr += TT.sum((TT.dot(z, A) - zp)**2)
            ntot += z.shape[0]

        return zerr / ntot

    def transition_error(self, X):
        ''' overwrites base method, adds zerr for all steps in horizon '''
        zerr = 0.
        ntot = 0.
        A = numpy.identity(self.T.shape[0])
        for i in xrange(1, self.h+1):

            A = numpy.dot(A, self.T)
            z = self.encode(X[:-i])
            zp = self.encode(X[i:]) #  TODO add Sz_inv
            zerr += numpy.sum((numpy.dot(z, A) - zp)**2)
            ntot += z.shape[0]

        return zerr / ntot

    def qr_step(self):
        Q, U = numpy.linalg.qr(self.Phi)
        self.Phi = Q
        self.T = numpy.dot(U, numpy.dot(self.T, numpy.linalg.inv(U)))
        self.q = numpy.dot(U, self.q)


class QR_RSIS(BASE):

    def __init__(self, *args, **kwargs):
        super(QR_RSIS, self).__init__(*args, **kwargs)
        self.compile_theano_funcs()

    
    def qr_step(self):
        Q, U = numpy.linalg.qr(self.Phi)
        self.Phi = Q
        self.T = numpy.dot(U, numpy.dot(self.T, numpy.linalg.inv(U)))
        self.q = numpy.dot(U, self.q)


class Alternating_RSIS(BASE):

    def __init__(self, *args, **kwargs):
        super(Horizon_RSIS, self).__init__(*args, **kwargs) 
        
        # and model loss with static Phi
        model_err_t = self._model_loss_t() 
        self.model_loss = theano.function(self.all_vars_t, model_err_t, on_unused_input='ignore')

        model_grad_t = theano.grad(model_err_t, self.model_params_t, disconnected_inputs='ignore')
        self.model_grad = theano.function(self.all_vars_t, model_grad_t, on_unused_input='ignore')
    
    @property
    def model_params_t(self):
        #return [self.T_t, self.q_t]
        return [self.Phi_t, self.T_t, self.q_t]

    @property
    def model_params(self):
        ''' subset of parameters concerned with learning the model in z-space
        given a feature mapping defined by Phi '''
        #return [self.T, self.q]
        return [self.Phi, self.T, self.q]

    @property
    def _model_args(self):
        return [self.Sz, self.sr]

    def get_model_args(self, params, X, R):
        params = params if (type(params) == list) else self._unpack_params(params)
        return params + self._model_args + [X, R]
        #return [self.Phi] + params + self._model_args + [X, R]
  
    def _model_loss_t(self):
        return (self._reward_error_t() + self._transition_error_t()) / self.n_t + self._regularization_t()

    def optimize_model_loss(self, params, X, R):
        self.params, self.params_t = self.model_params, self.model_params_t # sets params so they are unpacked correctly
        return self.model_loss(*self.get_model_args(params, X, R))

    def optimize_model_grad(self, params, X, R):
        self.params, self.params_t = self.model_params, self.model_params_t
        grad = self.model_grad(*self.get_model_args(params, X, R))
        return self._flatten(grad)

