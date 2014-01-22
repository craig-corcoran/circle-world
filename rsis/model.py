import numpy
import theano
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA




class RSIS(object):

    def __init__(self,
                d, # dim of state rep (initial features)
                k, # dim after linear transorm
                h, # length of backups to do
                Phi = None, # feature transform matrix f(x) = Phi x; [dxk]
                T = None, # transition matrix; Tz_t = z_t+1
                q = None, # reward function weights; r(x) = q' Phi x
                w = None, # value function weights; v(x) = w' Phi x
                Mz = None, # transition noise, kxl where l <= k
                sr = None, # reward noise
                l1 = 1e-3, # l1 regularization constant
                l2 = 1e-6, # l2 regularization constant
                shift = 1e-12,
                init_scale = 1e-1,
                ):
        self.d = d
        self.k = k
        self.h = h
        self.Phi = init_scale*numpy.random.standard_normal((d,k)) if Phi is None else Phi
        #self.Phi = numpy.zeros((d,k)) if Phi is None else Phi
        self.T = init_scale*numpy.random.standard_normal((k,k)) if T is None else T
        #self.T = numpy.identity(k) if T is None else T
        self.q = init_scale*numpy.random.standard_normal(k) if q is None else q
        #self.q = numpy.zeros(k) if q is None else q
        self.Mz = numpy.identity(k) if Mz is None else Mz
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
        self.Mz_t = TT.dmatrix('Mz')
        self.Sz_t = TT.dot(self.Mz_t,self.Mz_t.T) # forces Sz to be positive semidefinite
        self.sr_t = TT.dscalar('sr')
        self.X_t = TT.dmatrix('X')
        self.R_t = TT.dvector('R')

        self.n_t = TT.sum(TT.ones_like(self.R_t))

        self.Z_t = self._encode_t(self.X_t) # encode X into low-d state Z
        self.inv_shift_t = TT.sharedvar.scalar_constructor(shift) * TT.identity_like(self.T_t)

        self.param_names = [x.name for x in self.theano_params]
    
        # compute theano loss function 
        loss_t = self._loss_t() 
        self.theano_loss = theano.function(self.theano_vars, loss_t)

        grad = theano.grad(loss_t, self.theano_params)
        self.theano_grad = theano.function(self.theano_vars, grad)

    @property
    def theano_vars(self):
        return [self.Phi_t, self.T_t, self.q_t, self.Mz_t, self.sr_t, self.X_t, self.R_t]

    @property
    def theano_params(self):
        return [self.Phi_t, self.T_t, self.q_t, self.Mz_t, self.sr_t] 
    
    @property
    def params(self):
        return [self.Phi, self.T, self.q, self.Mz, self.sr]

    @property
    def flat_params(self):
        return self._flatten(self.params)
    
    @property
    def shapes(self):
        return map(numpy.shape, self.params)

    @property
    def Sz(self):
        return numpy.dot(self.Mz,self.Mz.T)

    @property
    def args(self):
        return []

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
        reg = self.l1 * numpy.sum(abs(self.Phi))
        reg += self.l2 * sum(numpy.sum(p**2) for p in self.params[1:]) # TODO what about S = M*M.T
        return reg

    def _regularization_t(self):
        reg = self.l1 * TT.sum(abs(self.Phi_t))
        reg += self.l2 * sum(TT.sum(p * p) for p in self.theano_params[1:])
        return reg

    def _reward_error_t(self):
        return TT.sum((self.R_t - self._reward_t(self.Z_t))**2)/self.sr_t**2

    def reward_error(self, Z, R):
        return numpy.sum((R - self._reward(Z))**2)/self.sr**2

    def _transition_error_t(self):
        zerr = 0.
        n_tot = 0.
        for h in xrange(self.h):
            zerr_v = self.Z_t[h+1:] - self._transition_t(self.Z_t[:-h-1])
            zerr_vp = TT.dot(zerr_v, LA.matrix_inverse(self.Sz_t + self.inv_shift_t))
            zerr += TT.sum(TT.mul(zerr_v, zerr_vp))
            n_tot += self.n_t - h - 1
        return zerr / n_tot * (self.n_t-1) # XXX

    def transition_error(self, Z):
        zerr_v = (Z[1:] - self.transition(Z[:-1])) #n-1 by k
        zerr_vp = numpy.dot(zerr_v, numpy.linalg.inv(self.Sz+self.inv_shift)) #n-1 by k
        return numpy.sum(numpy.multiply(zerr_vp, zerr_v))

    def _norm_t(self):
        ln2pi = numpy.log(2 * numpy.pi)
        return (self.n_t-1)*(TT.log(LA.det(self.Sz_t+self.inv_shift_t)) + self.k*ln2pi) + \
               self.n_t*(2*TT.log(self.sr_t) + ln2pi)

    def norm(self, n):
        ln2pi = numpy.log(2 * numpy.pi)
        return (n-1)*(numpy.log(numpy.linalg.det(self.Sz+self.inv_shift)) + self.k*ln2pi) + \
               n*(2*numpy.log(self.sr) + ln2pi)

    def _loss_t(self):
        ''' Generates the theano loss variable '''
        rerr = self._reward_error_t()
        zerr = self._transition_error_t()
        norm = self._norm_t()
        reg = self._regularization_t()

        return TT.sum(zerr + rerr + norm) / (self.n_t-1) + reg

    def _loss(self, X, R):
        ''' numpy version of loss function '''

        Z = self.encode(X)
        n = Z.shape[0]
        rerr = self.reward_error(Z,R)
        zerr = self.transition_error(Z)
        norm = self.norm(n)
        reg = self.regularization()
        
        return rerr / (n-1), zerr / (n-1), norm / (n-1), reg

    def unscaled_loss(self, X, R):
        ''' numpy version of unscaled loss function '''

        Z = self.encode(X)
        n = Z.shape[0]
        rerr = numpy.sum((R - self._reward(Z))**2)
        zerr_v = (Z[1:] - self.transition(Z[:-1])) #n-1 by k
        zerr = numpy.sum(numpy.multiply(zerr_v, zerr_v))
        norm = self.norm(n)
        reg = self.regularization()

        return rerr / (n-1), zerr / (n-1), norm / (n-1), reg

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


class CD_RSIS(RSIS):


    def __init__(self, *args, **kwargs):

        super(CD_RSIS, self).__init__(*args, **kwargs)
    
        self._Sz = numpy.identity(self.k)
        self.Sz_t = TT.dmatrix('Sz')
        del self.Mz
        del self.Mz_t
        
        # recompute theano loss function with new Sz variable 
        loss_t = self._loss_t() 
        self.theano_loss = theano.function(self.theano_vars, loss_t)

        grad = theano.grad(loss_t, self.theano_params)
        self.theano_grad = theano.function(self.theano_vars, grad)
    
    @property
    def Sz(self):
        return self._Sz

    @property
    def theano_vars(self):
        return [self.Phi_t, self.T_t, self.q_t, self.Sz_t, self.sr_t, self.X_t, self.R_t]        

    @property
    def theano_params(self):
        return [self.Phi_t, self.T_t, self.q_t] 
    
    @property
    def params(self):
        return [self.Phi, self.T, self.q]
    
    @property
    def args(self):
        return [self.Sz, self.sr]
    
    def set_noise_params(self, X, R):
        
        Z = self.encode(X)
        n = Z.shape[0]
        Rerr = R - self._reward(Z)
        Zerr = Z[1:] - self.transition(Z[:-1])
        self.sr = numpy.sqrt(numpy.sum(Rerr**2) / n) 
        self._Sz = numpy.dot(Zerr.T, Zerr) / (n-1) 

class DYN_RSIS(CD_RSIS):
    
    ''' dynamic rsis model which tracks the full dynamics of X '''
    def __init__(self,  *args, **kwargs):
        #U = kwargs.pop('U')
        #self.U = self.init_scale * numpy.random.standard_normal((self.d, self.k)) if U is None else U
        #self.U_t = TT.dmatrix('U')
        self.dyn_wt = kwargs.pop('dyn_wt')
        super(DYN_RSIS, self).__init__(*args, **kwargs)
    
    def decode(self, z):
        return numpy.dot(z, self.Phi.T)

    def _decode_t(self, z):
        return TT.dot(z, self.Phi_t.T)

    def dynamics_loss(self, X):
        return numpy.sum((X[1:] - self.decode(self.transition(self.encode(X[:-1]))))**2)

    def _dynamics_loss_t(self):
        return TT.sum((self.X_t[1:] - self._decode_t(self._transition_t(self._encode_t(self.X_t[:-1]))))**2) 

    def _loss_t(self):
        ''' Generates the theano loss variable '''
        return super(DYN_RSIS, self)._loss_t() + self.dyn_wt * self._dynamics_loss_t() / (self.n_t-1)

    def _loss(self, X, R):
        ''' numpy version of loss function '''
        rerr, zerr, norm, reg = super(DYN_RSIS, self)._loss(X,R)        
        return rerr, zerr, norm, reg, self.dyn_wt * self.dynamics_loss(X) 

    def unscaled_loss(self, X, R):
        ''' numpy version of unscaled loss function '''
        rerr, zerr, norm, reg = super(DYN_RSIS, self).unscaled_loss(X,R)
        return rerr, zerr, norm, reg, self.dynamics_loss(X) 



class AR_RSIS(CD_RSIS):
    
    def __init__(self, *args, **kwargs):
        
        self.offdiag = kwargs.pop('offdiag')
        self.lock_phi0 = kwargs.pop('lock_phi0')
        if self.lock_phi0 is None:
            self.lock_phi0 = False
        super(AR_RSIS, self).__init__(*args, **kwargs)
        
        #self.phi_t = self.Phi_t[:,0]

        # bc q is always [100..], the gradient for the reward loss is zero 
        # except for the first column of Phi 
        #self.q = numpy.zeros_like(self.q)
        #self.q[0] = 1.
        
        rerr_t = self._reward_error_t() 
        self.reward_loss = theano.function(self.theano_vars, rerr_t, on_unused_input='ignore')

        rerr_grad_t = theano.grad(rerr_t, self.theano_params, disconnected_inputs='ignore')
        self.reward_grad = theano.function(self.theano_vars, rerr_grad_t, on_unused_input='ignore')

    #@property
    #def theano_params(self):
        #return [self.Phi_t, self.T_t] 
    
    #@property
    #def params(self):
        #return [self.Phi, self.T]

    #@property
    #def args(self):
        #return [self.q, self.Sz, self.sr]
    
    #@property
    #def phi(self):
        #return self.Phi[:,0]

    def _loss(self, X, R):
        ''' numpy version of loss function '''

        Z = self.encode(X)
        n = Z.shape[0]
        rerr = self.reward_error(Z,R)
        zerr = self.transition_error(Z)
        norm = self.norm(n)
        reg = self.regularization(Z)
        
        return rerr / (n-1), zerr / (n-1), norm / (n-1), reg

    def unscaled_loss(self, X, R):
        ''' numpy version of unscaled loss function '''

        Z = self.encode(X)
        n = Z.shape[0]
        rerr = numpy.sum((R - self._reward(Z))**2)
        zerr_v = (Z[1:] - self.transition(Z[:-1])) #n-1 by k
        zerr = numpy.sum(numpy.multiply(zerr_v, zerr_v))
        norm = self.norm(n)
        reg = self.regularization(Z)

        return rerr / (n-1), zerr / (n-1), norm / (n-1), reg

        

    def regularization(self, Z):
        reg = super(AR_RSIS, self).regularization(Z)
        n = Z.shape[0]
        #C = numpy.dot(self.Phi.T, self.Phi) / n
        C = numpy.dot(Z.T, Z) / n
        #numpy.fill_diagonal(C, 0) # modifies in place
        #reg += self.offdiag * numpy.sum(abs(C))
        reg += self.offdiag * numpy.sum(numpy.abs(C - numpy.identity(C.shape[0]))) 
        return reg

    def _regularization_t(self):
        reg = super(AR_RSIS, self)._regularization_t()
        #C = TT.dot(self.Phi_t.T, self.Phi_t) / self.n_t
        C = TT.dot(self.Z_t.T, self.Z_t) / self.n_t 
        #C = TT.fill_diagonal(C, 0)
        #reg += self.offdiag * TT.sum(abs(C)) 
        reg += self.offdiag * TT.sum(abs(C - TT.identity_like(C))) 
        return reg
        
    def optimize_rew_loss(self, params, X, R):
        unpacked = self._unpack_params(params)
        return self.reward_loss(*(unpacked + self.args + [X, R]))

    def optimize_rew_grad(self, params, X, R):
        unpacked = self._unpack_params(params)
        grad = self.reward_grad(*(unpacked + self.args + [X, R]))
        return self._flatten(grad)

    def optimize_grad(self, params, X, R):
        grad = super(AR_RSIS, self).optimize_grad(params, X, R)
        if self.lock_phi0:
            grad[:self.d] = 0.
        return grad
        



    

