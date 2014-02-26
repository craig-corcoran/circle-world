import numpy
import scipy
import theano
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA
import matplotlib.pyplot as plt

theano.config.warn.subtensor_merge_bug = False

class Base_LSTD(object):
    
    ''' base LSTD with l2 regularization '''

    def __init__(self,
                d,
                gam,  
                l2_lstd = 1e-4,
                ):
    
        self.d = d        
        self.gam = gam
        self.l2_lstd = l2_lstd

        self.S = numpy.zeros((self.d,self.d))
        self.Sp = numpy.zeros((self.d,self.d))
        self.b = numpy.zeros(self.d)
        self.w = None

        self.lstd_shift = numpy.identity(self.d) * self.l2_lstd
        self.updated = {'w':False}

    @property
    def C(self):
        return self.S - self.gam * self.Sp
    
    def update_statistics(self, X, R, reset = False):
        
        if reset:
            self.S = numpy.zeros(self.S.shape)
            self.Sp = numpy.zeros(self.Sp.shape)
            self.b = numpy.zeros(self.n.shape)

        self.S += numpy.dot(X[:-1].T, X[:-1])
        self.Sp += numpy.dot(X[:-1].T, self.gam * X[1:])
        self.b += numpy.dot(X[:-1].T, R[:-1])
        
        self.updated['w'] = False

    def get_lstd_value(self, X, k = None):

        if not self.updated['w']:
            
            if k:
                u,d,v = scipy.sparse.linalg.svds(self.C + self.lstd_shift, k=self.k, which='LM')
                c_inv = numpy.dot(numpy.dot(v.T, numpy.diag(1./d)), u.T) # XXX plot svd features
                self.w = numpy.dot(c_inv, self.b)   
            else:
                self.w = scipy.linalg.solve(self.C + self.lstd_shift,  self.b)
            
            self.updated['w'] = True

        return numpy.dot(X, self.w)

    def td_error(self, v, r):
        return numpy.sqrt(numpy.sum((r[:-1] + self.gam * v[1:] - v[:-1])**2))

class Reconstructive_LSTD(Base_LSTD):
        
    ''' uses rica-like loss to learn a representation that captures the subspace
    spanned by r_t, r_t+1, ... '''

    def __init__(self, *args, **kwargs):
        
        self.k = args[0] #  number of columns of W
        
        self.states = kwargs.pop('states') # full state/reward sequence, subsampled during learning
        self.rewards = kwargs.pop('rewards')
        
        self.n = kwargs.pop('n', 100) # size of 'image' (subsample from statespace) input into encoding
        self.m = kwargs.pop('m', 100) # size of minibatch: number of images used for estimating gradient
        
        self.reg_loss = kwargs.pop('reg_loss', ('l2',1e-4)) # l2 regularization of W with reconstructive loss
        self.l2_subs = kwargs.pop('l2_subs', 1e-4) # l2 regularization when doing lstd on the subspace of W
        self.subs_shift = numpy.identity(self.k) * self.l2_subs
        
        self.g = kwargs.pop('g', lambda x: x) # encoding transformation (default is identity)
        init_scale = kwargs.pop('init_scale', 1e-4)
        self.eps = kwargs.pop('eps', 1e-4) # used to determine the horizon h w/ gam
        max_h = kwargs.pop('max_h', None)

        super(Reconstructive_LSTD, self).__init__(*args[1:], **kwargs)

        self.W = init_scale*numpy.random.randn(self.d, self.k)
        self.n_samples = len(self.rewards)
        horiz = int(numpy.log(self.eps) / numpy.log(self.gam)) # determine horizon where gam^h < eps
        self.h = min(horiz, max_h) if max_h else horiz 
                
        self.W_t = TT.dmatrix('W')
        self.X_t = TT.dtensor3('X')
        self.R_t = TT.dmatrix('R') # broadcastable

        self.compile_theano_funcs()
        self.update_statistics(self.states, self.rewards)
        self.updated['u'] = False

    def encode(self, x):
        return self.g(numpy.dot(x, self.W))

    def compile_theano_funcs(self): 
        reg_str, reg_val = self.reg_loss
        if reg_str == 'l2':        
            reg =  TT.sum(self.W_t**2)
        elif reg_str == 'l1':
            reg = TT.sum(abs(self.W_t))
        else: assert False
        loss = self.loss_t() + reg_val * reg
        loss_vars = [self.W_t, self.X_t, self.R_t]
        self.theano_loss = theano.function(loss_vars, loss) # on_unused_input='ignore')

        grad = theano.grad(loss, self.W_t) # , disconnected_inputs='ignore'
        self.theano_grad = theano.function(loss_vars, grad) # on_unused_input='ignore')

    def loss_t(self):
        # equiv to sum_i || Xi^T W g( W^T Xi r_i) - r_i ||^2
        H = self.g(TT.sum(TT.tensordot(self.W_t.T, self.X_t, 1) * self.R_t, axis=2)) # kxm matrix of hidden unit activations
        I = TT.dot(self.W_t, H).T # H is [k,m]; W is [d,k]; I is [m,d]
        Rhat = TT.sum(TT.mul(self.X_t.T, I), axis=2).T
        return TT.sum((Rhat - self.R_t)**2)

    def get_model_value(self, X):
        
        if not self.updated['u']:
            Z = self.encode(self.states)
            C = numpy.dot(Z[:-1].T, (Z[:-1] - self.gam * Z[1:]))
            b = numpy.dot(Z.T, self.rewards)
            self.u = scipy.linalg.solve(C + self.subs_shift,  b)
            self.updated['u'] = True

        return numpy.dot(self.encode(X), self.u)

    def optimize_loss(self, w, X, R):
        W = numpy.reshape(w, (self.d, self.k))
        return self.theano_loss(W, X, R)

    def optimize_grad(self, w, X, R):
        W = numpy.reshape(w, (self.d, self.k))
        grad = self.theano_grad(W, X, R)
        return grad.flatten()

    def set_params(self, w):
        self.W = numpy.reshape(w, (self.d, self.k))
        self.updated['u'] = False

    def sample_minibatch(self):
        ''' sample a minibatch of reward images. each image is a random
        subsample of the state space with a constant temporal offset between
        the reward r and the corresponding state x '''
            
        x = numpy.empty((self.m, self.n, self.d)) 
        r = numpy.empty((self.m, self.n)) 
        for i in xrange(self.m): 
            x[i], r[i] = self.sample_image(self.n)
        
        x = numpy.rollaxis(x, 2) # makes x [dxmxn]
        return x, r

    def sample_image(self, n):
        ''' picks a random time step difference, then randomly subsamples from
        the available samples (with replacement) '''
        i = numpy.random.randint(self.h+1) # XXX weight prob by gam**i? 
        assert i < self.n_samples
        t = numpy.random.randint(self.n_samples-i, size=n) # ok to sample with replacement?
        x = self.states[t]
        r = self.rewards[t+i]

        return x, r


class BKS_LSTD(object):
    
    ''' Bellman Krylov Subspace LSTD. Finds low-d krylov subspace using
    eigendecomposition, then solves for the value function in this 
    low-d space'''

    def __init__(self, *args, **kwargs):
        
        super(BKS_LSTD, self).__init__(*args[1:-1], **kwargs)

        self.k = args[0]
        self.l2_model = args[-1]
        self.model_shift = numpy.identity(self.k) * self.l2_model
        
        self.A = numpy.zeros((self.d, self.d)) # weighted covariance Q^T Q
        self.G = numpy.zeros((self.d, self.d)) # sum of feature outer products
        
        self.Phi = None        
        self.u = None
        
        self.updated.update({'u':False})

    @property
    def D(self):
        return numpy.dot(numpy.dot(self.Phi.T, self.C), self.Phi)

    @property
    def a(self):
        return numpy.dot(self.Phi.T, self.b)

    def encode(self, x):
        return numpy.dot(x, self.Phi)

    def update_statistics(self, X, R, reset_trace = False):

        super(BKS_LSTD, self).update_statistics(X, R)

        if reset_trace: 
            self.G = numpy.zeros((self.d, self.d))
        
        for x, r in zip(X,R): 
            self.G = numpy.outer(x, x) + self.gam * self.G # include gamma here? G = xx^T + gam * G
            self.A += r**2 * self.G
            
        self.updated.update({'u':False})

    def set_phi(self):
        eigvals, self.Phi = scipy.sparse.linalg.eigsh(self.A, self.k)

    def get_model_value(self, X):
        
        if not self.updated['u']:
            self.set_phi()
            self.u = scipy.linalg.solve(self.D + self.model_shift, self.a)
            self.updated['u'] = True

        return numpy.dot(self.encode(X), self.u)


class LowRankLSTD(object):
    
    ''' low rank lstd maintains full base feature statistics X^T X, but projects
    onto smaller basis Phi before solving for weights. Phi is learned iteratively.'''

    def __init__(self,
                d, # dim of input features
                k, # dim of learned representation
                gam, # discount factor for MDP
                l1 = 1e-3, # l1 regularization constant
                l2d = 1e-2, # l2 regularization constant
                l2k = 1e-6,
                init_scale = 1e-1,
                Phi = None, # feature transform matrix f(x) = Phi x; [dxk]
                S = None,
                Sp = None,
                b = None,
                ):
        
        self.d = d
        self.k = k
        self.gam = gam
        self.l1 = l1
        self.l2_lstd = l2d
        self.l2_model = l2k

        self.Phi = Phi if Phi else init_scale*numpy.random.standard_normal((d,k)) 
        self.S = S if S else numpy.zeros((d,d))
        self.Sp = Sp if Sp else numpy.zeros((d,d))
        self.b = b if b else numpy.zeros(d)

        self.lstd_shift = numpy.identity(d) * self.l2_lstd
        self.model_shift = numpy.identity(k) * self.l2_model

        self.w = None # can solve for this upon value funct query
        self.u = None 
        self.updated_u = False
        self.updated_w = False

        self.Phi_t = TT.dmatrix('Phi')
        self.S_t = TT.dmatrix('S') # S = X^T X ; Sp = X^T Xp
        self.Sp_t = TT.dmatrix('Sp') # C = (S - gam * Sp)
        self.b_t = TT.dvector('b') # b = X^T R accumulated
        self.X_t = TT.dmatrix('X') # X and R for error
        self.R_t = TT.dvector('R')
       
        self.lstd_shift_t = TT.sharedvar.scalar_constructor(self.l2_lstd) * TT.identity_like(self.S_t)        
        self.model_shift_t = TT.sharedvar.scalar_constructor(self.l2_model) * TT.identity_like(TT.dot(self.Phi_t.T, self.Phi_t))        
        self.Z_t = self.encode_t(self.X_t) # encode X into low-d state Z
        self.C_t = self.S_t - self.gam * self.Sp_t
        self.w_t = TT.dot(LA.matrix_inverse(self.C_t + self.lstd_shift_t), self.b_t)
        #self.w_t = LA.solve(self.C_t, self.b_t)

        self.D_t = TT.dot(TT.dot(self.Phi_t.T, self.C_t), self.Phi_t) # D = Phi^T C Phi
        self.c_t = TT.dot(self.Phi_t.T, self.b_t)
        #self.u_t = LA.solve(self.D_t, self.c_t) 
        self.u_t = TT.dot(LA.matrix_inverse(self.D_t + self.model_shift_t), self.c_t)

        self.Vu_t = TT.dot(self.Z_t, self.u_t)
        self.Vw_t = TT.dot(self.X_t, self.w_t)

        self.uloss_t = self.td_error_t(self.Vu_t)
        self.wloss_t = self.td_error_t(self.Vw_t)

        self.compile_theano_funcs()
    
    # XXX add transition and reward error
    @property
    def D(self):
        return numpy.dot(numpy.dot(self.Phi.T, self.C), self.Phi)

    @property
    def c(self):
        return numpy.dot(self.Phi.T, self.b)
        
    @property
    def C(self):
        return self.S - self.gam * self.Sp        
    
    @property
    def params_t(self):
        return [self.Phi_t]

    @property
    def params(self):
        return [self.Phi]

    @property
    def flat_params(self, params=None):
        return self._flatten(self.params)
    
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

    def compile_theano_funcs(self): 
        loss_vars = [self.Phi_t, self.S_t, self.Sp_t, self.b_t, self.X_t, self.R_t]
        self.theano_loss = theano.function(loss_vars, 
                                self.uloss_t + self.regularization_t()) # on_unused_input='ignore')

        grad = theano.grad(self.uloss_t, self.params_t) # , disconnected_inputs='ignore'
        self.theano_grad = theano.function(loss_vars, grad) # on_unused_input='ignore')

    def update_statistics(self, X, R):

        self.S += numpy.dot(X[:-1].T, X[:-1])
        self.Sp += numpy.dot(X[:-1].T, self.gam * X[1:])
        self.b += numpy.dot(X[:-1].T, R[:-1])
        self.updated_u = False
        self.updated_w = False

    def get_model_value(self, X):
        if not self.updated_u:
            self.u = scipy.linalg.solve(self.D + self.model_shift, self.c)
            self.updated_u = True
        return numpy.dot(self.encode(X), self.u)

    def get_lstd_value(self, X):
        if not self.updated_w:
            self.w = scipy.linalg.solve(self.C + self.lstd_shift,  self.b)
            self.updated_w = True
        return numpy.dot(X, self.w)

    def td_error(self, v, r):
        return numpy.sum((r[:-1] + self.gam * v[1:] - v[:-1])**2)    

    def td_error_t(self, V):
        ''' temporal difference (Bellman) error for given value function V wrt R '''
        return TT.sum((self.R_t[:-1] + self.gam * V[1:] - V[:-1])**2)
    
    def encode(self, x):
        return numpy.dot(x, self.Phi)

    def encode_t(self, x):
        return TT.dot(x, self.Phi_t)

    def regularization_t(self):
        return self.l1 * TT.sum(self.Phi_t**2) #  TT.sum(abs(self.Phi_t)) s

    def get_args(self, params, X, R):
        params = params if (type(params) == list) else self._unpack_params(params)        
        return params + [self.S, self.Sp, self.b, X, R]

    def optimize_loss(self, params, X, R):
        return self.theano_loss(*self.get_args(params, X, R))

    def optimize_grad(self, params, X, R):
        grad = self.theano_grad(*self.get_args(params, X, R))
        return self._flatten(grad)
