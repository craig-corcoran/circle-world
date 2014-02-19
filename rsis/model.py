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
                l2_lstd,
                ):
    
        self.d = d        
        self.gam = gam
        self.l2_lstd = l2_std

        self.S = numpy.zeros((self.d,self.d))
        self.Sp = numpy.zeros((self.d,self.d))
        self.b = numpy.zeros(self.d)
        self.w = None

        self.lstd_shift = numpy.identity(self.d) * self.l2_lstd
        self.updated = {'w':False}

    @property
    def C(self):
        return self.S - self.gam * self.Sp
    
    def update_statistics(self, X, R, reset_trace = False):

        self.S += numpy.dot(X[:-1].T, X[:-1])
        self.Sp += numpy.dot(X[:-1].T, self.gam * X[1:])
        self.b += numpy.dot(X[:-1].T, R[:-1])
        
        self.updated.update({'w':False})

    def get_lstd_value(self, X, k = None):

        if not self.updated['w']:
            
            if k:
                u,d,v = scipy.sparse.linalg.svds(self.C + self.lstd_shift, k=self.k, which='LM')
                c_inv = numpy.dot(numpy.dot(v.T, numpy.diag(1/d)), u.T) # XXX plot svd features
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
        
        self.k = args[0] #  number of columns of U
        self.states = kwargs.pop('states') # full state/reward sequence, subsampled during learning
        self.rewards = kwargs.pop('rewards')
        self.n = kwargs.pop('n', 100) # size of 'image' (subsample from statespace) input into encoding
        self.m = kwargs.pop('m', 100) # size of minibatch: number of images used for estimating gradient
        self.l2_subs = kwargs.pop('l2_subs', 1e-3) # l2 regularization when doing lstd on the subspace of U
        self.l2_loss = kwargs.pop('l2_loss', 1e-3) # l2 regularization of U with reconstructive loss
        self.eps = kwargs.pop('eps', 1e-4) # used to determine the horizon h w/ gam
        self.g = kwargs.pop('g', lambda x: x) # encoding transformation (default is identity)

        super(Reconstructive_LSTD, self).__init__(*args[1:-2], **kwargs)
        
        self.h = int(numpy.log(self.eps) / numpy.log(self.gam)) # determine horizon where gam^h < eps
        self.subs_shift = numpy.identity(self.k) * self.l2_subs
        self.U = numpy.zeros((self.d, self.k))
        
        self.U_t = TT.dmatrix('U')
        self.X_t = TT.dtensor3('X')
        self.R_t = TT.dmatrix('R')


    def compile_theano_funcs(self): 
        loss = self.loss_t()
        self.theano_loss = theano.function([self.U_t, self.X_t, self.R_t], 
                                    loss + self.l2_loss * TT.sum(self.U_t**2)) # on_unused_input='ignore')

        grad = theano.grad(loss, self.U_t) # , disconnected_inputs='ignore'
        self.theano_grad = theano.function(loss_vars, grad) # on_unused_input='ignore')

    def reconstruction_loss_t(self, X, r):
        '''reconstruction loss for a single vector r: ||W g(W^T r) - r||^2 '''

        W = TT.dot(X.T, self.U_t) 
        return TT.sum((TT.dot(W, self.g(TT.dot(W.T, r))) - r)**2)

    def loss_t(self):
        ''' accumulate reconstruction loss across a minibatch '''
        return theano.scan(
                        fn=lambda loss, x, r: loss + self.reconstruction_loss_t(x,r),
                        outputs_info=0.,
                        sequences=[self.X_t, self.R_t]
                        )[0][-1] 

    def optimize_loss(self, params, X, R):
        return self.theano_loss(self.get_args(params, X, R))

    def optimize_grad(self, params, X, R):
        grad = self.theano_grad(*self.get_args(params, X, R))
        return self._flatten(grad)

    def sample_minibatch(self):
        ''' sample a minibatch of reward images. each image is a random
        subsample of the state space with a constant temporal offset between
        the reward r and the corresponding state x '''
        
        x = numpy.empty((self.m, self.n, self.d)) r = numpy.empty((self.m,
        self.n)) for i in xrange(m): x[i], r[i] = self.sample_image(self.n)

        return x, r

    def sample_image(self, n):
        ''' picks a random time step difference, then randomly subsamples from
        the available samples (with replacement) '''
        i = numpy.random.randint(self.h+1) # XXX weight prob by gam**i? 
        t = numpy.random.randint(n-i+1, size=n) # ok to sample with replacement?
        x = self.states[t]
        r = self.rewards[t+i]

        return x, r


class RICA_LSTD(Base_LSTD):
    
    ''' linear reconstructive LSTD. First learns a subspace U and does l2 
    regularized LSTD in this space '''

    def __init__(self, *args, **kwargs):

        super(RICA_LSTD, self).__init__(*args[1:-1], **kwargs)

        self.k = args[0]
        self.l2_rica = args[-1]
        self.rica_shift = numpy.identity(self.k) * self.l2_rica
        self.U = numpy.zeros(self.d, self.k)
        self.l = 0c
        self.X_t = TT.dmatrix('X') # X and R for error
        self.R_t = TT.dvector('R')
       
        self.Z_t =  

    def loss_t(self):

        pass # XXX theano recursive/cumulative loss



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


    # RICA / Autoencoder loss version
                                

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
