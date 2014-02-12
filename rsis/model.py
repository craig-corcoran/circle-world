import numpy
import scipy
import theano
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA

theano.config.warn.subtensor_merge_bug = False


class BKS_LSTD(object):
    
    ''' Bellman Krylov Subspace LSTD. Finds low-d krylov subspace, then solves
    for the value function in this low-d space'''

    def __init__(self,
                d,
                k,
                h,
                gam,  
                l2d,
                l2k,
                ):

        self.d = d
        self.k = k
        self.h = h
        self.gam = gam
        self.l2d = l2d
        self.l2k = l2k

        self.S = numpy.zeros((d,d))
        self.Sp = numpy.zeros((d,d))
        self.b = numpy.zeros(d)
        self.A = numpy.zeros((self.d, self.d))

        self.dshift = numpy.identity(d) * self.l2d
        self.kshift = numpy.identity(k) * self.l2k
        
        self.Phi = None        
        self.w = None
        self.u = None
        
        self.updated = {'u':False, 'w':False}
    
    @property
    def C(self):
        return self.S - self.gam * self.Sp 

    @property
    def D(self):
        return numpy.dot(numpy.dot(self.Phi.T, self.C), self.Phi)

    @property
    def a(self):
        return numpy.dot(self.Phi.T, self.b)

    def encode(self, x):
        return numpy.dot(x, self.Phi)

    def update_statistics(self, X, R):

        self.S += numpy.dot(X[:-1].T, X[:-1])
        self.Sp += numpy.dot(X[:-1].T, self.gam * X[1:])
        self.b += numpy.dot(X[:-1].T, R[:-1])
        
        n_rows = min(self.h, X.shape[0])
        Q = numpy.zeros((n_rows, self.d))
        for i in xrange(n_rows):
            # each row of Q is the inner product of r_t+i with x_t
            Q[i,:] = numpy.dot(R[i:], X if i is 0 else X[:-i]) / len(R[i:])

        self.A += numpy.dot(Q.T, Q) / n_rows
        self.updated.update({'u':False, 'w':False})

    def set_phi(self):
        
        vals, vecs = numpy.linalg.eigh(self.A)
        vecs = vecs[:, numpy.argsort(vals)[::-1]]
        self.Phi = vecs[:, :self.k]

    def get_model_value(self, X):
        
        if not self.updated['u']:
            self.set_phi()
            self.u = scipy.linalg.solve(self.D + self.kshift, self.a)
            self.updated['u'] = True

        return numpy.dot(self.encode(X), self.u)

    def get_lstd_value(self, X):

        if not self.updated['w']:
            self.w = scipy.linalg.solve(self.C + self.dshift,  self.b)
            self.updated['w'] = True

        return numpy.dot(X, self.w)

    def td_error(self, v, r):
        return numpy.sqrt(#min(numpy.sum((r[:-1])**2), 
                              numpy.sum((r[:-1] + self.gam * v[1:] - v[:-1])**2))#)
                                

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
        self.l2d = l2d
        self.l2k = l2k

        self.Phi = Phi if Phi else init_scale*numpy.random.standard_normal((d,k)) 
        self.S = S if S else numpy.zeros((d,d))
        self.Sp = Sp if Sp else numpy.zeros((d,d))
        self.b = b if b else numpy.zeros(d)

        self.dshift = numpy.identity(d) * self.l2d
        self.kshift = numpy.identity(k) * self.l2k

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
       
        self.dshift_t = TT.sharedvar.scalar_constructor(self.l2d) * TT.identity_like(self.S_t)        
        self.kshift_t = TT.sharedvar.scalar_constructor(self.l2k) * TT.identity_like(TT.dot(self.Phi_t.T, self.Phi_t))        
        self.Z_t = self.encode_t(self.X_t) # encode X into low-d state Z
        self.C_t = self.S_t - self.gam * self.Sp_t
        self.w_t = TT.dot(LA.matrix_inverse(self.C_t + self.dshift_t), self.b_t)
        #self.w_t = LA.solve(self.C_t, self.b_t)

        self.D_t = TT.dot(TT.dot(self.Phi_t.T, self.C_t), self.Phi_t) # D = Phi^T C Phi
        self.c_t = TT.dot(self.Phi_t.T, self.b_t)
        #self.u_t = LA.solve(self.D_t, self.c_t) 
        self.u_t = TT.dot(LA.matrix_inverse(self.D_t + self.kshift_t), self.c_t)

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
            self.u = scipy.linalg.solve(self.D + self.kshift, self.c)
            self.updated_u = True
        return numpy.dot(self.encode(X), self.u)

    def get_lstd_value(self, X):
        if not self.updated_w:
            self.w = scipy.linalg.solve(self.C + self.dshift,  self.b)
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
