import numpy
import scipy
import theano
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA
import matplotlib.pyplot as plt

theano.config.warn.subtensor_merge_bug = False

class TD(object):
    
    def __init__(self, 
                d, 
                gam, 
                l2_td,
                ):
        self.d = d        
        self.gam = gam
        self.l2_lstd = l2_lstd
        

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
        self.lstd_shift = numpy.identity(self.d) * self.l2_lstd
        self.w = None

        self.updated = {}
        self.reset_updated()
        self.ntot = 0

    @staticmethod
    def running_avg(A, x, n, np):
        return (A * n + x) / np

    @property
    def C(self):
        return self.S - self.gam * self.Sp

    def reset_updated(self):
        self.updated.update({'w':False, 'w_k':False, 
                             'q':False, 'q_k':False,
                             'T':False, 'T_k':False})

    def update_statistics(self, X, R, reset = False):
        
        if reset:
            self.S = numpy.zeros(self.S.shape)
            self.Sp = numpy.zeros(self.Sp.shape)
            self.b = numpy.zeros(self.n.shape)
            self.ntot = 0
            
        nadd = len(R)

        self.S = self.running_avg(self.S, 
                             numpy.dot(X[:-1].T, X[:-1]), 
                             self.ntot, 
                             self.ntot+nadd)
        
        self.Sp = self.running_avg(self.Sp, 
                              numpy.dot(X[:-1].T, self.gam * X[1:]),
                              self.ntot,
                              self.ntot+nadd)
                              
        self.b = self.running_avg(self.b,
                             numpy.dot(X.T, R),
                             self.ntot,
                             self.ntot+nadd)

        self.ntot += nadd
        self.reset_updated()

    def get_lstd_value(self, X, k=None):
        if k:
            if not self.updated['w_k']:
                u,d,v = scipy.sparse.linalg.svds(self.C + self.lstd_shift, k=self.k, which='LM')
                c_inv = numpy.dot(numpy.dot(v.T, numpy.diag(1./d)), u.T) # XXX plot svd features
                self.w_k = numpy.dot(c_inv, self.b)   
                self.updated['w_k'] = True

            return numpy.dot(X, self.w_k)

        else:
            if not self.updated['w']:
                self.w = scipy.linalg.solve(self.C + self.lstd_shift,  self.b)
                self.updated['w'] = True

            return numpy.dot(X, self.w)

    def get_lstd_reward(self, X, k=None):
        if k:
            if not self.updated['q_k']: # use k-svd for inverse
                u,d,v = scipy.sparse.linalg.svds(self.S + self.lstd_shift, k=k, which='LM')
                s_inv = numpy.dot(numpy.dot(v.T, numpy.diag(1./d)), u.T) # XXX store s_inv
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
                u,d,v = scipy.sparse.linalg.svds(self.S + self.lstd_shift, k=k, which='LM')
                s_inv = numpy.dot(numpy.dot(v.T, numpy.diag(1./d)), u.T) 
                self.T_k = numpy.dot(s_inv, self.Sp)   
                self.updated['T_k'] = True
                
            return numpy.dot(X, self.T_k)

        else:
            if not self.updated['T']:
                self.T = scipy.linalg.solve(self.S + self.lstd_shift, self.Sp)
                self.updated['T'] = True
            
            return numpy.dot(X, self.T)

    def td_error(self, v, r):
        return numpy.sqrt(numpy.sum((r[:-1] + self.gam * v[1:] - v[:-1])**2)) / (len(r) - 1)

    def lstd_reward_error(self, x, r, k=None):
        rhat = self.get_lstd_reward(x, k=k)
        return numpy.sqrt(numpy.sum((rhat - r)**2)) / len(r)

    def lstd_transition_error(self, x, k=None):
        xhat = self.get_lstd_transition(x[:-1], k=k)
        return numpy.sqrt(numpy.sum((xhat - x[1:])**2)) / len(xhat)


class Reconstructive_LSTD(Base_LSTD):
        
    ''' uses rica-like loss to learn a representation that captures the subspace
    spanned by r_t, r_t+1, ... '''

    def __init__(self, *args, **kwargs):
        
        self.k = args[0] #  number of columns of U
        
        self.states = kwargs.pop('states') # full state/reward sequence, subsampled during learning
        self.rewards = kwargs.pop('rewards')
        
        self.n = kwargs.pop('n', 100) # size of 'image' (subsample from statespace) input into encoding
        self.m = kwargs.pop('m', 100) # size of minibatch: number of images used for estimating gradient
        
        self.reg_loss = kwargs.pop('reg_loss', ('l2',1e-4)) # l2 regularization of U with reconstructive loss
        self.l2_subs = kwargs.pop('l2_subs', 1e-4) # l2 regularization when doing lstd on the subspace of U
        self.subs_shift = numpy.identity(self.k) * self.l2_subs
        
        self.g, self.g_t = kwargs.pop('g', (lambda x: x,)*2) # encoding transformation (default is identity)
        init_scale = kwargs.pop('init_scale', 1e-4)
        self.eps = kwargs.pop('eps', 1e-4) # used to determine the horizon h w/ gam
        max_h = kwargs.pop('max_h', None)

        super(Reconstructive_LSTD, self).__init__(*args[1:], **kwargs)

        self.U = init_scale*numpy.random.randn(self.d, self.k)
        self.n_samples = len(self.rewards)
        horiz = int(numpy.log(self.eps) / numpy.log(self.gam)) # determine horizon where gam^h < eps
        self.h = min(horiz, max_h) if max_h else horiz 
                
        self.U_t = TT.dmatrix('U')
        self.X_t = TT.dtensor3('X')
        self.R_t = TT.dmatrix('R') # broadcastable
        self.H_t = self.g_t(TT.sum(TT.tensordot(self.U_t.T, self.X_t, 1) * self.R_t, axis=2)) # kxm matrix of hidden unit activations

        self.compile_theano_funcs()
        self.update_statistics(self.states, self.rewards)
        self.reset_model_updated()

    def reset_model_updated(self):
        self.updated.update(
                    {'model-stats':False, 'wz':False, 'qz':False, 'Tz':False})

    def encode(self, x):
        return self.g(numpy.dot(x, self.U))

    def decode(self, z):
        return numpy.dot(z, self.U.T)
        
    def compile_theano_funcs(self): 
        reg_str, reg_val = self.reg_loss
        if reg_str == 'l2':        
            reg =  TT.sum(self.U_t**2)
        elif reg_str == 'l1':
            reg = TT.sum(abs(self.H_t))
        else: assert False
        loss = self.loss_t() + reg_val * reg
        loss_vars = [self.U_t, self.X_t, self.R_t]
        self.theano_loss = theano.function(loss_vars, loss) # on_unused_input='ignore')

        grad = theano.grad(loss, self.U_t) # , disconnected_inputs='ignore'
        self.theano_grad = theano.function(loss_vars, grad) # on_unused_input='ignore')

    def loss_t(self):
        # equiv to sum_i || Xi^T U g( U^T Xi r_i) - r_i ||^2
        I = TT.dot(self.U_t, self.H_t).T # H is [k,m]; U is [d,k]; I is [m,d]
        Rhat = TT.sum(TT.mul(self.X_t.T, I), axis=2).T
        return TT.sum((Rhat - self.R_t)**2)

    def optimize_loss(self, u, X, R):
        U = numpy.reshape(u, (self.d, self.k))
        return self.theano_loss(U, X, R)

    def optimize_grad(self, u, X, R):
        U = numpy.reshape(u, (self.d, self.k))
        grad = self.theano_grad(U, X, R)
        return grad.flatten()

    def set_params(self, u):
        self.U = numpy.reshape(u, (self.d, self.k))
        self.reset_model_updated()

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
        i = numpy.random.randint(self.h+1) # uniform dist over horiz lengths 

        # sampling with replacement until nonzero reward signal
        t = numpy.random.randint(self.n_samples-i, size=n) 
        while len(self.rewards[t+i].nonzero()[0]) == 0:
            t = numpy.random.randint(self.n_samples-i, size=n) 
        
        x = self.states[t]
        r = self.rewards[t+i]

        return x, r

    def update_model_statistics(self):
        if not self.updated['model-stats']:
            Z = self.encode(self.states)
            self.S_z = numpy.dot(Z.T, Z) / self.n_samples
            self.Sp_z = numpy.dot(Z[:-1].T, Z[1:]) / (self.n_samples-1)
            self.b_z = numpy.dot(Z.T, self.rewards) / self.n_samples
            self.updated['model-stats'] = True

    def get_model_value(self, X):
        
        self.update_model_statistics()
        
        if not self.updated['wz']:
            Z = self.encode(self.states)
            C = numpy.dot(Z[:-1].T, (Z[:-1] - self.gam * Z[1:]))
            b = numpy.dot(Z.T, self.rewards)
            self.wz = scipy.linalg.solve(C + self.subs_shift,  b)
            self.updated['wz'] = True

        return numpy.dot(self.encode(X), self.wz)

    def model_reward_error(self, x, r):

        self.update_model_statistics()        

        if not self.updated['qz']:
            #self.q = scipy.linalg.lstsq(self.states, self.rewards)[0]
            self.qz = scipy.linalg.solve(self.S_z + self.subs_shift, self.b_z)
            self.updated['qz'] = True
        
        rhat = numpy.dot(self.encode(x), self.qz) 
        return numpy.sqrt(numpy.sum((rhat - r)**2)) / len(r)

    def model_transition_error(self, x, Tz = None): 

        self.update_model_statistics()  

        if not self.updated['Tz']:
            # self.T = scipy.linalg.lstsq(self.states[:-1], self.states[1:])[0]
            self.Tz = scipy.linalg.solve(self.S_z + self.subs_shift, self.Sp_z)
            self.updated['Tz'] = True

        xhat = self.decode(numpy.dot(self.encode(x[:-1]),self.Tz))
        return numpy.sqrt(numpy.sum((xhat - x[1:])**2)) / len(x)  

