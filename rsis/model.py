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
                l1 = 1e-3, # l1 regularization constant
                init_scale = 1e-1,
                ):
        self.k = k
        self.d = d
        self.Phi = init_scale*numpy.random.standard_normal((d,k)) if Phi is None else Phi
        #self.Phi = numpy.zeros((d,k)) if Phi is None else Phi
        self.T = numpy.identity(k) if T is None else T
        self.q = init_scale*numpy.random.standard_normal(k) if q is None else q
        #self.q = numpy.zeros(k) if q is None else q
        #self.w = numpy.random.standard_normal(k) if w is None else w
        self.Mz = numpy.identity(k) if Mz is None else Mz
        #self.Sz = numpy.dot(self.Mz,self.Mz.T)
        self.sr = numpy.array(1.) if sr  is None else sr
        self.inv_shift = shift*numpy.identity(k)
        self.l1 = l1
        self.init_scale = init_scale
        self.param_names = ['Phi', 'T', 'q', 'Mz', 'sr']

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
    
        # compute theano loss function and add l1 regularization
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

            #self.Sz = numpy.dot(self.Mz,self.Mz.T)

    def _unpack_params(self, vec):
        i = 0
        params = []
        for s in self.shapes:
            j = i + numpy.product(s)
            params.append(vec[i:j].reshape(s))
            i = j
        return params

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


    def _loss_t(self):
        ''' Generates the theano loss variable '''
        #return self.R_t - TT.dot(self.Z_t, self.q_t)
        rerr = TT.sum(TT.sqr(self.R_t - self._reward_t(self.Z_t)))/self.sr_t**2
        zerr_v = self.Z_t[1:] - self._transition_t(self.Z_t[:-1])
        zerr_vp = TT.dot(zerr_v, LA.matrix_inverse(self.Sz_t+self.inv_shift_t))
        zerr = TT.sum(TT.mul(zerr_v, zerr_vp))

        n = TT.sum(TT.ones_like(self.R_t))
        norm = (n-1)*TT.log(LA.det(self.Sz_t+self.inv_shift_t)) + 2*n*TT.log(self.sr_t)

        reg = self.l1 * sum(TT.sum(abs(p)) for p in self.theano_params)

        return TT.sum(zerr + rerr + norm) / (n-1) + reg

    def _loss(self, X, R, debug = False):
       
        ''' numpy version of loss function '''

        Z = self.encode(X)
        rerr = numpy.sum((R - self.reward(Z))**2)/self.sr**2
        zerr_v = (Z[1:] - self.transition(Z[:-1])) #n-1 by k
        zerr_vp = numpy.dot(zerr_v, numpy.linalg.inv(self.Sz+self.inv_shift)) #n-1 by k
        zerr = numpy.sum(numpy.multiply(zerr_vp, zerr_v))

        # TODO remove, move to tests
        if debug:
            inv = numpy.linalg.inv(self.Sz)
            other_zerr = 0.
            for ii in xrange(Z.shape[0]-1):
                v = Z[ii+1,:] - self.transition(Z[ii,:])
                vp = numpy.dot(v, inv)
                other_zerr += numpy.dot(v, vp.T)

            assert abs(zerr - other_zerr) < 1e-10

        n = Z.shape[0]
        norm = (n-1)*numpy.log(numpy.linalg.det(self.Sz+self.inv_shift)) + 2*n*numpy.log(self.sr)

        reg = self.l1 * sum(numpy.sum(abs(p)) for p in self.params)

        return (rerr + zerr + norm) / (n-1) + reg

    def loss(self, X, R):
        ''' callable, takes array of features X and rewards R and returns the
        loss given the current set of parameters. Examples through time are
        indexed by row '''

        return self.theano_loss(self.Phi, self.T, self.q, self.Mz, self.sr, X, R)

    def unscaled_losses(self, X, R):
        ''' returns a list of loss components, not scaled by the guassian
        parameters'''

        Z = self.encode(X)
        rerr = numpy.sum((R - self.reward(Z))**2)
        zerr =  numpy.sum(Z[1:] - self.transition(Z[:-1]))**2
        n = Z.shape[0]
        norm = (n-1)*numpy.log(numpy.linalg.det(self.Sz)) + 2*n*numpy.log(self.sr)

        reg = self.l1 * sum(numpy.sum(abs(p)) for p in self.params)
        
        return rerr/(n-1), zerr/(n-1), norm/(n-1), reg

    def grad(self, X, R):
        ''' returns gradient at the current parameters with the given inputs X
        and R. '''

        return self.theano_grad(self.Phi, self.T, self.q, self.Mz, self.sr, X, R)

    def optimize_loss(self, params, X, R):
        unpacked = self._unpack_params(params)
        #print zip(self.param_names, unpacked)
        return self.theano_loss(*(unpacked + [X, R]))

    def optimize_grad(self, params, X, R):
        Phi, T, q, Mz, sr = self._unpack_params(params)
        grad = self.theano_grad(Phi, T, q, Mz, sr, X, R)
        return self._flatten(grad)

    def reset_nans(self):
        
        for i, name in enumerate(self.param_names): 
            p = getattr(self, name)
            if numpy.isnan(p).any():
                if name in ['Phi', 'q']:
                    setattr(self, name, self.init_scale * numpy.random.standard_normal(numpy.shape(p)))
                else:
                    if p.shape == ():                    
                        setattr(self, name, numpy.array(1.))
                    else:
                        setattr(self, name, numpy.identity(p.shape))


class CD_RSIS(RSIS):


    def __init__(self, *args, **kwargs):

        super(CD_RSIS, self).__init__(*args, **kwargs)
    
        self._Sz = numpy.identity(self.k)
        self.Sz_t = TT.dmatrix('Sz')
        self.param_names = ['Phi', 'T', 'q']
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

    def set_noise_params(self, X, R):
        
        Z = self.encode(X)
        n = Z.shape[0]
        Rerr = R - self.reward(Z)
        Zerr = Z[1:] - self.transition(Z[:-1])
        self.sr = numpy.sum(Rerr**2) / n
        self._Sz = numpy.dot(Zerr.T, Zerr) / (n-1)

    def loss(self, X, R):
        ''' callable, takes array of features X and rewards R and returns the
        loss given the current set of parameters. Examples through time are
        indexed by row '''

        return self.theano_loss(self.Phi, self.T, self.q, self.Sz, self.sr, X, R)

    def grad(self, X, R):
        ''' returns gradient at the current parameters with the given inputs X
        and R. '''

        return self.theano_grad(self.Phi, self.T, self.q, self.Sz, self.sr, X, R)

    def optimize_loss(self, params, X, R):
        unpacked = self._unpack_params(params)
        return self.theano_loss(*(unpacked + [self.Sz, self.sr, X, R]))

    def optimize_grad(self, params, X, R):
        unpacked = self._unpack_params(params)
        grad = self.theano_grad(*(unpacked + [self.Sz, self.sr, X, R]))
        return self._flatten(grad)



    

