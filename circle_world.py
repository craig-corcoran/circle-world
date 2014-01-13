import time
import matplotlib.gridspec as gridspec
import itertools as it
import numpy
from numpy import array, dot
from numpy.random import standard_normal
import matplotlib.pyplot as plt
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA
import theano

class CircleWorld(object):
    """ A unit circle domain where samples consist of (x,y) positions on the
    circle and rewards. The rewards and transitions (rotations) are linear in
    the position z with gaussian noise.  """
    
    def __init__(self, theta = numpy.pi/12, eps_z = 1e-3, eps_r = 1e-3):
        self.z = standard_normal(2) # random initial position            
        self.q = numpy.array([1,0])
        self.T = numpy.array([[numpy.cos(theta), -numpy.sin(theta)], 
                              [numpy.sin(theta), numpy.cos(theta)]])

        self.eps_z = eps_z
        self.eps_r = eps_r

    def get_samples(self, n):
        ''' return n samples of the state and reward '''
        states = numpy.empty((n, 2))
        rewards = numpy.empty(n)
        for i in xrange(n):
            states[i] = numpy.dot(self.T, self.z) + self.eps_z*standard_normal(2)
            rewards[i] = numpy.dot(self.q, self.z) + self.eps_r*standard_normal() 
            self.z = states[i]

        return states, rewards


class ROML(object):
    
    def __init__(self,
                d, # dim of state rep (initial features)
                k, # dim after linear transorm
                Phi = None, # feature transform matrix f(x) = Phi x; [dxk]
                T = None, # transition matrix; Tz_t = z_t+1
                q = None, # reward function weights; r(x) = q' Phi x 
                w = None, # value function weights; v(x) = w' Phi x
                Mz = None, # transition noise, kxl where l <= k
                sr = None # reward noise
                ):
            
        self.Phi = 1e-2*numpy.random.standard_normal((d,k)) if Phi is None else Phi
        self.T = numpy.identity(k) if T is None else T
        self.q = numpy.random.standard_normal(k) if q is None else q
        self.w = numpy.random.standard_normal(k) if w is None else w
        self.Mz = numpy.identity(k) if Mz is None else Mz
        self.Lz = numpy.dot(self.Mz,self.Mz.T)
        self.sr = 1. if sr  is None else sr

        self.Phi_t = TT.dmatrix('Phi')
        self.T_t = TT.dmatrix('T')
        self.q_t = TT.dvector('q')
        self.Mz_t = TT.dmatrix('Mz') 
        self.Lz_t = TT.dot(self.Mz_t,self.Mz_t.T) # forces Lz to be positive semidefinite
        self.sr_t = TT.dscalar('sr')
        self.X_t = TT.dmatrix('X')
        self.R_t = TT.dvector('R')        
        self.Z_t = self._encode_t(self.X_t) # encode X into low-d state Z
    
        loss_t = self._loss_t()
        self.theano_loss = theano.function(self.theano_vars, loss_t, on_unused_input='ignore')
        
        grad = theano.grad(loss_t, self.theano_params)
        self.theano_grad = theano.function(self.theano_vars, grad)


    @property
    def theano_vars(self):
        return [self.X_t, self.R_t, self.Phi_t, self.T_t, self.q_t, self.Mz_t, self.sr_t]


    @property
    def theano_params(self):
        return [self.Phi_t, self.T_t, self.q_t, self.Mz_t, self.sr_t]


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
        zerr_vp = numpy.dot(zerr_v, self.Lz) #n-1 by k
        zerr = numpy.sum(numpy.multiply(zerr_vp,zerr_v))
        n = Z.shape[0]
        norm = (n-1)*numpy.log(numpy.linalg.det(numpy.linalg.inv(self.Lz))) \
               + 2*n*numpy.log(self.sr)

        return rerr + zerr + norm

    def _loss_t(self):
        ''' Generates the theano loss variable '''
        #return self.R_t - TT.dot(self.Z_t, self.q_t)
        rerr = TT.sum(TT.sqr(self.R_t - self._reward_t(self.Z_t)))/self.sr_t**2
        zerr_v = self.Z_t[1:] - self._transition_t(self.Z_t[:-1])
        zerr_vp = TT.dot(zerr_v, self.Lz_t)
        zerr = TT.sum(TT.mul(zerr_v, zerr_vp))
        
        n = TT.sum(TT.ones_like(self.R_t))
        norm = (n-1)*TT.log(LA.det(LA.matrix_inverse(self.Lz_t))) \
               + 2*n*TT.log(self.sr_t)

        return TT.sum(zerr + rerr + norm)

    def _loss(self, X, R):
        ''' slightly slower theano version of loss function '''

        return self.theano_loss(X, R, self.Phi, self.T, self.q, self.Mz, self.sr)

    def grad(self, X, R):
        ''' returns gradient at the current parameters with the given inputs X
        and R. '''

        return self.theano_grad(X, R, self.Phi, self.T, self.q, self.Mz, self.sr)


    def train(self, dataset):
        pass



class FourierFeatureMap(object):
    
    def __init__(self, N, sin = False):
        # Implicit dividing freqs by two here bc range is -1, 1
        freqs = numpy.pi * numpy.arange(N,dtype=float) 
        self.W = numpy.array(list(it.product(freqs, freqs))).T
        self.sin = sin

    def transform(self,P):
        
        if self.sin:
            return numpy.sin(numpy.dot(P,self.W))  # complex component
        else: 
            return numpy.cos(numpy.dot(P,self.W))  # real component


def view_fourier_basis(N = 10, n_plot = 64, 
                       shuffle = False, last = False, sin = False):

    # plot a regular grid
    P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
    fmap = FourierFeatureMap(N, sin)
    X = fmap.transform(P) 

    print X[:,0] 
    print X[:,9] # XXX two constant functions, whats up with that?
    
    if shuffle: # shuffle the columns of X
        numpy.random.shuffle(numpy.transpose(X))

    plot_filters(X, n_plot, 'fourier_basis.png', last)
    
    
def plot_filters(X, n_plot, file_name = 'basis.png', last = False):
    
    plt.clf()
    n_sp = numpy.sqrt(n_plot)
    side = numpy.sqrt(X.shape[0])
    gs = gridspec.GridSpec(int(n_sp), int(n_sp))
    gs.update(wspace=0., hspace=0.)

    for i in xrange(n_plot):
        
        #ax = plt.subplot(n_sp,n_sp,i+1)
        ax = plt.subplot(gs[i])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = X[:,-i] if last else X[:,i]
        plt.imshow(numpy.reshape(im, (side,side)), cmap = 'gray')
        
    plt.tight_layout()
    plt.savefig(file_name)
    
    
def test_circleWorld(n=100, N = 100):
    
    cworld = CircleWorld()
    P, R = cworld.get_samples(n)

    # transform the samples into fourier feature space
    freqs = numpy.arange(N,dtype=float) / 2
    W = numpy.array(list(it.product(freqs, freqs))).T
    X = numpy.cos(numpy.dot(P,W)) # real component
    Y = numpy.exp(1j*numpy.dot(P,W)) # both real and imaginary
    
    assert (numpy.real(Y) == X).all()
    assert (numpy.imag(X) == 0).all()
    assert (abs(X) <= 1).all() 

def test_ROML(n=10, N = 10):
    
    # generate samples from circle world
    cworld = CircleWorld()
    P, R = cworld.get_samples(n)

    assert P.shape == (n,2)
    assert R.shape == (n,)
    
    # transform positions into fourier feature representation
    fmap = FourierFeatureMap(N)
    X = fmap.transform(P)

    assert X.shape == (n,N**2)

    model = ROML(X.shape[1], 2)

    # check that the theano loss and numpy loss are the same
    assert model.loss(X,R) == model._loss(X,R)
    
    # take gradient wrt [Phi, T, q, Mz, sr]
    shapes = map(numpy.shape, model.grad(X,R))
    assert shapes == [(N**2,2), (2,2), (2,), (2,2), ()]

def main(n=10, N = 10, k = 2):
    
    cworld = CircleWorld()
    P, R = cworld.get_samples(n)
    fmap = FourierFeatureMap(N)

    X = fmap.transform(P)
    assert X.shape == (n, N**2)
    
    model = ROML(N**2, k)

    print model.loss(X,R)
    print model._loss(X,R)
    print map(numpy.shape, model.grad(X,R))

if __name__ == '__main__':
    #test_circleWorld()
    #test_ROML()
    main()

