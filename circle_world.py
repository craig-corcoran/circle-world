import matplotlib.gridspec as gridspec
import itertools as it
import numpy
from numpy import array, dot
from numpy.random import standard_normal
import matplotlib.pyplot as plt

# Dataset
# Model
# Environment

# email leif, kate
# literature review
# post writeup
# contact profs?

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


class RLDataset(object):
    ''' A collection of features X and rewards R for a sequence of states'''
    
    def __init__(self, X, R):
        self.X = R

    @property
    def states(self):
        '''Returns a feature representation of the state sequence'''
        return self.X
    
    @property
    def rewards(self):
        return self.R






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

        
def main(n=100, N = 15, n_plot = 64):
    
    cworld = CircleWorld()
    P, R = cworld.get_samples(n)

    fmap = FourierFeatureMap(N)
    # transform positions into feature representation
    X = fmap.transform(P)

    data = RLDataset(X,R)

    model = ROML()
    model.train(data)
    model.evaluate()

def view_fourier_basis(N = 10, n_plot = 64, 
                       shuffle = False, last = False, sin = False):

    # plot a regular grid
    P = numpy.reshape(numpy.mgrid[-1:1:N*1j,-1:1:N*1j], (2,N*N)).T
    fmap = FourierFeatureMap(N, sin)
    X = fmap.transform(P) 

    print X[:,0]
    print X[:,9]
    
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


if __name__ == '__main__':
    view_fourier_basis()
    #main()

