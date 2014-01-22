import numpy
from numpy.random import standard_normal


class CircleWorld(object):
    """ A unit circle domain where samples consist of (x,y) positions on the
    circle and rewards. The rewards and transitions (rotations) are linear in
    the position z with gaussian noise.  """

    def __init__(self, theta = numpy.pi/numpy.e**numpy.pi, gam = 1.-1e-4, eps_z = 4e-2, eps_th = 2e-3, eps_r = 1e-6):
        self.z = self.init_z
        self.q = numpy.array([1,1])
        self.theta = theta # roatation angle mean/bias
        self.T = self._rotation(self.theta)  # expected transition/rotation matrix 
        self.gam = gam

        self.eps_z = eps_z
        self.eps_th = eps_th
        self.eps_r = eps_r

        self.w = numpy.linalg.solve(numpy.identity(self.T.shape[0]) - self.gam * self.T, self.q)
    
    @staticmethod
    def _rotation(theta):
        return  numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                              [numpy.sin(theta), numpy.cos(theta)]])
    
    @property
    def init_z(self):
        z = standard_normal(2) # random initial position w/ unit norm
        z /= 2*numpy.linalg.norm(z)
        return z

    def value_func(self, x):
        ''' expected value function '''
        return numpy.dot(x, self.w)

    def reward_func(self, x):
        ''' expected reward function '''
        return numpy.dot(x, self.q)

    def get_samples(self, n, reset_state = True):
        ''' return n samples of the state and reward '''
        if reset_state:
            self.z = self.init_z

        states = numpy.empty((n, 2))
        rewards = numpy.empty(n)
        for i in xrange(n):
            T = self._rotation(self.theta + self.eps_th*standard_normal())
            self.z = numpy.dot(T, self.z) 
            self.z += self.eps_z*standard_normal()*numpy.linalg.norm(self.z)*self.z
    
            # the state can't leave the unit circle
            if numpy.sum(self.z**2) > 1: 
                self.z /= numpy.linalg.norm(self.z)

            states[i] = self.z
            rewards[i] = numpy.dot(self.q, self.z) + self.eps_r*standard_normal()

        return states, rewards

    

