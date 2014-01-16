import numpy
from numpy.random import standard_normal


class CircleWorld(object):
    """ A unit circle domain where samples consist of (x,y) positions on the
    circle and rewards. The rewards and transitions (rotations) are linear in
    the position z with gaussian noise.  """

    def __init__(self, theta = numpy.pi/numpy.e**numpy.pi, eps_z = 1e-2, eps_th = 1e-3, eps_r = 1e-6):
        self.z = standard_normal(2) # random initial position
        self.z /= numpy.linalg.norm(self.z)
        self.q = numpy.array([1,1])
        self.theta = theta # roatation angle mean/bias

        self.eps_z = eps_z
        self.eps_th = eps_th
        self.eps_r = eps_r
    
    @staticmethod
    def _rotation(theta):
        return  numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                              [numpy.sin(theta), numpy.cos(theta)]])

    def get_samples(self, n):
        ''' return n samples of the state and reward '''
        states = numpy.empty((n, 2))
        rewards = numpy.empty(n)
        for i in xrange(n):
            T = self._rotation(self.theta + self.eps_th*standard_normal())
            self.z = numpy.dot(T, self.z) 
            self.z += self.eps_z*standard_normal()*numpy.linalg.norm(self.z)*self.z
            states[i] = self.z
            rewards[i] = numpy.dot(self.q, self.z) + self.eps_r*standard_normal()

        return states, rewards

