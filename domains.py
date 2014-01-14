import numpy
from numpy.random import standard_normal


class CircleWorld(object):
    """ A unit circle domain where samples consist of (x,y) positions on the
    circle and rewards. The rewards and transitions (rotations) are linear in
    the position z with gaussian noise.  """

    def __init__(self, theta = numpy.pi/17.7, eps_z = 1e-8, eps_r = 1e-3):
        self.z = standard_normal(2) # random initial position
        self.z /= numpy.linalg.norm(self.z)
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

