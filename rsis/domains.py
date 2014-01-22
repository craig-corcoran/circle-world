import numpy as np
import numpy.random as rng


class CircleWorld(object):
    """ A unit circle domain where samples consist of (x,y) positions on the
    circle and rewards. The rewards and transitions (rotations) are linear in
    the position z with gaussian noise.  """

    def __init__(self, theta = np.pi/np.e**np.pi, eps_z = 1e-2, eps_th = 1e-3, eps_r = 1e-6):
        self.theta = theta # roatation angle mean/bias

        self.eps_z = eps_z
        self.eps_th = eps_th
        self.eps_r = eps_r
    
    @staticmethod
    def rotation(theta):
        return  np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])

    def get_samples(self, n):
        ''' return n samples of the state and reward '''
        q = np.ones(2)
        z = rng.randn(2) # random initial position
        z /= np.linalg.norm(z)
        states = np.empty((n, 2))
        rewards = np.empty(n)
        for i in xrange(n):
            T = CircleWorld.rotation(self.theta + self.eps_th*rng.randn())
            z = np.dot(T, z)
            z += self.eps_z*rng.randn()*np.linalg.norm(z)*z
            states[i] = z
            rewards[i] = np.dot(q, z) + self.eps_r*rng.randn()

        return states, rewards


class TorusWorld(object):

    def __init__(self, eps_z = 1e-2, eps_t = 1e-1):
        self.eps_z = eps_z
        self.eps_t = eps_t

    def get_samples(self, n):
        q = np.zeros(2)
        z = (1 + rng.randn(2)) % 2 - 1
        T = self.eps_t * rng.randn(2, 2) + CircleWorld.rotation(0.2)
        states = np.empty((n, 2))
        rewards = np.empty(n)
        for i in xrange(n):
            z = (1 + np.dot(T, z) + self.eps_z * rng.randn(2)) % 2 - 1
            states[i] = z
            rewards[i] = 1 - np.linalg.norm(z - q)
        return states, rewards
