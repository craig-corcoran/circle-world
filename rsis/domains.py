import numpy as np
import numpy.random as rng


class CircleWorld(object):
    """ A unit circle domain where samples consist of (x,y) positions on the
    circle and rewards. The rewards and transitions (rotations) are linear in
    the position z with gaussian noise.  """

    def __init__(self, theta = 16*np.pi/np.e**np.pi, gam = 1-1e-2, eps_z = 1e-2, eps_th = 1e-3, eps_r = 1e-6):
        self.z = self.init_z
        self.q = np.array([1,1])
        self.theta = theta # roatation angle mean/bias
        self.T = CircleWorld.rotation(self.theta)  # expected transition/rotation matrix 
        self.gam = gam

        self.eps_z = eps_z
        self.eps_th = eps_th
        self.eps_r = eps_r

        self.w = np.linalg.solve(np.identity(self.T.shape[0]) - self.gam * self.T.T, self.q)
    
    @staticmethod
    def rotation(theta):
        return  np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    
    @property
    def init_z(self):
        z = rng.randn(2) # random initial position w/ unit norm
        z /= 2*np.linalg.norm(z)
        return z

    def value_func(self, x):
        ''' expected value function '''
        return np.dot(x, self.w)

    def reward_func(self, x):
        ''' expected reward function '''
        return np.dot(x, self.q)

    def get_samples(self, n, reset_state = True):
        ''' return n samples of the state and reward '''
        if reset_state:
            self.z = self.init_z

        states = np.empty((n, 2))
        rewards = np.empty(n)
        for i in xrange(n):
            T = CircleWorld.rotation(self.theta + self.eps_th*rng.randn())
            self.z = np.dot(T, self.z) 
            self.z += self.eps_z*rng.randn()*np.linalg.norm(self.z)*self.z
    
            # the state can't leave the unit circle
            if np.sum(self.z**2) > 1: 
                self.z /= np.linalg.norm(self.z)

            states[i] = self.z
            rewards[i] = np.dot(self.q, self.z) + self.eps_r*rng.randn()

        return states, rewards


class TorusWorld(object):

    def __init__(self, reward = 0.1, eps_z = 0.05, eps_R = 0., eps_T = 0.02):
        self.reward = reward
        self.eps_z = eps_z
        self.R = np.eye(2) + eps_R * rng.randn(2, 2)
        self.T = eps_T * (np.ones(2) + 0.1 * rng.randn(2))
        self.g = np.zeros(2)

    def _reward(self, p):
        return int(np.linalg.norm(p - self.g) < self.reward)

    def reward_func(self, x):
        return np.array([self._reward(x[i]) for i in xrange(x.shape[0])])

    def get_samples(self, n):
        z = (rng.randn(2) + 1) % 2 - 1
        states = np.empty((n, 2))
        rewards = np.empty(n)
        for i in xrange(n):
            z = (self.eps_z * rng.randn(2) + np.dot(self.R, z + self.T) + 1) % 2 - 1
            states[i] = z
            rewards[i] = self._reward(z)

        return states, rewards
