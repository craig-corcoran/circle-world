import numpy as np
import numpy.random as rng


class CircleWorld(object):
    """ A unit circle domain where samples consist of (x,y) positions on the
    circle and rewards. The rewards and transitions (rotations) are linear in
    the position z with gaussian noise.  """

    def __init__(self, theta = np.pi/np.e**np.pi, gam = 1-1e-2, eps_z = 1e-2, eps_th = 1e-3, eps_r = 1e-6):
        self.z = self.init_z
        self.q = np.array([1,1])
        self.theta = theta # roatation angle mean/bias
        self.T = self._rotation(self.theta)  # expected transition/rotation matrix 
        self.gam = gam

        self.eps_z = eps_z
        self.eps_th = eps_th
        self.eps_r = eps_r

        self.w = np.linalg.solve(np.identity(self.T.shape[0]) - self.gam * self.T.T, self.q)
    
    @staticmethod
    def _rotation(theta):
        return  np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    
    @property
    def init_z(self):
        z = np.random.standard_normal(2) # random initial position w/ unit norm
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
            T = self._rotation(self.theta + self.eps_th*np.random.standard_normal())
            self.z = np.dot(T, self.z) 
            self.z += self.eps_z*np.random.standard_normal()*np.linalg.norm(self.z)*self.z
    
            # the state can't leave the unit circle
            if np.sum(self.z**2) > 1: 
                self.z /= np.linalg.norm(self.z)

            states[i] = self.z
            rewards[i] = np.dot(self.q, self.z) + self.eps_r*np.random.standard_normal()

        return states, rewards


class TorusWorld(object):

    def __init__(self, reward = 0.1, eps_z = 1e-2, eps_t = 1e-1):
        self.reward = reward
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
            rewards[i] = int(np.linalg.norm(z - q) < self.reward)
        return states, rewards
