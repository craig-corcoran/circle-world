import numpy as np
import numpy.random as rng


class CircleWorld(object):
    """ A unit circle domain where samples consist of (x,y) positions on the
    circle and rewards. The rewards and transitions (rotations) are linear in
    the position z with gaussian noise.  """

    def __init__(self, theta = 8*np.pi/np.e**np.pi, gam = 1-1e-2, eps_z = 1e-2, eps_th = 1e-3, eps_r = 1e-6):
        self.z = self.init_z
        self.q = np.array([1,1])
        self.theta = theta # roatation angle mean/bias
        self.T = CircleWorld.rotation(self.theta)  # expected transition/rotation matrix 
        self.gam = gam

        self.eps_z = eps_z
        self.eps_th = eps_th
        self.eps_r = eps_r

        self.w = np.linalg.solve(np.identity(self.T.shape[0]) - self.gam * self.T, self.q)
    
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

    def get_samples(self, n, reset_state = True, seed = None):
        ''' return n samples of the state and reward '''
        if reset_state:
            self.z = self.init_z

        if seed:
            rng.seed(seed)

        states = np.empty((n, 2))
        rewards = np.empty(n)
        for i in xrange(n):
            T = CircleWorld.rotation(self.theta + self.eps_th*rng.randn())
            self.z = np.dot(self.z, T) 
            self.z += self.eps_z*rng.randn()*np.linalg.norm(self.z)*self.z
    
            # the state can't leave the unit circle
            if np.sum(self.z**2) > 1: 
                self.z /= np.linalg.norm(self.z)

            states[i] = self.z
            rewards[i] = np.dot(self.q, self.z) + self.eps_r*rng.randn()

        return states, rewards


class TorusWorld(object):

    def __init__(self, reward_scal=10, reward_rad=0.2, eps_reward=0, eps_z=0.1, bias_scal=0.2):

        self.reward_rad = reward_rad
        self.reward_scal = reward_scal
        self.eps_z = eps_z # movement noise
        self.eps_reward = eps_reward
        self.bias = bias_scal*np.ones(2)
        self.g = np.zeros(2) # goal pos

    def _reward(self, p):
        return self.reward_scal if np.linalg.norm(p - self.g) < self.reward_rad \
                                else -0.1*self.reward_scal \
                                + self.eps_reward * np.random.randn()

    def reward_func(self, x):
        return np.array([self._reward(x[i]) for i in xrange(x.shape[0])])

    def get_samples(self, n, seed = None):

        if seed: rng.seed(seed)

        z = 2*rng.random(2)-1
        states = np.empty((n, 2))
        rewards = np.empty(n)
        for i in xrange(n):
            z = (z + self.bias + self.eps_z * rng.randn(2) + 1) % 2 - 1
            states[i] = z
            rewards[i] = self._reward(z)

        return states, rewards
