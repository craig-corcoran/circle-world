import os
import numpy as np
import rsis
import rsis.experiments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools as it
from scipy.sparse.linalg import svds
from scipy.interpolate import griddata, Rbf, SmoothBivariateSpline
from scipy.optimize import fmin_cg, fmin_l_bfgs_b
from numpy.linalg import norm
import theano
import theano.tensor as TT
import theano.sandbox.linalg.ops as LA
from sklearn.decomposition import dict_learning_online, dict_learning, MiniBatchDictionaryLearning
import cPickle as pickle

def svd_PQ(H, k):

    U,s,V = svds(H, k=k)
    U=U[:,::-1]
    s=s[::-1]
    V=V[::-1,:].T

    P = U*np.sqrt(1/s)
    Q = V*np.sqrt(1/s)

    return P, Q

class SparseHankelDecomposition(object):

    def __init__(self, H, k, alpha):

        self.H = H
        self.d = H.shape[0]
        self.k = k
        self.alpha = alpha

        self.X = np.eye(self.d)[:,:self.k]  # 1e-3*np.random.randn(self.d, self.k)
        self.W = 1e-3*np.random.randn(self.d, self.k)

        self.X_t = TT.dmatrix('X')
        self.W_t = TT.dmatrix('W')
        self.H_t = TT.dmatrix('H')

        data_loss = TT.sum((TT.dot(self.X_t, self.W_t.T) - self.H_t)**2)
        reg = TT.sum(abs(self.X_t))
        # constraint = TT.sum((LA.diag(TT.dot(self.W_t.T, self.W_t)) - TT.ones(self.k))**2)
        loss = data_loss + self.alpha * reg
        self._loss_t = theano.function([self.H_t, self.X_t, self.W_t], loss)

        grad = theano.grad(loss, [self.X_t, self.W_t])
        self._grad_t = theano.function([self.H_t, self.X_t, self.W_t], grad)

    def loss(self):
        data_loss = np.sum((np.dot(self.X, self.W.T) - self.H)**2)
        reg = np.sum(abs(self.X))
        constraint = np.sum((np.diag(np.dot(self.W.T, self.W)) - np.ones(self.k))**2)
        return data_loss, self.alpha * reg, self.const * constraint

    def loss_t(self, x):
        X = np.reshape(x[:self.d*self.k], (self.d, self.k))
        W = np.reshape(x[self.d*self.k:], (self.d, self.k))
        return self._loss_t(self.H, X, W)

    def grad_t(self, x):
        X = np.reshape(x[:self.d*self.k], (self.d, self.k))
        W = np.reshape(x[self.d*self.k:], (self.d, self.k))
        return np.concatenate([a.flatten() for a in self._grad_t(self.H, X, W)])

    @property
    def flat_params(self):
        return np.concatenate([a.flatten() for a in [self.X, self.W]])

    def optimize(self, max_iter=10):
        print 'doing dictionary learning'
        # U, V = dict_learning_online(self.H, self.k, self.alpha, verbose=2, method='lars')
        U, V, _ = dict_learning(self.H, self.k, self.alpha, verbose=2, method='lars', max_iter=max_iter)
        self.X = U
        self.W = V.T

    def PQ(self):
        # H is [h,m]
        # X is [k,m]
        # Q is [m,h]
        # X.T Q = I ; Q,X are [m,k]
        # P H Q = I ; P H = X.T
        #P = np.linalg.lstsq(self.H.T, self.X)[0].T
        #P = np.linalg.lstsq(np.dot(self.H, Q).T, np.eye(self.k))[0].T
        #Q = np.linalg.lstsq(self.X.T, np.eye(self.k))[0]
        Q = self.X.T
        print Q.shape
        print self.H.shape
        P = np.linalg.lstsq(np.dot(Q.T, self.H.T), np.eye(self.k))[0].T
        return P, Q

# separate plotting/analysis from learning, save learned data
# make H rectangular
# smoothness constraint (wrt the state space)
# state value function
# plot different data value function formulations
# try gaussian blur for scatter plots
# mapping from S to X


def main(d = 100, # dim of tile features
         n = 5000, # number of total samples
         h = 2000,
         k = 64,
         pl = 64,
         gam = 1-1e-1,
         alpha = 2e0,
         max_iter = 10,
         seed=None):
    
    world = rsis.TorusWorld()
    #world = rsis.CircleWorld()
    fmap = rsis.TileFeatureMap(d)

    Pos, R = world.get_samples(n+1, seed=seed)
    Pos = Pos[:-1]
    R = R - np.mean(R)
    R = R / np.std(R)
    # S = fmap.transform(Pos)

    m = len(R) - h - 1
    H = np.reshape([R[i+j] for i,j in it.product(xrange(h), xrange(m))], (h,m))
    H_ = np.reshape([R[i+j+1] for i,j in it.product(xrange(h), xrange(m))], (h,m))

    model = SparseHankelDecomposition(H, k, alpha)
    model.optimize(max_iter) # dictionary learning
    P, Q = model.PQ()

    HP = np.dot(H, P)
    print 'P.T H Q - I error: ', np.linalg.norm(np.dot(np.dot(P.T, H), Q) - np.eye(k))
    print 'HP - X norm: ', np.linalg.norm(HP - model.X)
    print 'avg abs value of X', np.mean(abs(model.X))
    print 'avg abs value of HP', np.mean(abs(HP))
    print 'sparsity of X', len(np.nonzero(model.X)[0]) / np.prod(model.X.shape, dtype='float')
    print 'sparsity of HP', len(np.nonzero(HP)[0]) / np.prod(HP.shape, dtype='float')

    r0 = H[:,0]
    A = np.dot(np.dot(P.T, H_), Q)
    # b = np.linalg.lstsq(np.dot(H, P), r0)[0]
    # b = np.linalg.lstsq(shd.X, r0)[0]
    b = np.dot(Q.T, r0)
    v = np.linalg.solve((np.eye(k) - gam * A), b)  # solve for the value function

    out = {'H': H, 'P': P, 'Q': Q, 'X': model.X, 'A': A, 'b': n, 'v': v}
    with file('out.pickle', 'wb') as pic_file:
        pickle.dump(out, pic_file)

    # test_dynamics(H, R, P, A, b)

    plot_image(Pos[:n/2], np.dot(Q, b), 'plots/reward_scatter.png')
    plot_image(Pos[:n/2], np.dot(Q, v), 'plots/value_scatter.png')


    for i in xrange(k//pl):

        plot_image(Pos[:n/2], model.X[:,i*pl:(i+1)*pl], 'plots/Xbasis_scatter%i.png' % i)
        plot_image(Pos[:n/2], Q[:, i*pl:(i+1)*pl], 'plots/Qbasis_scatter%i.png' % i)
        plot_image(Pos[:n/2], P[:, i*pl:(i+1)*pl], 'plots/Pbasis_scatter%i.png' % i)

    # plot_image(Pos[:n/2], shd.W, 'plots/Wbasis_scatter.png')
    # plot_image(Pos[:n/2], np.dot(H,P), 'plots/HPbasis_scatter.png')
    #plot_interpolated_image(Pos[:n/2], X, 'plots/basis_interp.png')


    os.system('open plots/*.png')

def plot_singvals(s):
    plt.clf()
    plt.bar(range(len(s)), s)
    plt.savefig('plots/singularvals.png')

def test_dynamics(H, R, P, A, b):

    m = 500
    dyn_err = np.zeros(m)  # dynamics error in reward space
    sta_err = np.zeros(m)  # dynamics error in state space
    rew_err = np.zeros(m)  # reward prediction error

    x0 = np.dot(P.T, H[:,0])

    T = np.eye(A.shape[0])
    x = x0
    for i in xrange(m):

        x = np.dot(T, x)
        T = np.dot(A, T)
        r = H[:,i]

        rew_err[i] = np.linalg.norm(np.dot(b, x) - R[i])
        dyn_err[i] = np.linalg.norm(np.dot(P, x) - r)
        sta_err[i] = np.linalg.norm(x - np.dot(P.T, r))

    'average reward error: ', np.mean(rew_err)
    'average reward transition error: ', np.mean(dyn_err)
    'average state transition error: ', np.mean(sta_err)

    plt.clf()
    plt.semilogy(rew_err, label='reward error')
    plt.semilogy(sta_err, label='latent space transition error')
    plt.semilogy(dyn_err, label='reward space transition error')
    plt.legend()
    plt.savefig('plots/errors.png')




def plot_image(P, I, file_name = 'plots/img.png', d=50, npad = 2):

    plt.clf()

    print 'plotting columns of matrix, shape: ', I.shape

    if (I.ndim == 1):
        I = I[:, None]
        f = 1
    else:
        f = int(np.sqrt(I.shape[1]))

    f = int(np.sqrt(I.shape[1]))

    for i in xrange(f**2):

        ax = plt.subplot(f,f,i+1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.hexbin(P[:,0], P[:,1], C=I[:,i], gridsize=12, cmap='gist_earth')
        #ax.scatter(P[:,0], P[:,1], c=I[:,i], alpha=0.2, linewidths=0, cmap='gist_earth', s=2)
        #ax.scatter(P[:,0], P[:,1], c=I[:,i], alpha=0.2, linewidths=0, cmap='gist_earth', s=2)

        #xi, yi = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, d))
        #rbf = Rbf(P[:,0], P[:,1], I[:,i], smooth=1e-6, function='gaussian', epsilon=2e-1)
        #zi = rbf(yi.flatten(), xi.flatten())
        #ax.imshow(np.reshape(zi, (d, d))[npad:-npad,npad:-npad])

    print 'saving'

    plt.savefig(file_name)


def plot_interpolated_image(P, I, file_name='plots/img.png', pregrid=False, dim=50):

    plt.clf()
    
    print 'plotting columns of matrix, shape: ', I.shape
    if (I.ndim == 1):
        I = I[:, None]
        f = 1
    else:
        f = int(np.sqrt(I.shape[1]))

    for i in xrange(f**2):

        ax = plt.subplot(f,f,i+1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)


        xlim = ylim = 0.9
        xi = np.linspace(-xlim, xlim, dim)
        yi = np.linspace(-ylim, ylim, dim)
        XI, YI = np.meshgrid(xi, yi)

        # grid the data.
        # contour the gridded data, plotting dots at the randomly spaced data points.

        if pregrid:
            # grid then spline smooth
            ZI = griddata((P[:, 0], P[:, 1]), I[:, i],
                          (xi[None, :], yi[:, None]),
                          method='linear',
                          fill_value=np.mean(I[:, i]))

            spline = SmoothBivariateSpline(XI.flatten(), YI.flatten(), ZI.flatten())

        else:
            spline = SmoothBivariateSpline(P[:, 0], P[:, 1], I[:, i])


        zi = spline.ev(XI.flatten(), YI.flatten())
        zi = np.reshape(zi, (dim,dim))

        # rbf = Rbf(XI, YI, zi, epsilon=1e-1)
        # zz = rbf(XI, YI)

        plt.imshow(zi, interpolation='nearest')

        #plt.contour(xi,yi,zi,10, linewidths=0.5,colors='k')
        #plt.contourf(xi,yi,zi,10, cmap=plt.cm.spectral)
        #ax.scatter(P[:,0], P[:,1], c='k', s=2, linewidths=0)
        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)

    plt.savefig(file_name)

class RICA():

    def __init__(self,k, H, lam):

        self.d = H.shape[0]
        self.k = k
        self.H = H
        self.W = np.random.randn(self.d, self.k)
        self.lam = lam


        self.W_t = TT.dmatrix('W')
        self.H_t = TT.dmatrix('H')

        loss_t = TT.sum((TT.dot(self.W_t, TT.dot(self.W_t.T, self.H_t)) - self.H_t)**2)
        reg_t = TT.sum(abs(TT.dot(self.W_t.T, self.H_t)))
        self._loss_t = theano.function([self.H_t, self.W_t], loss_t + self.lam * reg_t)
        grad_t = theano.grad(loss_t, self.W_t)
        self._grad_t = theano.function([self.H_t, self.W_t], grad_t)

    def loss_t(self, w):
        W = np.reshape(w, (self.d, self.k))
        return self._loss_t(self.H, W)

    def grad_t(self, w):
        W = np.reshape(w, (self.d, self.k))
        return self._grad_t(self.H, W).flatten()

    def loss(self, w):
        if w.ndim == 1:
            W = np.reshape(w, (self.d, self.k))
        else:
            W = w
        return np.sum((self.H - np.dot(np.dot(W, W.T), self.H))**2) + self.regularization(W)

    def regularization(self, W):
        return self.lam * np.sum(abs(np.dot(W.T, self.H)))

    def grad(self, w):
        if w.ndim == 1:
            W = np.reshape(w, (self.d, self.k))
        else:
            W = w
        M = np.dot(W, W.T) - np.eye(self.d)
        return 4*np.dot(np.dot(np.dot(self.H, M), self.H), W).flatten() + self.lam * abs(np.dot(self.H, self.W)).flatten()

    def optimizeW(self):
        w = fmin_l_bfgs_b(self.loss_t,
                          self.W.flatten(),
                          self.grad_t)[0]

        self.W = np.reshape(w, (self.d, self.k))

    def PQ(self):
        Q = np.linalg.lstsq(np.dot(self.W.T, self.H), np.eye(self.k))[0]

        return self.W, Q



if __name__ == "__main__":
    main()
