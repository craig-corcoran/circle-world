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
#import rsis.domains.TorusWorld

def svd_PQ(H, k):

    U,s,V = svds(H, k=k)
    U=U[:,::-1]
    s=s[::-1]
    V=V[::-1,:].T

    P = U*np.sqrt(1/s)
    Q = V*np.sqrt(1/s)

    return P, Q

class SparseHankelDecomposition(object):

    def __init__(self, H, k, lam, const):

        self.H = H
        self.d = H.shape[0]
        self.k = k
        self.lam = lam
        self.const = const

        self.X = np.eye(self.d)[:,:self.k]  # 1e-3*np.random.randn(self.d, self.k)
        self.W = 1e-3*np.random.randn(self.d, self.k)

        self.X_t = TT.dmatrix('X')
        self.W_t = TT.dmatrix('W')
        self.H_t = TT.dmatrix('H')

        data_loss = TT.sum((TT.dot(self.X_t, self.W_t.T) - self.H_t)**2)
        reg = TT.sum(abs(self.X_t))
        constraint = TT.sum((LA.diag(TT.dot(self.W_t.T, self.W_t)) - TT.ones(self.k))**2)
        loss = data_loss + self.lam * reg + self.const * constraint
        self._loss_t = theano.function([self.H_t, self.X_t, self.W_t], loss)

        grad = theano.grad(loss, [self.X_t, self.W_t])
        self._grad_t = theano.function([self.H_t, self.X_t, self.W_t], grad)

    def loss(self):
        data_loss = np.sum((np.dot(self.X, self.W.T) - self.H)**2)
        reg = np.sum(abs(self.X))
        constraint = np.sum((np.diag(np.dot(self.W.T, self.W)) - np.ones(self.k))**2)
        return data_loss, self.lam * reg, self.const * constraint

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

    def optimize(self):
        x = fmin_l_bfgs_b(self.loss_t,
                          self.flat_params,
                          self.grad_t)[0]

        self.X = np.reshape(x[:self.d*self.k], (self.d, self.k))
        self.W = np.reshape(x[self.d*self.k:], (self.d, self.k))

    def PQ(self):
        Q = np.linalg.lstsq(self.X.T, np.eye(self.k))[0]
        P = np.linalg.lstsq(self.H, self.X)[0]

        return P, Q


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

def main(d = 100, # dim of tile features
         n = 5000, # number of total samples
         k = 64,
         gam = 1-1e-2,
         lam = 1e4,
         const = 1e8,
         seed=None):
    
    world = rsis.TorusWorld()
    #world = rsis.CircleWorld()
    fmap = rsis.TileFeatureMap(d)

    Pos, R = world.get_samples(n+1, seed=seed)
    Pos = Pos[:-1]
    F = fmap.transform(Pos)
    
    h = len(R)//2

    H = np.reshape([R[i+j] for i,j in it.product(xrange(h), xrange(h))], (h,h))
    H_ = np.reshape([R[i+j+1] for i,j in it.product(xrange(h), xrange(h))], (h,h))

    #P, Q = svd_PQ(H, k)

    def log(model):
        loss, reg, constraint = model.loss()
        print 'total loss: ', loss
        print 'constriant loss: ', constraint
        print 'regularization: ', reg
        print 'reg ratio: ', (loss-constraint-reg) / reg
        print 'constraint ratio: ', (loss-constraint-reg) / constraint

    shd = SparseHankelDecomposition(H,k,lam,const)
    log(shd)
    shd.optimize()
    log(shd)

    P, Q = shd.PQ()

    # rica = RICA(k,H,lam)
    #
    # loss = rica.loss(rica.W)
    # reg = rica.regularization(rica.W)
    # print 'before: '
    # print 'total loss: ', loss
    # print 'regularization: ', reg
    # print 'ratio: ', (loss-reg) / reg

    # rica.optimizeW()
    # P, Q = rica.PQ()
    #
    # loss = rica.loss(rica.W)
    # reg = rica.regularization(rica.W)
    # print 'after:'
    # print 'total loss: ', loss
    # print 'regularization: ', reg
    # print 'ratio after: ', (loss-reg) / reg


    print 'P.T H Q - I error: ', np.linalg.norm(np.dot(np.dot(P.T,H),Q) - np.eye(k))

    X = np.dot(H,P)

    Y = np.dot(H,shd.X)
    print 'HP - X norm: ', np.linalg.norm(X - shd.X)

    r0 = H[:,0]
    A = np.dot(np.dot(P.T, H_), Q)
    b = np.linalg.lstsq(shd.X, r0)[0]

    # print 'nuclear norm: ', sum(s)
    # print 'fraction of nuclear norm in first %i singular values: ' % k, sum(s[:k]) / sum(s)
    #
    # print 'V column orthogonal error: ', np.linalg.norm(np.eye(k) - np.dot(V.T, V))
    # print 'U column orthogonal error: ', np.linalg.norm(np.eye(k) - np.dot(U.T,U))
    #
    # print '||P.T H Q - I||', np.linalg.norm(np.eye(k) - np.dot(np.dot(P.T,H),Q))
    # print 'matrix factorization error: ||USV.T-H||', np.linalg.norm(np.dot(U * s, V.T) - H)
    # print 'norm of H:', np.linalg.norm(H)
    # print 'fraction of variance captured in SVD: ',

    # solve for the value function
    v = np.linalg.solve((np.eye(k) - gam * A), b)

    # plt.clf()
    # plt.bar(range(len(s)), s)
    # plt.savefig('plots/singularvals.png')

    # H = USV -> (U.T H).T , H V.T

    # test_dynamics(H, R, P, A, b)

    # plot_interpolated_image(Pos[:n/2], np.dot(Q, b), 'plots/reward_interp.png')
    plot_image(Pos[:n/2], np.dot(Q, b), 'plots/reward_scatter.png')

    # plot_interpolated_image(Pos[:n/2], np.dot(Q, v), 'plots/value_interp.png')
    plot_image(Pos[:n/2], np.dot(Q, v), 'plots/value_scatter.png')

    #plot_interpolated_image(Pos[:n/2], X, 'plots/basis_interp.png')
    plot_image(Pos[:n/2], Y, 'plots/Ybasis_scatter.png')
    plot_image(Pos[:n/2], shd.X, 'plots/Xbasis_scatter.png')
    plot_image(Pos[:n/2], Q, 'plots/Qbasis_scatter.png')
    plot_image(Pos[:n/2], P, 'plots/Pbasis_scatter.png')
    plot_image(Pos[:n/2], shd.W, 'plots/Wbasis_scatter.png')

    os.system('open plots/*.png')


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




def plot_image(P, I, file_name = 'plots/img.png'):

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
        ax.hexbin(P[:,0], P[:,1], C=I[:,i], gridsize=15, cmap='gist_earth')
        ax.scatter(P[:,0], P[:,1], c=I[:,i], alpha=0.2, linewidths=0, cmap='gist_earth', s=1)

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


if __name__ == "__main__":
    main()
