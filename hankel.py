import os
import numpy as np
import rsis
import rsis.experiments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools as it
from scipy.sparse.linalg import svds
from scipy.interpolate import griddata, Rbf, SmoothBivariateSpline
#import rsis.domains.TorusWorld

def main(d = 100, # dim of tile features
         n = 4000, # number of total samples
         k = 16,
         gam = 1-2e-1,
         seed=None):
    
    world = rsis.TorusWorld()
    #world = rsis.CircleWorld()
    fmap = rsis.TileFeatureMap(d)

    Pos, R = world.get_samples(n+1, seed=seed)
    Pos = Pos[:-1]
    X = fmap.transform(Pos)
    
    h = len(R)//2

    H = np.reshape([R[i+j] for i,j in it.product(xrange(h), xrange(h))], (h,h))
    H_ = np.reshape([R[i+j+1] for i,j in it.product(xrange(h), xrange(h))], (h,h))
    
    U,s,V = svds(H, k=k)
    U=U[:,::-1] 
    s=s[::-1]
    V=V[::-1,:].T

    P = U*np.sqrt(1/s)
    Q = V*np.sqrt(1/s)
    X = np.dot(H,P)

    r0 = H[:,0]
    A = np.dot(np.dot(P.T, H_), Q)
    b = np.dot(Q.T, r0)

    print 'nuclear norm: ', sum(s)
    print 'fraction of nuclear norm in first %i singular values: ' % k, sum(s[:k]) / sum(s)

    print 'V column orthogonal error: ', np.linalg.norm(np.eye(k) - np.dot(V.T, V))
    print 'U column orthogonal error: ', np.linalg.norm(np.eye(k) - np.dot(U.T,U))

    print '||P.T H Q - I||', np.linalg.norm(np.eye(k) - np.dot(np.dot(P.T,H),Q))
    print 'matrix factorization error: ||USV.T-H||', np.linalg.norm(np.dot(U * s, V.T) - H)
    print 'norm of H:', np.linalg.norm(H)

    # solve for the value function
    v = np.linalg.solve((np.eye(k) - gam * A), b)

    print v.shape
    print P.shape
    print Pos[:n/2].shape

    #plt.clf()
    #plt.bar(range(len(s)), s)
    #plt.savefig('plots/singularvals.png')

    # H = USV -> (U.T H).T , H V.T

    # test_dynamics(H, R, P, A, b)

    plot_interpolated_image(Pos[:n/2], np.dot(P, b), 'plots/reward_interp.png')
    plot_interpolated_image(Pos[:n/2], np.dot(Q, b), 'plots/rewardQ_interp.png')

    plot_image(Pos[:n/2], np.dot(P, b), 'plots/reward_scatter.png')
    plot_image(Pos[:n/2], np.dot(Q, b), 'plots/rewardQ_scatter.png')

    plot_interpolated_image(Pos[:n/2], np.dot(P, v), 'plots/value_interp.png')
    plot_interpolated_image(Pos[:n/2], np.dot(Q, v), 'plots/valueQ_interp.png')

    plot_image(Pos[:n/2], np.dot(P, v), 'plots/value_scatter.png')
    plot_image(Pos[:n/2], np.dot(Q, v), 'plots/valueQ_scatter.png')

    plot_interpolated_image(Pos[:n/2], X, 'plots/basis_interpolated.png')
    plot_interpolated_image(Pos[:n/2], X, 'plots/basis_interpolated_pregrid.png', pregrid=True)
    #plot_image(Pos[:n/2], X, 'plots/Pbasis.png')
    #plot_image(Pos[:n/2], np.dot(H, Q), 'plots/Qbasis.png')

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
        #ax.hexbin(P[:,0], P[:,1], C=I[:,i], gridsize=10)
        ax.scatter(P[:,0], P[:,1], c=I[:,i], alpha=0.5, linewidths=0, cmap=plt.cm.spring)

    print 'saving'

    plt.savefig(file_name)


def plot_interpolated_image(P, I, file_name = 'plots/img.png', pregrid = False, dim=50):

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

        print 'gridding'

        if pregrid:
            # grid then spline smooth
            ZI = griddata((P[:, 0], P[:, 1]), I[:, i],
                          (xi[None, :], yi[:, None]),
                          method='linear',
                          fill_value=np.mean(I[:, i]))

            spline = SmoothBivariateSpline(XI.flatten(), YI.flatten(), ZI.flatten())

        else:
            spline = SmoothBivariateSpline(P[:, 0], P[:, 1], I[:, i])


        print 'smoothing'
        zi = spline.ev(XI.flatten(), YI.flatten())
        zi = np.reshape(zi, (dim,dim))

        # rbf = Rbf(XI, YI, zi, epsilon=1e-1)
        # zz = rbf(XI, YI)

        print 'plotting'
        plt.imshow(zi, interpolation='nearest')

        #plt.contour(xi,yi,zi,10, linewidths=0.5,colors='k')
        #plt.contourf(xi,yi,zi,10, cmap=plt.cm.spectral)
        #ax.scatter(P[:,0], P[:,1], c='k', s=2, linewidths=0)
        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)

    print 'saving'
    
    plt.savefig(file_name)


if __name__ == "__main__":
    main()
