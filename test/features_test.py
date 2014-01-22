import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import numpy as np
import rsis


def plot_features(fmap, size, rows, cols):
    gs = grid.GridSpec(rows, cols)
    gs.update(wspace=0.1, hspace=0.1)

    s = size * 1j
    P = np.mgrid[-1:1:s, -1:1:s].reshape((2, size * size)).T
    X = fmap.transform(P)
    lim = abs(X).max()
    print 'encoded', P.shape, '->', 'as', X.shape

    for i in xrange(min(rows * cols, X.shape[1])):
        ax = plt.subplot(gs[i], aspect='equal')
        ax.imshow(X[:, i].reshape((size, size)),
                  cmap='gray',
                  vmin=-lim,
                  vmax=lim,
                  interpolation='nearest')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
    plt.show()


if __name__ == '__main__':
    plot_features(rsis.FourierFeatureMap(9, True), 21, 18, 9)
    plot_features(rsis.TileFeatureMap(3), 20, 7, 12)
