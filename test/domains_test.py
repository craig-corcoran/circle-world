import matplotlib.pyplot as plt

import rsis


def plot_samples(sub, world, n=100):
    states, rewards = world.get_samples(n)
    ax = plt.subplot(sub, aspect='equal')
    ax.plot(states[:, 0], states[:, 1], '-', alpha=0.7, color='#bbddff')
    ax.scatter(states[:, 0], states[:, 1], lw=0, alpha=0.7, s=30, c=rewards)
    ax.grid(True)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('outward', 6))
    ax.spines['left'].set_position(('outward', 6))


if __name__ == '__main__':
    plot_samples(121, rsis.CircleWorld())
    plot_samples(122, rsis.TorusWorld(eps_z=0), 300)
    plt.show()
