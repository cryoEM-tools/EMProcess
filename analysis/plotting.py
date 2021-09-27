import numpy as np
import matplotlib.pylab as plt

def plot_fig(
        xs, ys, ax=None, label=None, color='black', bar=False, norm_ys=True, alpha=1.0, **kwargs):
    """Wrapper for matplotlib plotting

    Input
    ----------
    xs : array-like, shape=(n_points, )
        List of x values to plot.
    ys : array-like, shape=(n_points, )
        List of y values to plot.
    ax : matplotlib.pylab.figure

    Returns
    ----------
    pfig : 
    ax :
    """
    if norm_ys:
        ys = int_norm(xs, ys)
    if ax is None:
        pfig = plt.figure(figsize=(12,5))
        ax = plt.subplot(
            111, **kwargs) #ylabel='probability', xlabel='FRET E', title='Res 182-241')
        for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
        ax.tick_params(direction='out', length=10, width=3, colors='black')
    else:
        pfig = ax.figure
    if bar:
        width = (xs[1] - xs[0])
    if label is None:
        if bar:
            plt.bar(xs, ys, color=color, width=width, edgecolor='black', alpha=alpha)
        else:
            plt.plot(xs, ys, color=color, linewidth=3)
    else:
        if bar:
            plt.bar(
                xs, ys, color=color, label=label, width=width,
                edgecolor='black', alpha=alpha)
        else:
            plt.plot(
                xs, ys, label=label,
                color=color, linewidth=3)
        plt.legend(loc=2, prop={'size': 18})
    return pfig, ax
