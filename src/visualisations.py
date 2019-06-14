import numpy

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['savefig.format'] = 'svg'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import datetime

import pandas as pd


def features_figure(X, ts=None, feature_names=None, fig=None, ax=None,
                    figsize=None):
    """
    Returns figure with lines for every feature in the y axis and time in the x axis.

    Parameters
    ----------
    X : (N, D) ndarray
        Matrix with N samples and D feature values.

    ts : (N, ) ndarray of datetime, optional
        One-dimensional array with the datetime of every sample. If None
        assigns numbers from 0 to N.

    feature_names : (D, ) array_like, optional
        List of names corresponding to each feature. It assumes that the order
        corresponds to the columns of matrix X. If None the names are integers
        from 0 to D.

    fig : matplotlib.figure.Figure, optional
        Matplotlib figure where to create the axes for the plot, if None a new
        figure is created.

    ax : matplotlib.axes.Axes, optional
        Maptlotlib Axes where to create the plot, if None a new axes is
        created.

    figsize : (float, float), optional
        width, height in inches. If not provided default from matplotlib.

    Returns
    -------
    fig : matplotlib figure

    ax  : matplotlib axis


    Examples
    --------

    >>> X = np.array([[25, 70], [26, 60], [23, 65], [25, 70], [23, 77]])
    >>> ts = np.datetime64(0, 's') + np.arange(len(X))
    >>> feature_names = ['temperature', 'humidity']
    >>> features_figure(X, ts, feature_names)
    (<Figure size 640x480 with 1 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x7f8d8086f400>)
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    if ts is None:
        ts = numpy.arange(X.shape[0])

    if feature_names is None:
        feature_names = np.arange(X.shape[1])

    for i, feature in enumerate(feature_names):
        ax.plot(ts, X[:, i], label=feature)
    plt.gcf().autofmt_xdate()
    ax.legend(loc='upper center', ncol=6, fancybox=True, shadow=False,
              fontsize=9, framealpha=0.7)
    xfmt = mpl.dates.DateFormatter('%H:%M\n%d/%m')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_ylim(X.min(), X.max() + X.std(axis=0).max())
    ax.set_xlim(ts[0], ts[-1])
    plt.gcf().autofmt_xdate()
    ax.grid(b=True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, ax


def labels_figure(y_array, ts=None, labels=None, fig=None, ax=None,
                  figsize=None):
    """
    Returns figure with labels in the y axis and time in the x axis.

    All the contiguous samples with the same label are aggregated and a
    horizontal box is drawn that extends from the first until de last sample.
    This is repeated for every label and sample.

    Parameters
    ----------
    y_array : (N, ) ndarray of integers
        One-dimensional array with all the labels in numerical discrete format.

    ts : (N, ) ndarray of datetime, optional
        One-dimensional array with the datetime of every label. If None assigns
        numbers from 0 to N.

    labels : (K, ) array_like, optional
        List of names corresponding to each label. It assumes that the order
        corresponds to the values in y_array. If None assigns numbers from 0 to
        K (where K is the number of unique elements in y_array).

    fig : matplotlib.figure.Figure, optional
        Matplotlib figure where to create the axes for the plot, if None a new
        figure is created.

    ax : matplotlib.axes.Axes, optional
        Maptlotlib Axes where to create the plot, if None a new axes is
        created.

    figsize : (float, float), optional
        width, height in inches. If not provided default from matplotlib.

    Returns
    -------
    fig : matplotlib figure

    ax  : matplotlib axis


    Examples
    --------

    >>> y_array = np.array([0, 0, 0, 1, 1, 2, 2, 0])
    >>> ts = np.datetime64(0, 's') + np.arange(len(y_array))
    >>> labels = ['stand', 'walk', 'run']
    >>> labels_figure(y_array, ts, labels)
    (<Figure size 640x480 with 1 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x7f0f50361e10>)
    """
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize)
        ax = fig.add_subplot()

    if labels is None:
        labels = numpy.arrange(numpy.unique(y_array))

    if ts is None:
        ts = numpy.arange(len(y_array))

    norm = mpl.colors.Normalize(vmin=0, vmax=len(labels))
    cmap = cm.gist_rainbow
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    y_change = numpy.where(y_array != numpy.roll(y_array, 1))[0]

    # First label needs to be added manually
    if len(y_change) > 0:
        y = y_array[0]
        interval = (ts[0], ts[y_change[1]-1])
        line_xs = numpy.array(interval)
        line_ys = numpy.array((y, y))
        ax.fill_between(line_xs, line_ys-0.5, line_ys+0.5,
                        facecolor=m.to_rgba(y), edgecolor='dimgray',
                        linewidth=0.2)
    for i, change_id in enumerate(y_change):
        if i == len(y_change)-1:
            y = y_array[-1]
            interval = (ts[change_id],
                        ts[-1])
        else:
            y = y_array[change_id]
            interval = (ts[change_id],
                        ts[y_change[i+1]])
        line_xs = numpy.array(interval)
        line_ys = numpy.array((y, y))
        ax.fill_between(line_xs, line_ys-0.5, line_ys+0.5,
                        facecolor=m.to_rgba(y), edgecolor='dimgray',
                        linewidth=0.2)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(labels), rotation=45, ha='right')
    ax.set_ylim([-1, len(labels)])
    xfmt = mpl.dates.DateFormatter('%H:%M\n%d/%m')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlim((ts[0], ts[-1]))
    plt.gcf().autofmt_xdate()
    ax.grid(b=True)
    ax.set_axisbelow(True)
    return fig, ax
