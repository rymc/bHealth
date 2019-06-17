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


def polar_labels_figure(labels, label_names, xticklabels, empty_rows=0,
                        leading_labels=0, spiral=False,
                        title=None, m=None, fig=None, ax=None, figsize=None):
    """
    Returns polar plot with categorical bins from a matrix.

    Parameters
    ----------
    labels : (R, C) ndarray of integers
        Matrix of integers from [-1, K] denoting different labels and with the
        special value of -1 denoting no label. The value is used as an index
        for the list label_names.

    label_names : (K, ) array_like of strings
        List of strings representing each of the K labels. (eg. bedroom,
        living room, ..., kitchen)

    xticklabels : (D, ) array_like of strings
        List of strings to print around the circle with equal spacing in
        between and with the first element corresponding to the 90 degree and
        the following in a clockwise order. (eg. Monday, Tuesday, ..., Sunday)

    empty_rows : integer, optional (default 0)
        Number of empty rows to insert at the beginning of the labels matrix.
        This can be used to reduce or increase the empty space at the centre of
        the circle.

    leading_labels : integer, optional (default 0)
        Number of empty labels to insert at the beginning of the first row in
        order to start the first label in a different position than 90 degrees.

    spiral : boolean, optional (default False)
        If True, the labels are arranged in a spiral in which a row starts at
        the same level than the end bin of the previous row.
        If False, each row is in its own concentric circle, the previous one
        always smaller than the following one.

    title : string, optional (default None)
        Title for the figure.

    m : matplotlib colormap, optional (default None)
        Colormap that is used for each of the K labels.
        If None:
            If K < 11:
                m = cm.get_cmap('tab10')
            Else if K < 21:
                m = cm.get_cmap('tab20')
            Else:
                m = cm.gist_rainbow (Normalised with maximum colour value at K)

    fig : matplotlib.figure.Figure, optional
        Matplotlib figure where to create the axes for the plot, if None a new
        figure is created.

    ax : matplotlib.axes.Axes, optional (default None)
        Maptlotlib Axes where to create the plot in polar form. If None a new
        axes is created.

    figsize : (float, float), optional (default None)
        width, height in inches. If not provided default from matplotlib.

    Returns
    -------
    fig : matplotlib figure

    ax  : matplotlib axis

    """
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize)
        ax = fig.add_axes([0, 0, 1.0, 0.9], polar=True)

    if labels is None:
        labels = numpy.arrange(numpy.unique(y_array))

    n_rows = labels.shape[0] + empty_rows
    n_columns = labels.shape[1]
    labels = labels.flatten()
    ax.set_title(title)

    if m is None:
        if len(label_names) < 11:
            m = cm.get_cmap('tab10')
        elif len(label_names) < 21:
            m = cm.get_cmap('tab20')
        else:
            norm = mpl.colors.Normalize(vmin=0, vmax=len(label_names))
            cmap = cm.gist_rainbow
            colors = cm.ScalarMappable(norm=norm, cmap=cmap)
            m = colors.to_rgba

    width = 2 * numpy.pi / n_columns # All boxes are the same width
    indices = numpy.arange(len(labels)) + leading_labels + empty_rows*n_columns
    x = indices * 2 * numpy.pi / n_columns
    bottom = indices / n_columns
    if not spiral:
        bottom = bottom.astype(int)
    colors = [m(y) if y!= -1 else 'white' for y in labels]
    ax.bar(x, height=1, width=width, bottom=bottom, align='edge', color=colors)
    #for i, y in enumerate(labels):
    #    x = i * 2 * numpy.pi / n_columns
    #    bottom = i / n_columns
    #    if not spiral:
    #        bottom = int(bottom)

    #    ax.bar(x, height=1, width=width, bottom=bottom, color=m(y))
    if spiral:
        plt.ylim(0,n_rows+1)
    else:
        plt.ylim(0,n_rows)
    ax.set_yticks([])

    ax.set_xticks(2 * numpy.pi * numpy.arange(len(xticklabels)) /
                  len(xticklabels))
    ax.set_xticklabels(xticklabels)

    ax.set_theta_direction(-1)
    ax.set_theta_offset(numpy.pi/2.0)
    handles = [mpatches.Patch(color=m(i), label=y) for i, y in
               enumerate(label_names)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.2, 1.1),
              ncol=1, fontsize=7 )
    return fig, ax
