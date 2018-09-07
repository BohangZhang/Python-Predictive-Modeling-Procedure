# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:16:30 2018

Script for data visualization

@author: bohzhang
"""

import numpy as np
import pandas as pd
from models import lift
from scikitplot.metrics import cumulative_gain_curve
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, plot_mpl
init_notebook_mode()


def plot_lift_stability_over_time(y_true, y_probas, period_dt, title='Lift Stability Over Time', skip_dt=2, group_by_month=False,
                                  ax=None, figsize=None, title_fontsize="large", text_fontsize="medium", plotly=False):

    """
    Generate a graph that plots cumulative lift lines over different dates.

    Parameters
    --------------------------------
    y_true : Series
        true target column in the test set.
    y_probas : numpy.ndarray (float64)
        predicted probabilities of target column.
    period_dt : Series
        period dates for each observation in the test set.
    title : str
        title of the generated plot.
    skip_dt : int
        number of dates to skip from plotting.
        skip_dt=1 -> include all the periods.
    group_by_month : bool
        plots cumulative lift lines that are grouped by month or not.
        skip_dt will be useless when set to True.
    ax : matplotlib.axes.Axes
        the axes upon which to plot the learning curve.
        the plot is drawn on a new set of axes if None.
    figsize : 2-tuple of float or int
        tuple denoting figure size of the plot.
    title_fontsize, text_fontsize : int or str
        matplotlib-style fontsizes.
        "small", "medium", "large" or an integer.
    plotly : bool
        convert the generated plot to be dynamic or not (in plotly).

    Returns
    --------------------------------
    plotly : False
        matplotlib.axes.Axes
            The axes on which the plot was drawn.

    Effects
    ---------------------------
    plotly : True
        save an html file for the dynamic plot.

    """
    y_probas = pd.Series(y_probas[:, 1], index=y_true.index)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    months = []
    dates = np.unique(period_dt)
    dates.sort()
    if not group_by_month:
        dates = dates[::skip_dt]

    for i in dates:
        if group_by_month:
            if pd.DatetimeIndex([i]).month[0] in months:
                continue
            months.append(pd.DatetimeIndex([i]).month[0])
            indexes = y_true.loc[pd.DatetimeIndex(period_dt).month == pd.DatetimeIndex([i]).month[0]].index
        else:
            indexes = y_true.loc[(period_dt == i)].index

        percentages, gains = cumulative_gain_curve(y_true[indexes], y_probas[indexes])

        y_fit = np.unique(gains)
        x_fit = np.array([(percentages[gains.tolist().index(i)] + percentages[len(gains) - 1 - gains.tolist()[::-1].index(i)]) / 2 for i in y_fit])
        x_fit[0], x_fit[-1] = 0, 1

        x_plot = np.linspace(x_fit.min(), x_fit.max(), 101)
        f = interp1d(x_fit, y_fit, kind="linear")
        y_plot = f(x_plot)

        if group_by_month:
            ax.plot(x_plot, y_plot, lw=1.5, label="{0}-{1}".format(pd.DatetimeIndex([i]).year[0], pd.DatetimeIndex([i]).month[0]))
        else:
            ax.plot(x_plot, y_plot, lw=1.5, label="{0}-{1}-{2}".format(pd.DatetimeIndex([i]).year[0], pd.DatetimeIndex([i]).month[0],
                                                                       pd.DatetimeIndex([i]).day[0]))

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.plot([0, 1], [0, 1], 'k-', lw=0.5)
    ax.set_xlabel('Percentage of Sample', fontsize=text_fontsize)
    ax.set_ylabel('Cumulative Lift', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')

    if plotly:
        plot_mpl(fig)
    else:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])
        ax.legend(loc='upper center', fontsize=text_fontsize, bbox_to_anchor=(0.5, -0.16), ncol=4)
        return ax


def plot_decile_stability_over_time(y_true, y_probas, period_dt, title='Decile Stability Over Time', skip_dt=2, group_by_month=False,
                                    ax=None, figsize=None, title_fontsize="large", text_fontsize="medium", plotly=False):

    """
    Generate a graph that plots lifts with different deciles over different dates.

    Parameters
    --------------------------------
    y_true : Series
        true target column in the test set.
    y_probas : numpy.ndarray (float64)
        predicted probabilities of target column.
    period_dt : Series
        period dates for each observation in the test set.
    title : str
        title of the generated plot.
    skip_dt : int
        number of dates to skip from plotting.
        skip_dt=1 -> include all the periods.
    group_by_month : bool
        plots lifts that are grouped by month or not.
        skip_dt will be useless when set to True.
    ax : matplotlib.axes.Axes
        the axes upon which to plot the learning curve.
        the plot is drawn on a new set of axes if None.
    figsize : 2-tuple of float or int
        tuple denoting figure size of the plot.
    title_fontsize, text_fontsize : int or str
        matplotlib-style fontsizes.
        "small", "medium", "large" or an integer.
    plotly : bool
        convert the generated plot to be dynamic or not (in plotly).

    Returns
    --------------------------------
    plotly : False
        matplotlib.axes.Axes
            The axes on which the plot was drawn.

    Effects
    ---------------------------
    plotly : True
        save an html file for the dynamic plot.

    """
    y_probas = pd.DataFrame(y_probas, index=y_true.index)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    dates = np.unique(period_dt)
    dates.sort()
    if not group_by_month:
        dates = dates[::skip_dt]

    for depth in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        lifts = []
        x_date = []
        months = []
        for i in dates:
            if group_by_month:
                if pd.DatetimeIndex([i]).month[0] in months:
                    continue
                months.append(pd.DatetimeIndex([i]).month[0])
                indexes = y_true.loc[pd.DatetimeIndex(period_dt).month == pd.DatetimeIndex([i]).month[0]].index
                x_date.append(datetime.strptime(pd.to_datetime(i, format="%Y00%m").strftime("%Y-%m"), "%Y-%m"))
            else:
                indexes = y_true.loc[(period_dt == i)].index
                x_date.append(pd.to_datetime(i))

            lifts.append(lift(y_true[indexes], y_probas.loc[indexes, :].as_matrix(), depth=depth))
        ax.plot(x_date, lifts, 'o-', lw=1, label=str(depth * 10))

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Period Date', fontsize=text_fontsize)
    ax.set_ylabel('Lift', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)

    if plotly:
        plot_mpl(fig)
    else:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])
        ax.legend(loc='upper center', fontsize=text_fontsize, bbox_to_anchor=(0.5, -0.16), ncol=5)
        return ax


def plot_target_rate_over_time(y_true, period_dt, title='Target Rate Over Time', skip_dt=2, group_by_month=False,
                               ax=None, figsize=None, title_fontsize="large", text_fontsize="medium", plotly=False):

    """
    Generate a graph that plots target rates over different dates.

    Parameters
    --------------------------------
    y_true : Series
        true target column in the test set.
    period_dt : Series
        period dates for each observation in the test set.
    title : str
        title of the generated plot.
    skip_dt : int
        number of dates to skip from plotting.
        skip_dt=1 -> include all the periods.
    group_by_month : bool
        plots target rates that are grouped by month or not.
        skip_dt will be useless when set to True.
    ax : matplotlib.axes.Axes
        the axes upon which to plot the learning curve.
        the plot is drawn on a new set of axes if None.
    figsize : 2-tuple of float or int
        tuple denoting figure size of the plot.
    title_fontsize, text_fontsize : int or str
        matplotlib-style fontsizes.
        "small", "medium", "large" or an integer.
    plotly : bool
        convert the generated plot to be dynamic or not (in plotly).

    Returns
    --------------------------------
    plotly : False
        matplotlib.axes.Axes
            The axes on which the plot was drawn.

    Effects
    ---------------------------
    plotly : True
        save an html file for the dynamic plot.

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    dates = np.unique(period_dt)
    dates.sort()
    if not group_by_month:
        dates = dates[::skip_dt]

    targets = []
    x_date = []
    months = []
    for i in dates:
        if group_by_month:
            if pd.DatetimeIndex([i]).month[0] in months:
                continue
            months.append(pd.DatetimeIndex([i]).month[0])
            indexes = y_true.loc[pd.DatetimeIndex(period_dt).month == pd.DatetimeIndex([i]).month[0]].index
            x_date.append(datetime.strptime(pd.to_datetime(i, format="%Y00%m").strftime("%Y-%m"), "%Y-%m"))
        else:
            indexes = y_true.loc[(period_dt == i)].index
            x_date.append(pd.to_datetime(i))

        targets.append(sum(y_true[indexes]) / len(y_true[indexes]))

    ax.plot(x_date, targets, 'o-', lw=2)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Period Date', fontsize=text_fontsize)
    ax.set_ylabel('Target Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')

    if plotly:
        plot_mpl(fig)
    else:
        for rate in zip(x_date, targets):
            ax.annotate('{}%'.format(np.round(rate[1] * 100, 2)), xy=rate, textcoords='data')
        return ax
