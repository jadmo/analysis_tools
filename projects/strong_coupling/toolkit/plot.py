# coding: utf-8

import sys
import matplotlib.pyplot as plt
import numpy as np

from .operation import *
from .data import *

sys.path.append('/content/drive/My Drive/Research/qutip_sim/site-packages')
sys.path.append('/content/drive/My Drive/Research/qutip_sim/site-packages/qutip')
sys.path.append('/content/drive/My Drive/Research/qutip_sim/custom_files/VQE_files')
from qutip import *
from H2Functions import read_in_data
from google.colab import files

def plot_4_4(data_x, data_y, fit_y, label, x_label=None, y_label=None, save=False):
    """
    Plot grids of graphs in four-by-four.
    Mainly used for 2-qubit state tomography results.
    data_x should be 1d-array for x-axis
    The shape of data_y should be (4, 4, d1, d2) where,
    d1: corresponds to x-axis
    d2: different colours in one subplot
    label: titles for each subplots
    """

    data_y = transform_to_4_4(data_y)
    data_y = add_dim(data_y, 4)

    fig, axes = plt.subplots(nrows=4, ncols=4, sharex = True, sharey = True, figsize=(12.8, 9.6))

    plt.subplots_adjust(hspace=0.1)

    for i in range(4):
        for j in range(4):
            for k in range(np.shape(data_y)[-1]):
                # axes[i, j].set_title(r'$\langle{%s}\rangle$' % title[i * 4 + j])
                axes[i, j].scatter(data_x, data_y[i,j,:,k], s=20, marker=".", label=label[k])
                axes[3, j].set_xlabel(x_label, fontsize=18)
                axes[i, j].set_ylabel(r'$\langle{%s}\rangle$' % y_label[i * 4 + j], fontsize=18)
                # axes[i, j].yaxis.label.set_size(16)
                if fit_y is not None:
                    func = fit_y[i][j][k]['func']
                    # ub, lb = error_bar(fit_y[i][j][k]['popt'], fit_y[i][j][k]['pcov'])
                    # axes[i, j].fill_between(data_x, func(data_x, *ub), func(data_x, *lb), color='b', alpha=0.2)
                    # axes[i, j].plot(data_x, func(data_x, *fit_y[i][j][k]['popt']))
                axes[i, j].set_ylim(-1.2, 1.2)

    plt.legend()
    if save is not False:
        plt.savefig('{}.svg'.format(save))
        files.download('{}.svg'.format(save))
    plt.show()

def plot_1(data_x, data_y, fit_y=None, x_lim=None, y_lim=None, label=None, x_label=None, y_label=None, save=False):
    """
    Plot different kind of points in a plot.
    data_x should be 1d-array for x-axis
    The shape of data_y should be (1, d1, d2) where,
    d1: corresponds to x-axis
    d2: different kind of points in a plot
    label: titles for each subplots
    """

    fig = plt.figure(figsize=(6.4, 4.8))

    for k in range(np.shape(data_y)[-1]):
        # plt.title(label)
        plt.scatter(data_x, data_y[:,k], s=20, marker=".", label=label[k])

        if fit_y is not None:
            func = fit_y[k]['func']
            ub, lb = error_bar(fit_y[k]['popt'], fit_y[k]['pcov'])
            if x_lim is not None:
                fit_x = np.linspace(x_lim[0], x_lim[1], 101)
            else:
                fit_x = np.linspace(data_x[0], data_x[-1], 101)

            plt.fill_between(fit_x, func(fit_x, *ub), func(fit_x, *lb), color='b', alpha=0.2)
            plt.plot(fit_x, func(fit_x, *fit_y[k]['popt']))
            #plt.plot(data_x, data_y)

        if y_lim is None:
            plt.ylim(min(data_y[:][k]), max(data_y[:][k]))
        else:
            plt.ylim(y_lim[0], y_lim[1])

        if x_lim is None:
            plt.xlim(min(data_x), max(data_x))
        else:
            plt.xlim(x_lim[0], x_lim[1])

    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.legend()
    if save is not False:
        plt.savefig('{}.svg'.format(save))
        files.download('{}.svg'.format(save))
    plt.show()

def plot_energy_curve(theta, paulis, label=None, save=False):

    # dataset = add_dim(extrapolate(dataset), 3)
    # dataset = add_dim(np.average(dataset, axis=2), 3)
    r_values, g0, g1, g2, g3, g4, g5, p = read_in_data()

    s = np.shape(paulis)

    for k in range(s[2]):
        for j in range(k+1):
            y = []
            for i in range(len(g0)):
                E = g0[i] + g1[i] * paulis[0,:,j] + g2[i] * paulis[1,:,j] + g3[i] * paulis[2,:,j] + g4[i] * paulis[3,:,j] + g5[i] * paulis[4,:,j]

                mindex = np.argmin(E)
                y.append(E[mindex])
            if label is not None:
                plt.scatter(r_values, y, s=2, label=label[j])
            else:
                plt.scatter(r_values, y)
        plt.plot(r_values, p)
        plt.legend()
        plt.show()

        # for i in range(5):
        #     plt.plot(theta, paulis[i,:,k])
        #
        # plt.show()

def plot_sphere(data_x, data_y, fit_y, label, save=False):
    """
    Plot rows of graphs on a sphere.
    Mainly used for 1-qubit state tomography results.
    data_x should be 1d-array for x-axis
    The shape of data_y should be (3, d1, d2) where,
    d1: corresponds to x
    d2: different colours in one plot
    label: titles for each subplots
    """

    # fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12.8, 9.6))
    # plt.subplots_adjust(hspace=0.3)
    b = Bloch()
    b.point_color = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    b.sphere_color = 'w'
    b.sphere_alpha = 0
    for k in range(np.shape(data_y)[-1]):
        pnts = []
        lines = []
        fits = []
        for i in range(3):
            # axes[i,0].set_title(r'$\langle{%s}\rangle$' % label[i])
            pnts.append(data_y[i,:,k])
            lines.append(data_y[i,:,k])


            if fit_y is not None:
                func = fit_y[i][k]['func']
                fits.append(func(data_x, *fit_y[i][k]['popt']))

        b.add_points(pnts)# , meth='l')
        b.add_points(lines , meth='l')
        if fit_y is not None:
            b.add_points(fits, meth='l')
    b.show()

    if save is not False:
        plt.savefig('{}.svg'.format(save))
        files.download('{}.svg'.format(save))
    plt.show()

def Plot_3D_Rabi(data, t, fit, save=False):
    # print(np.array(data).shape, t.shape)

    params = fit

    fig = plt.figure()
    xax = fig.add_subplot(6,1,1)
    yax = fig.add_subplot(6,1,2)
    zax = fig.add_subplot(6,1,3)

    xax.scatter(t, data[0])
    yax.scatter(t, data[1])
    zax.scatter(t, data[2])

    t_fit = np.linspace(t[0],t[-1],10*len(t))

    ifit = params['IAmp']*np.sin(2.0*np.pi*params['Frequency']*(t_fit-t_fit[0])+params['IPhase'])
    qfit = params['QAmp']*np.sin(2.0*np.pi*params['Frequency']*(t_fit-t_fit[0])+params['QPhase'])

    xfit,yfit,zfit = [params['IAxis'][n]*ifit + params['QAxis'][n]*qfit + params['Centre']*params['RotationAxis'][n] for n in (0,1,2)]
    xax.plot(t_fit,xfit)
    yax.plot(t_fit,yfit)
    zax.plot(t_fit,zfit)

    plt.show()

def plot_3_1(data_x, data_y, fit_y, y_lim, label, x_label=None, y_label=None, save=False):
    """
    Plot rows of graphs in three-by-one.
    Mainly used for 1-qubit state tomography results.
    data_x should be 1d-array for x-axis
    The shape of data_y should be (3, d1, d2) where,
    d1: corresponds to x
    d2: different colours in one plot
    label: titles for each subplots
    """

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex = 'col', sharey = 'row', figsize=(8, 4))
    plt.subplots_adjust(hspace=0.3)

    for i in range(3):
        for k in range(np.shape(data_y)[-1]):
            # axes[i,0].set_title(r'$\langle{%s}\rangle$' % label[i])
            axes[i].scatter(data_x, data_y[i,:,k], s=10, marker=".", label=label[k])

            if fit_y is not None:
                func = fit_y[i][k]['func']
                # ub, lb = error_bar(fit_y[i, k]['popt'], fit_y[i, k]['pcov'])
                # plt.fill_between(data_x, func(data_x, *ub), func(data_x, *lb), color='b', alpha=0.2)
                freq = 'Frequency: {}'.format(round(fit_y[i][k]['Frequency'], 3))
                axes[i].plot(data_x, func(data_x, *fit_y[i][k]['popt']), label=freq)
                axes[i].set(xlabel = x_label)
                axes[i].set(ylabel = y_label)
                axes[i].legend()
                #plt.plot(data_x, data_y)

            if y_lim is None:
                axes[i].set_ylim(min(data_y[i,:,k]), max(data_y[i,:,k]))
            else:
                axes[i].set_ylim(y_lim[0], y_lim[1])

    if save is not False:
        plt.savefig('{}.svg'.format(save))
        files.download('{}.svg'.format(save))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

