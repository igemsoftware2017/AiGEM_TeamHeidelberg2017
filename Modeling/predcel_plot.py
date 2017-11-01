import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from math import *
import json
import sys
import numpy as np
import pprint
import os
import pickle
import random

plt.style.use(json.load(open('style.json', 'r')))

with open('colors.json') as pickle_file:
    colors = json.load(pickle_file)

cdict = {'red': [(0.0, 0.0, 0.0),
                 (0.35, 0.2, 0.2),
                 (0.5, 0.6056862745, 0.6256862745),
                 (0.65, 0.7725490196, 0.7725490196),
                 (1.0, 0.9725490196, 1.0)],

         'green': [(0.0, 0.0, 0.3294117647),
                   (0.35, 0.2294117647, 0.2294117647),
                   (0.5, 0.0998039216, 0.1198039216),
                   (0.65, 0.5196078431, 0.5196078431),
                   (1.0, 0.6196078431, 1.0)],

         'blue': [(0.0, 0.0, 0.5764705882),
                  (0.35, 0.4764705882, 0.4764705882),
                  (0.5, 0.1154901961, 0.1354901961),
                  (0.65, 0.1137254902, 0.1137254902),
                  (1.0, 0.1137254902, 1.0)]}

cdict2 = {'red': [(0.0, 0.0, 0.0),
                 (0.05, 0.2, 0.2),
                 (0.15, 0.6056862745, 0.6256862745),
                 (0.4, 0.7725490196, 0.7725490196),
                 (1.0, 0.9725490196, 1.0)],

         'green': [(0.0, 0.0, 0.3294117647),
                   (0.05, 0.2294117647, 0.2294117647),
                   (0.15, 0.0998039216, 0.1198039216),
                   (0.4, 0.5196078431, 0.5196078431),
                   (1.0, 0.6196078431, 1.0)],

         'blue': [(0.0, 0.0, 0.5764705882),
                  (0.05, 0.4764705882, 0.4764705882),
                  (0.15, 0.1154901961, 0.1354901961),
                  (0.4, 0.1137254902, 0.1137254902),
                  (1.0, 0.1137254902, 1.0)]}


def valid_phage_titer(titer, o):
    gen = 1

    while o.min_cp < titer[(gen * (o.tsteps + 1)) - 1] < o.max_cp:
        if gen == o.epochs:
            return gen, 0
        gen += 1

    if o.min_cp > titer[(gen * (o.tsteps + 1)) - 1]:
        tendency = -1
    else:
        tendency = 1

    return gen, tendency



def evaluate_meta(data, o, plotting=False):
    reduced = {}  # get datas structure
    for key1 in list(data.keys()):
        reduced[key1] = {}
        for key2 in list(data[key1].keys()):
            reduced[key1][key2] = {}
            for key3 in list(data[key1][key2].keys()):
                if plotting:
                    plot(data[key1][key2][key3], prefix ='meta/', suffix='{}_{}_{}'.format(key1, key2, key3))
                reduced[key1][key2][key3] = {}  # overwrite data still left from data
                reduced[key1][key2][key3]['cpend'] = data[key1][key2][key3].cp[-2]
                valid_epochs, tendency = valid_phage_titer(data[key1][key2][key3].cp[:-1], o)
                reduced[key1][key2][key3]['valid_epochs'] = valid_epochs
                reduced[key1][key2][key3]['tendency'] = tendency # true, when min_cp > titer in the end
    return reduced


def coltransform(point, maxepochs):
    return point['tendency'] * (1-(point['valid_epochs']/maxepochs))


def d(array):  # discretizes values.
    for idx in range(array.shape[0]):
        if array[idx] < 1:
            array[idx] = 0
    return array


def plot(o, v, summariesdir, prefix="", suffix=""):
    # direct plot
    all_ecoli = (np.asarray(v.cep[:-1]) + np.asarray(v.cei[:-1]) + np.asarray(v.ceu[:-1])).tolist()

    draw(v.time[:-1], v.cei[:-1], v.ceu[:-1], v.cep[:-1], all_ecoli, v.cp[:-1], True, 'predcel', prefix, suffix, summariesdir)
    draw(v.time[:-1], v.cei[:-1], v.ceu[:-1], v.cep[:-1], all_ecoli, v.cp[:-1], False, 'predcel', prefix, suffix, summariesdir)


    # derivatives
    all_ecoli = (np.asarray(v.sdcep[:-1]) + np.asarray(v.sdcei[:-1]) + np.asarray(v.sdceu[:-1])).tolist()
    draw(v.time[:-1], v.sdcei[:-1], v.sdceu[:-1], v.sdcep[:-1], all_ecoli, v.sdcp[:-1], True, 'predcel_derivatives', prefix, suffix, summariesdir)
    draw(v.time[:-1], v.sdcei[:-1], v.sdceu[:-1], v.sdcep[:-1], all_ecoli, v.sdcp[:-1], False, 'predcel_derivatives', prefix, suffix, summariesdir)

    if o.fitnessmode == 'dist':
        plot_dist_f(v, o, prefix, suffix, summariesdir)
        drawdist(v.time[:-1], True, 'predcel', prefix, suffix, o, v, summariesdir)
        drawdist(v.time[:-1], False, 'predcel', prefix, suffix, o, v, summariesdir)


def draw(x, y1, y2, y3, y4, y5, log, name, prefix, suffix, summariesdir):
    plt.figure(1, dpi=300)
    plt.plot(x, y2, label='Uninfected', color=colors['mblue'])
    plt.plot(x, y1, label='Infected', color=colors['lblue'])
    plt.plot(x, y3, label='Phage-producing', color=colors['blue'])
    plt.plot(x, y4, label='All E. coli', color=colors['fblue'])
    plt.plot(x, y5, label='Phage', color=colors['red'])

    plt.legend()
    logstr = ''
    if log:
        plt.yscale('log')
        logstr = '_log'

    plt.ylabel('c in Lagoon [cfu]/[pfu]')
    plt.title('Calculation of Concentrations during PREDCEL')
    plt.xlabel('Time [min]')

    plt.savefig(os.path.join(summariesdir, '{}{}_{}.png'.format(prefix, name, logstr, suffix)))
    plt.gcf().clear()


def drawdist(x, log, name, prefix, suffix, o, v, summariesdir):
    col = 0
    plt.figure(1, dpi=300)
    ax = plt.subplot(111)
    if o.plot_dist == "True":
        for f_val in list(v.dist_f[0].keys()):
            cepf = np.asarray([v.dist_f[t][f_val] for t in range(len(v.dist_f) - 1)]) \
                * np.asarray(v.cep[:-1])
            try:
                p1 = p1 + ax.plot(x, d(cepf), label='cep(f={})'.format(f_val), color=list(colors.values())[col])
            except:
                p1 = ax.plot(x, d(cepf), label='cep(f={})'.format(f_val), color=list(colors.values())[col])
            col += 1

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.7, box.height*1.0])

        legs = [a.get_label() for a in p1]
        ax.legend(p1, legs, loc=0, bbox_to_anchor=(1.0, 1.0))

    logstr = ''

    if log:
        plt.yscale('log')
        logstr = '_log'

    plt.ylabel('c in Lagoon [cfu]/[pfu]')
    plt.title('Concentration of phage producing E. coli')
    plt.xlabel('Time [min]')

    plt.savefig(os.path.join(summariesdir, '{}{}{}_dist_{}.png'.format(prefix, name, logstr, suffix)))
    plt.gcf().clear()

#
# def drawdist(x, log, name, prefix, suffix, o, v, summariesdir):
#     plt.figure(1, dpi=300)
#     col = 0
#     if o.plot_dist == "True":
#         for f_val in list(v.dist_f[0].keys()):
#             cepf = np.asarray([v.dist_f[t][f_val] for t in range(len(v.dist_f) - 1)]) \
#                    * np.asarray(v.cep[:-1])
#             plt.plot(x, d(cepf), label='cep(f={})'.format(f_val), color=list(colors.values())[col])
#             col += 1
#     plt.legend()
#     logstr = ''
#
#     if log:
#         plt.yscale('log')
#         logstr = '_log'
#
#     plt.ylabel('c in Lagoon [cfu]/[pfu]')
#     plt.title('Concentration of phage producing E. coli')
#     plt.xlabel('Time [min]')
#
#     plt.savefig(os.path.join(summariesdir, '{}{}{}_dist_{}.png'.format(prefix, name, logstr, suffix)))
#     plt.gcf().clear()



def plot_meta(reduced, maxepochs, summariesdir, prefix='', suffix='', vl=1):

    scoret = np.ndarray([len(reduced),
                         len(list(reduced.values())[0]),
                         len(list(list(reduced.values())[0].values())[0])])

    akeys = list(reduced.keys())
    xkeys = list(reduced[akeys[0]].keys())
    ykeys = list(reduced[akeys[0]][xkeys[0]].keys())

    for a in range(scoret.shape[0]):
        for x in range(scoret.shape[1]):
            for y in range(scoret.shape[2]):
                scoret[a, x, y] = coltransform(reduced[akeys[a]][xkeys[x]][ykeys[y]], maxepochs)

    draw_meta(scoret, akeys, xkeys, ykeys, prefix, suffix + 'f', 'Time in one lagoon [min]', 'Transfer volume/lagoon volume', 'Search for tl and vt  with f', summariesdir)

    for a in range(scoret.shape[1]):
        for x in range(scoret.shape[0]):
            for y in range(scoret.shape[2]):
                scoret[x, a, y] = coltransform(reduced[akeys[a]][xkeys[x]][ykeys[y]], maxepochs)

    draw_meta(scoret, xkeys, akeys, ykeys, prefix, suffix + 't', 'Initial fitness', 'Transfer volume/lagoon volume', 'Search for f and vt  with tl', summariesdir)


    for a in range(scoret.shape[2]):
        for x in range(scoret.shape[0]):
            for y in range(scoret.shape[1]):
                scoret[y, a, x] = coltransform(reduced[akeys[a]][xkeys[x]][ykeys[y]], maxepochs)

    draw_meta(scoret, list(np.asarray(ykeys)/vl), akeys, xkeys, prefix, suffix + 'v', 'Initial fitness', 'Time in one lagoon [min]', 'Search for f0 and tl  with vt/vl', summariesdir)


def draw_meta(scoret, akeys, xkeys, ykeys, prefix, suffix, xlabel, ylabel, title, summariesdir):

    ccmap = mplcolors.LinearSegmentedColormap('by_cmap', cdict)
    count = 1

    for fdraw in range(scoret.shape[0]):
        fig, ax = plt.subplots(dpi=300)

        pcm = ax.imshow(scoret[fdraw, :, :], origin='lower',
                        extent=[xkeys[0], xkeys[-1], ykeys[0], ykeys[-1]],
                        aspect='auto',
                        cmap = ccmap,
                        clim=[-1, 1])
        clb = fig.colorbar(pcm, ax=ax, extend='both',
                           ticks=np.asarray([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]))
        clb.set_ticklabels([-0, -25, -50, -75, 100, 75, 50, 25, 0])
        clb.set_label('# of epochs the phage concentration accepted', y=0.5)
        clb.set_clim(-1, 1)
        plt.title(title + ' = {:.3f}'.format(akeys[fdraw]))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        plt.savefig(os.path.join(summariesdir, "{}meta_a_{:.3f}_{}.png".format(prefix, akeys[fdraw], suffix)))
        plt.close()
        print('Saved {}th figure.'.format(count))
        count += 1


def plot_dist_f(v, o, prefix, suffix, summariesdir):
    xkeys = range(len(v.dist_f))
    ykeys = list(v.dist_f[0].keys())
    f_share = np.ndarray([len(ykeys), len(xkeys)])


    for x in range(f_share.shape[1]):
        for y in range(f_share.shape[0]):
                f_share[y, x] = v.dist_f[xkeys[x]][ykeys[y]]
    ccmap = mplcolors.LinearSegmentedColormap('by_cmap', cdict2)
    fig, ax = plt.subplots(dpi=300)

    pcm = ax.imshow(f_share, origin='lower',
                    extent=[v.time[0], v.time[-1], ykeys[0], ykeys[-1]],
                    aspect='auto',
                    cmap=ccmap)
    clb = fig.colorbar(pcm, ax=ax)

    clb.set_label('share of given share of M13 wt fitness', y=0.5)
    plt.title('Development of fitness distribution')
    plt.ylabel('Fitness relative to wt M13 fitness')
    plt.xlabel('Time [min]')

    plt.savefig(os.path.join(summariesdir, "{}fitness_distribution_{}.png".format(prefix, suffix)))
    plt.close()
