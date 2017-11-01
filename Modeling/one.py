"""
Writes plots. A config dict needs to be passed.
"""

import sys
import pickle
import os
import json
import pprint

import predcel_plot as plot_predcel
import predcel_calc as calc_predcel

def main():
    config_json = sys.argv[1]
    with open(config_json) as config_fobj:
        config_dict = json.load(config_fobj)

    o, v = calc_predcel.initialisation(config_dict)

    summariesdir = sys.argv[2]
    if not os.path.exists(summariesdir):
        os.mkdir(summariesdir)

    logfile = open(os.path.join(sys.argv[2], 'logfile.txt'), 'w')
    if config_dict['fitnessmode'] == 'dist':
        calc_predcel.dist_setup(o, v, logfile)
    else:
        calc_predcel.setup(o, v, logfile)

    plot_predcel.plot(o, v, suffix='skew', prefix='', summariesdir=summariesdir)
    open(os.path.join(summariesdir, 'config.json'), 'w').write(
        open(sys.argv[1], 'r').read())
    print('Done.')


if __name__ == '__main__':
    main()
