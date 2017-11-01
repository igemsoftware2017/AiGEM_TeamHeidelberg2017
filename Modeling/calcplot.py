"""
Calculate metadata and plot it. A config dict needs to be passed.
sys.argv[1] -> config.json
sys.argv[2] -> summaries directory
sys.argv[3] -> number of datapoints
"""

import sys
import os
import json
import pprint

import predcel_plot as plot_predcel
import predcel_calc as calc_predcel
import pickle


def main():
    config_json = sys.argv[1]
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])

    with open(config_json, 'rb') as ifile:
        config_dict = json.load(ifile)
    logfile = open(os.path.join(sys.argv[2], 'logfile.txt'), 'w')
    pprint.pprint(config_dict, logfile)

    reduced, data = calc_predcel.calc(config_json, sys.argv[2], int(sys.argv[3]), logfile)
    pickle.dump(reduced, open(os.path.join(sys.argv[2], 'reduced.p'), 'wb'))
    plot_predcel.plot_meta(reduced, maxepochs=config_dict['epochs'], prefix='', summariesdir=sys.argv[2], vl=config_dict['vl'])

    print('Done.')

if __name__ == '__main__':
    main()