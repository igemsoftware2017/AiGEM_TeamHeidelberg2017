"""
Writes plots for a single PREDCEL experiment modeled. A config dict needs to be passed.
sys.argv[1] -> config.json
sys.argv[2] -> summaries directory
"""

import sys
import pickle
import os
import json
import pprint

import predcel_plot as plot
import predcel_calc as calc


def main():
    config_json = sys.argv[1]
    with open(config_json, 'r') as config_fobj:
        config_dict = json.load(config_fobj)
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])


    logfile = open(os.path.join(sys.argv[2], 'logfile.txt'), 'w')
    pprint.pprint(config_dict, logfile)
    o, v = calc.initialisation(config_dict)
    if o.fitnessmode == 'dist':
        print('dist')
        calc.dist_setup(o, v, logfile)
    else:
        print('not dist')
        calc.setup(o, v, logfile)

    with open(os.path.join(sys.argv[2], 'data.p'), 'wb') as pfile:
        pickle.dump([o, v, config_dict], pfile)
    plot.plot(o, v, summariesdir=sys.argv[2])

if __name__ == '__main__':
    main()
