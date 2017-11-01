"""
Writes plots. A config dict needs to be passed.
sys.argv[1] -> config.json
sys.argv[2] -> summariesdir, also data is loaded from there
"""

import sys
import pickle
import os
import json

import predcel_plot as predcel


def main():
    config_json = sys.argv[1]
    with open(config_json, 'r') as config_fobj:
        config_dict = json.load(config_fobj)

    reduced = pickle.load(open(os.path.join(sys.argv[2], 'reduced.p'), 'rb'))
    summariesdir = sys.argv[2]

    predcel.plot_meta(reduced, summariesdir=summariesdir, maxepochs=config_dict['epochs'], suffix='5', vl=config_dict['vl'])
    open(os.path.join(sys.argv[2], 'config.json'), 'w').write(
        open(sys.argv[1], 'r').read())

if __name__ == '__main__':
    main()
