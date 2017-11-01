"""
Calculate metadata. A config dict needs to be passed.
sys.argv[1] -> config.json
sys.argv[2] -> summaries directory
sys.argv[3] -> number of datapoints
"""

import sys
import predcel_calc as predcel
import pickle
import os

def main():
    config_json = sys.argv[1]
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])
    logfile = open(os.path.join(sys.argv[2], 'logfile.txt'), 'w')
    reduced, data = predcel.calc(config_json, sys.argv[2], int(sys.argv[3]), logfile)

    pickle.dump(reduced, open(os.path.join(sys.argv[2], 'reduced.p'), 'wb'))
    pickle.dump(data, open(os.path.join(sys.argv[2], 'data.p'), 'wb'))

    open(os.path.join(sys.argv[2], 'config.json'), 'w').write(
        open(sys.argv[1], 'r').read())

if __name__ == '__main__':
    main()
