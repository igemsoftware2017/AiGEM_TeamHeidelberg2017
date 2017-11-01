import sys
import os
import pickle

import gaia as gaia


def main():
    """sys.argv[1] is the summariesdir, where gaia wrote the data."""
    os.environ['MPLCONFIGDIR'] = "." # set this to something reasonable not mounted
    highlight_names = ['GO:0004818', 'GO:0004827']
    # save all used scripts to the summaries dir

    summaries_dir = sys.argv[1]
    if not os.path.exists(summaries_dir):
        os.mkdir(summaries_dir)
    if not os.path.exists(os.path.join(summaries_dir, 'scripts')):
        os.mkdir(os.path.join(summaries_dir, 'scripts'))

    with open(os.path.join(summaries_dir, 'reduced.p'), 'rb') as pfile:
        data = pickle.load(pfile)

    gaia.plot_all_data(
        data=data, wdir=summaries_dir,
        logfile=open(os.path.join(summaries_dir, 'plot_logfile.txt'), 'w'),
        highl_names=highlight_names,
        highl=[175, 154],
        goal=174, f_width=6, f_height=4, res=300, name='data_HD.png')
    return

if __name__ == '__main__':
    main()
