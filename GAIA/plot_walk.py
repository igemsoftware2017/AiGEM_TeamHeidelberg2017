import sys
import os
import pickle

import gaia as gaia


def main():
    os.environ['MPLCONFIGDIR'] = "." # set this to something reasonable not mounted
    """sys.argv[1] is the summariesdir, where gaia wrote the data."""

    # save all used scripts to the summaries dir

    summaries_dir = sys.argv[1]
    if not os.path.exists(summaries_dir):
        os.mkdir(summaries_dir)
    if not os.path.exists(os.path.join(summaries_dir, 'scripts')):
        os.mkdir(os.path.join(summaries_dir, 'scripts'))

    with open(os.path.join(summaries_dir, 'walk_data.p'), 'rb') as pfile:
        data = pickle.load(pfile)

    with open(os.path.join(summaries_dir, 'wt_scores.p'), 'rb') as pfile:
        wtscores = pickle.load(pfile)
    gaia.plot_walk(data=data,
                   goal=174,
                   wdir=summaries_dir,
                   logfile=open(os.path.join(summaries_dir, 'plot_walk_logfile.txt'), 'w'),
                   wt_scores=wtscores,
                   highl=[175, 154],
                   highl_names=['GO:0004818', 'GO:0004827'],
                   aas = range(20),
                   f_width=5,
                   f_height=4,
                   res=300, name='walk_data_HD.png')
    return

if __name__ == '__main__':
    main()
