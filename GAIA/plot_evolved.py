import sys
import os

from genetic_alg import Stats


def main():
    """sys.argv[1] is the summariesdir gaia used for the data to plot"""
    # save all used scripts to the summaries dir

    summaries_dir = sys.argv[1]
    if not os.path.exists(summaries_dir):
        os.mkdir(summaries_dir)
    if not os.path.exists(os.path.join(summaries_dir, 'scripts')):
        os.mkdir(os.path.join(summaries_dir, 'scripts'))

    stats = Stats(pickledir=os.path.join(summaries_dir, 'stats.p'))
    stats.plot_all(f_width=20, f_height=8, res=200, name='stats_HD.png')
    stats.plotdistoverseq(f_width=20, f_height=8, res=200, name='hist_HD.png')
    return

if __name__ == '__main__':
    main()
