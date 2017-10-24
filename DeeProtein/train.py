"""
Train DeeProtein. A config dict needs to be passed.
"""
import sys
import json
from DeeProtein import DeeProtein
import helpers
import shutil
import os

def main():
    config_json = sys.argv[1]
    if sys.argv[2] == 'True':
        restore_whole = True
    elif sys.argv[2] == 'False':
        restore_whole = False

    with open(config_json) as config_fobj:
        config_dict = json.load(config_fobj)

    # save all used scripts to the summaries dir
    summaries_dir = config_dict['summaries_dir']
    if not os.path.exists(summaries_dir):
        os.mkdir(summaries_dir)
    if not os.path.exists(os.path.join(summaries_dir, 'scripts')):
        os.mkdir(os.path.join(summaries_dir, 'scripts'))

    optionhandler = helpers.OptionHandler(config_dict)
    model = DeeProtein(optionhandler)
    model.train(restore_whole=restore_whole)

if __name__ == '__main__':
    main()
