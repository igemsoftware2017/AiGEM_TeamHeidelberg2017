"""
Train DeeProtein. A config dict needs to be passed.
"""
import argparse
import json
from DeeProtein import DeeProtein
import helpers
import os

def main():
    with open(FLAGS.config_json) as config_fobj:
        config_dict = json.load(config_fobj)

    # set the gpu context
    if not FLAGS.gpu:
        if config_dict["gpu"] == 'True':
            config_dict["gpu"] = "False"

    # save all used scripts to the summaries dir
    summaries_dir = config_dict['summaries_dir']
    if not os.path.exists(summaries_dir):
        os.mkdir(summaries_dir)
    if not os.path.exists(os.path.join(summaries_dir, 'scripts')):
        os.mkdir(os.path.join(summaries_dir, 'scripts'))

    optionhandler = helpers.OptionHandler(config_dict)
    model = DeeProtein(optionhandler)
    model.train(restore_whole=FLAGS.restore_whole)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_json',
        type=str,
        required=True,
        help='Path to the config.JSON')
    parser.add_argument(
        '--restore_whole',
        type=str,
        default=True,
        help='Wheter to restore the whole model including the outlayer '
             '(optional). Defaults to True.')
    parser.add_argument(
        '--gpu',
        type=str,
        default=True,
        help='Wheter to train in gpu context or not '
             '(optional). Defaults to True.')
    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print('Error, unrecognized flags:', unparsed)
        exit(-1)
    main()
