"""
Generate tf.Records files for the Training of DeeProtein.
"""
import json
import argparse
from helpers import OptionHandler, TFrecords_generator


def main():
    with open(FLAGS.config_json) as config_fobj:
        config_dict = json.load(config_fobj)

    optionhandler = OptionHandler(config_dict)
    batchgen = TFrecords_generator(optionhandler)
    print('Initialized TFRecords generator')
    print('Producing batches...')
    batchgen.produce_train_valid()
    print('DONE.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_json',
        type=str,
        required=True,
        help='Path to the config.JSON')
    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print('Error, unrecognized flags:', unparsed)
        exit(-1)
    main()