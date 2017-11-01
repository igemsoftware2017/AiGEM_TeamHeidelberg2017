""" Invoke the Model in validation mode and perform a run over the valid set."""
import argparse
import json
from DeeProtein import DeeProtein
import helpers


def main():
    with open(FLAGS.config_json) as config_fobj:
        config_dict = json.load(config_fobj)

    # set the gpu context
    if not FLAGS.gpu:
        if config_dict["gpu"] == 'True':
            config_dict["gpu"] = "False"

    optionhandler = helpers.OptionHandler(config_dict)
    model = DeeProtein(optionhandler)
    model.validate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_json',
        type=str,
        required=True,
        help='Path to the config.JSON')
    parser.add_argument(
        '--gpu',
        type=str,
        default=True,
        help='Whether to train in gpu context or not '
             '(optional). Defaults to True.')
    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print('Error, unrecognized flags:', unparsed)
        exit(-1)
    main()
