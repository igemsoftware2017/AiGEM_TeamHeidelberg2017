""" Invoke the Model in validation mode and perform a run over the valid set."""
import sys
import json
from DeeProtein import DeeProtein
import helpers


def main():
    config_json = sys.argv[1]
    with open(config_json) as config_fobj:
        config_dict = json.load(config_fobj)

    optionhandler = helpers.OptionHandler(config_dict)
    model = DeeProtein(optionhandler)
    model.validate()

if __name__ == '__main__':
    main()
