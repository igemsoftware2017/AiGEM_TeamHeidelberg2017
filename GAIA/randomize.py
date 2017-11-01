"""
Randomizes a sequence, plots scores of DeeProtein. A config dict needs to be passed.
"""
import json
import os
import argparse

from DeeProtein import DeeProtein
import helpers as helpers
from gaia import GeneticAlg


def main():
    # load the config_json into the optionhandler object
    config_json = FLAGS.config_json
    with open(config_json) as config_fobj:
        config_dict = json.load(config_fobj)
    optionhandler = helpers.OptionHandler(config_dict)

    # Handle the summaries_dir output directory
    summaries_dir = optionhandler._summariesdir
    if FLAGS.output_dir:
        summaries_dir = FLAGS.output_dir
    if not os.path.exists(summaries_dir):
        os.mkdir(summaries_dir)
    if not os.path.exists(os.path.join(summaries_dir, 'scripts')):
        os.mkdir(os.path.join(summaries_dir, 'scripts'))

    # Enable overwriting the sequence path
    if FLAGS.sequence:
        optionhandler.seqfile = FLAGS.sequence

    classifier = DeeProtein(optionhandler)
    classifier.init_for_machine_infer(optionhandler._batchsize)

    evolver = GeneticAlg(optionhandler, classifier, helpers.TFrecords_generator(optionhandler))
    evolver.randomize_all(FLAGS.amount2randomize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_json',
        type=str,
  #      aliases=['c'],
        required=True,
        help='Path to the config.JSON')
    parser.add_argument(
        '--amount2randomize',
        type=int,
        #      aliases=['a'],
        required=True,
        help='Maximum number of residues to mutate')
    parser.add_argument(
        '--sequence',
        type=str,
  #      aliases=['s'],
        default=False,
        help='''Path to the sequence file.
        Structure of sequence file.
        -----------------
        Line:
        1 (optional) starts with '>' contains title of the stats.png plot
        2 '[Float]>GO:[GO-term],[Float]>GO:[GO-term],[...]' contains goal GO-terms with their weight
        3 'Float]>GO:[GO-term],[Float]>GO:[GO-term],[...]' contains avoid GO-terms with their weights
        4 [Sequence to evolve]
        5 'Maxmut: [Integer]' contains maximum number of mutations
        6 'Notgarbage weight: [Float]' contains weight for not_garbage score
        7 'Garbage_weight: [Float]' cotains weight for garbage score''')
    parser.add_argument(
        '--output_dir',
   #     aliases=['o'],
        type=str,
        default=False,
        help='Path to the directory for the output. Overwrites the summaries_dir given in the config JSON.')
    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print('Error, unrecognized flags:', unparsed)
        exit(-1)
    main()
