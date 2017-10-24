"""
Preprocess the data to train a DeeProtein model
"""
import argparse
from DatasetGenerator import DatasetGenerator

def main():
    """
    The main function used to call the methods from the datasetgenerator.
    """
    dsgen = DatasetGenerator(FLAGS.uniprotfile_path, FLAGS.uniprot_csv, FLAGS.save_dir)
    dsgen.uniprot_to_csv()
    dsgen.separate_classes_by_GO()
    dsgen.filter_count_and_write_all()
    dsgen.generate_dataset_by_GO_list(GO_file=FLAGS.EC_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--uniprot_path',
        type=str,
        required=True,
        help='Path to the uniprot/swissprot download .txt.')
    parser.add_argument(
        '--uniprot_csv',
        type=str,
        required=True,
        help='The path where to output the resulting uniprot.csv. '
             'If file exists still pass the path.')
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='The output dir. Does not need to match the two previous filepaths.')
    parser.add_argument(
        '--EC_File',
        type=str,
        default='',
        help='A file containing the EC/GO Terms for which to generate a dataset from the processed uniprot download. '
             'If an EC_file is passed the script expects the processed download in the --save_dir.')

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print('Error, unrecognized flags:', unparsed)
        exit(-1)
    main()


