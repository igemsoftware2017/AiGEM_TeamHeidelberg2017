"""
Inference mode for DeeProtein.
"""
import tensorflow as tf
import numpy as np
import json
import wget
import helpers
import argparse
from goatools.obo_parser import GODag
from DeeProtein import DeeProtein


def main():
    if FLAGS.GOdag:
        obo_file = FLAGS.obo_file
    else: # no GOdag file was specified -> download it.
        url = 'http://purl.obolibrary.org/obo/go.obo'
        obo_file = wget.download(url)
    GOdag = GODag(obo_file, optional_attrs=['relationship'])

    with open(FLAGS.config_json) as config_fobj:
        config_dict = json.load(config_fobj)

    # set the gpu context
    if not FLAGS.gpu:
        if config_dict["gpu"] == 'True':
            config_dict["gpu"] = "False"

    opts = helpers.OptionHandler(config_dict)
    deeprotein = DeeProtein(opts, inference=True)

    with tf.Graph().as_default():
        deeprotein.initialize_helpers()

        # graph for inference:
        deeprotein.is_train = False
        deeprotein._opts._batchsize = 1
        input_seq_node = tf.placeholder(tf.float32, [deeprotein._opts._batchsize, deeprotein._opts._depth,
                                                     deeprotein._opts._windowlength, 1])
        inference_net, _ = deeprotein.model(input_seq_node, valid_mode=False)
        double_sigmoid_logits = tf.nn.sigmoid(inference_net.outputs)

        # as its a double outlayer take a mean and get the var
        mean_inf_sigmoid_logits, var_inf_sigmoid_logits = tf.nn.moments(double_sigmoid_logits,
                                                                        axes=[2])

        deeprotein.session.run(tf.global_variables_initializer())
        deeprotein.saver = tf.train.Saver()
        # restore the session from ckpt
        if FLAGS.restore_from_explicit_ckpt:
            deeprotein.restore_model_from_checkpoint(FLAGS.restore_from_explicit_ckpt, deeprotein.session)
        else:
        # restore all the weights from dir in checkpointfile:
            _ = deeprotein.load_model_weights(inference_net, session=deeprotein.session,
                                              name='Classifier') #THIS RUNS THE SESSION INTERNALLY

        encoded_seq = deeprotein.batchgen._encode_single_seq(FLAGS.sequence)
        # expand it to batch dim and channels dim:
        assert len(encoded_seq.shape) == 2
        encoded_seq = np.reshape(encoded_seq, [1, encoded_seq.shape[0], encoded_seq.shape[1], 1])

        # run the session and retrieve the predicted classes
        class_logits, class_variance = deeprotein.session.run([mean_inf_sigmoid_logits, var_inf_sigmoid_logits],
                                                              feed_dict={input_seq_node: encoded_seq})
        # logits is a 2D array of shape [1, nclasses]
        thresholded_logits = np.where(class_logits >= 0.5)

        # get the IDs of the classes:
        predicted_IDs = thresholded_logits[1].tolist()

        # get the corresponding classes from the EC_file:
        predicted_classes = [deeprotein.batchgen.id_to_class[_id] for _id in predicted_IDs]
        deeprotein.log_file.write('Predicted classes:\n')


        for i, go in enumerate(predicted_classes):
            try:
                go = go.split('_')[1]
            except:
                pass
            output = '%s\t%s:\t%f, %f\n' % (go, GOdag[go].name,
                                                             class_logits[0, predicted_IDs[i]],
                                                             class_variance[0, predicted_IDs[i]])
            deeprotein.log_file.write(output)
            print(output)
        if not predicted_classes:
            output = 'The sequence could not be classified with enough fidelity.\n\n'
            deeprotein.log_file.write(output)
            print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sequence',
        type=str,
        required=True,
        help='The amino_acid sequence to classify. Should at least 175AA long and contain only canonical AAs.')
    parser.add_argument(
        '--config_json',
        type=str,
        required=True,
        help='Path to the config.JSON')
    parser.add_argument(
        '--obo_file',
        type=str,
        default=False,
        help='The path to the download of the gene ontology file (go.obo). '
             'If None is spefified, the file will be downloaded prior to inference.')
    parser.add_argument(
        '--restore_from_explicit_ckpt',
        type=str,
        default=False,
        help='Restore the model from this explicit ckpt and not form the dir/ specified in config_json.')
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
