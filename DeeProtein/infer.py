"""
Inference mode for DeeProtein.
"""
import tensorflow as tf
import numpy as np
import json
import sys
import helpers
from goatools.obo_parser import GODag
from DeeProtein import DeeProtein
np.set_printoptions(threshold=np.inf) #for debug

def main():
    seq = sys.argv[1]
    config_json = "/net/data.isilon/igem/2017/scripts/DeeProtein/config/DeeProtein_configINFERENCE1509.JSON"
    GOdag = GODag('/net/data.isilon/igem/2017/data/gene_ontology/go.obo', optional_attrs=['relationship'])
    with open(config_json) as config_fobj:
        config_dict = json.load(config_fobj)
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
        deeprotein.restore_model_from_checkpoint('/net/data.isilon/igem/2017/data/nn_train/DeeProtein_UNIPROT_SQEMBED_FLt_noGarbage_ResNet20_886_INFERENCE/checkpoints/ckpt', deeprotein.session)

        # restore all the weights
        # _ = deeprotein.load_model_weights(inference_net, session=deeprotein.session,
        #                                   name='Classifier') #THIS RUNS THE SESSION INTERNALLY

        encoded_seq = deeprotein.batchgen._encode_single_seq(seq)
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

        # save this shit into a checkpoint:
        # deeprotein.saver.save(deeprotein.session, '/net/data.isilon/igem/2017/data/nn_train/DeeProtein_UNIPROT_SQEMBED'
        #                                           '_FLt_noGarbage_ResNet20_886_INFERENCE/checkpoints/')

        # save all params to a binary:
        deeprotein.save_params(inference_net, session=deeprotein.session, ignore=None)

        #predicted_roots = [go for go in predicted_classes if not GOdag[go].get_all_children()]

        for i, go in enumerate(predicted_classes):
            try:
                go = go.split('_')[1]
            except:
                pass
            deeprotein.log_file.write('%s\t%s:\t%f, %f\n' % (go, GOdag[go].name,
                                                             class_logits[0, predicted_IDs[i]],
                                                             class_variance[0, predicted_IDs[i]]))
        if not predicted_classes:
            deeprotein.log_file.write('NONE\n\n')

if __name__ == '__main__':
    main()
