import helpers as helpers
import customlayers as customlayers
import tensorlayer as tl
import tensorflow as tf
import time, os, glob
import pickle
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn import decomposition
np.set_printoptions(threshold=np.inf) #for debug


class DeeProtein():
    def __init__(self, optionhandler, inference=False):
        self._opts = optionhandler
        self._opts.write_dict()
        self._embedding_saver = 'depracted'
        self._kmer2vec_embedding = 'depracted'
        self._kmer2id = 'depracted'
        self.reader = ''
        self.saver = 'saver_object'
        self.graph_complete = 'the whole model'
        self.graph_conv_part = 'the conv part of the model'
        self.opt = 'It\'s me ADAGRAD'
        self.is_train = True
        self.batchgen = 'where we will put the batchgenerator'
        self.ROCtracker = 'where we will put the ROCtracker'
        if not inference:
            self.log_file = open(self._opts._summariesdir + '/model.log', "w", 1)
        else:
            self.log_file = open(self._opts._summariesdir + '/inference.log', "w", 1)
        self.valid_graph_initialized = False
        # load_ckpt the kmer2vec dict
        # try:
        #     with open(self._opts._kmer2vec_kmerdict, 'rb') as pickle_f:
        #         self._kmer2id = pickle.load(pickle_f)
        # except:
        #     with open(self._opts._kmer2vec_kmercounts, 'r') as in_fobj:
        #         for line in in_fobj:
        #             kmer = line.strip().split()[0]
        #             self._kmer2id[kmer] = len(self._kmer2id)

        print(self._opts._batchgenmode)

    def save_model(self, network, session, step, name='DeeProtein'):
        """Save the dataset to a file.npz so the model can be reloaded."""
        # save model as dict:
        param_save_dir = os.path.join(self._opts._summariesdir,
                                      'checkpoint_saves/')
        conv_vars = [var for var in network.all_params if 'dense' and 'outlayer' not in var.name]

        lower_enc = [var for var in network.all_params if 'lower_encoder' in var.name]

        # print(conv_vars)  #TODO delete this

        if not os.path.exists(param_save_dir):
            os.makedirs(param_save_dir)
        if conv_vars:
            tl.files.save_npz_dict(conv_vars,
                                   name=os.path.join(param_save_dir, '%s_conv_part.npz' % name),
                                   sess=session)
        if name == 'Encoder':
            tl.files.save_npz_dict(lower_enc,
                                   name=os.path.join(param_save_dir, '%s_lower_enc_part.npz' % name),
                                   sess=session)

        tl.files.save_npz_dict(network.all_params,
                               name=os.path.join(param_save_dir, '%s_complete.npz' % name),
                               sess=session)

        # save also as checkpoint
        ckpt_file_path = os.path.join(param_save_dir, '%s.ckpt' % name)
        self.saver.save(session, ckpt_file_path, global_step=step)

    def load_conv_weights_npz(self, network, session, name='DeeProtein'):
        """
        Load the weights for the convolutional layers from a pretrained model
        :return:
        """
        # check if filepath exists:
        file = os.path.join(self._opts._restorepath, '%s_conv_part.npz' % name)
        self.log_file.write('[*] Loading %s\n' % file)
        if tl.files.file_exists(file):
            # custom load_ckpt op:
            d = np.load(file)
            params = [val[1] for val in sorted(d.items(), key=lambda tup: int(tup[0]))]
            # params = [p for p in params if not 'outlayer' in p.name]
            # original OP:
            # params = tl.files.load_npz_dict(name=file)
            # if name == 'Classifier':
            #     params = [p for p in params[:-4]]
            tl.files.assign_params(session, params, network)
            self.log_file.write('[*] Restored conv weights!\n')
            return network
        else:
            self.log_file.write('[*] Loading %s FAILED. File not found.\n' % file)
            return False

    def load_model_weights(self, network, session, include_outlayer=False, name='DeeProtein'):
        """
        Load the weights for the convolutional layers from a pretrained model
        :return:
        """

        # check if filepath exists:
        file = os.path.join(self._opts._restorepath, '%s_complete.npz' % name)
        if tl.files.file_exists(file):
            # custom load_ckpt op:
            d = np.load(file)
            params = [val[1] for val in sorted(d.items(), key=lambda tup: int(tup[0]))]
            #if name == 'Classifier' and not include_outlayer:
                #params = [p for p in params[:-4]]
            # print(params)
            # original OP:
            # params = tl.files.load_npz_dict(name=file)
            tl.files.assign_params(session, params, network)
            self.log_file.write('[*] Restored model weights!\n')
            print('[*] Restored model weights!\n')
            return network
        else:
            self.log_file.write('[*] Loading %s FAILED. File not found.\n' % file)
            print('[*] Loading %s FAILED. File not found.\n' % file)
            return False

    def load_complete_model_eval(self, network, session, name='DeeProtein'):
        """
        Restores the complete model from its latest save (.npz) for evaluation.
        :return:
        """
        # the directory where we saved our checkpoints:
        file = os.path.join(self._opts._summariesdir,
                            'checkpoint_saves/%s_complete.npz' % name)
        # check if filepath exists:
        if tl.files.file_exists(file):
            # custom load_ckpt op:
            d = np.load(file)
            params = [val[1] for val in sorted(d.items(), key=lambda tup: int(tup[0]))]
            # original OP:
            # params = tl.files.load_npz_dict(name=file)
            tl.files.assign_params(session, params, network)
            self.log_file.write('[*] Restored model for inference!\n')
            return network
        else:
            self.log_file.write('[*] Loading %s FAILED. File not found.\n' % file)
            return False

    def check_data(self, tfrecords_filename):
        """
        Check if the data format in the example files is correct
        :param tfrecords_filename:
        :return:
        """

        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

        for string_record in record_iterator:
            # Parse the next example
            example = tf.train.Example()
            example.ParseFromString(string_record)

            # Get the features you stored (change to match your tfrecord writing code)
            seq = (example.features.feature['seq_raw']
                   .bytes_list
                   .value[0])

            label = (example.features.feature['label_raw']
                     .bytes_list
                     .value[0])

            # Convert to a numpy array (change dtype to the datatype you stored)
            seq_array = np.fromstring(seq, dtype=np.float64)
            label_array = np.fromstring(label, dtype=np.float64)

            # Print the image shape; does it match your expectations?
            print(seq_array.shape)
            print(label_array.shape)

    def check_data_comprehensively(self, file_paths, valid_mode=True):
        """
        Check if the data format in the example files is correct
        :param tfrecords_filename:
        :return:
        """

        filename_queue = tf.train.string_input_producer(file_paths, num_epochs=None, shuffle=False, seed=None,
                                                        capacity=10, shared_name=None, name='fileQueue', cancel_op=None)

        id = 'valid' if valid_mode else 'train'

        reader = tf.TFRecordReader()

        with tf.name_scope('input_pipeline_%s' % id):
            _, serialized_batch = reader.read(filename_queue)

            features = tf.parse_single_example(
                serialized_batch, features={
                    'windowlength': tf.FixedLenFeature([], tf.int64),
                    'depth': tf.FixedLenFeature([], tf.int64),
                    'label_classes': tf.FixedLenFeature([], tf.int64),
                    'seq_raw': tf.FixedLenFeature([], tf.string),
                    'label_raw': tf.FixedLenFeature([], tf.string),
                }
            )
            seq_tensor = tf.cast(tf.decode_raw(features['seq_raw'], tf.float64), tf.float32)
            label = tf.cast(tf.decode_raw(features['label_raw'], tf.float64), tf.float32)

            # tf.Print(seq_tensor, [seq_tensor])
            # tf.Print(label, [label])

            windowlength = tf.cast(features['windowlength'], tf.int32)
            depth = tf.cast(features['depth'], tf.int32)

            #TODO check if we neeed this
            n_classes = tf.cast(features['label_classes'], tf.int32)

            seq_shape = tf.stack([depth, windowlength])
            label_shape = [n_classes]

            seq_tensor = tf.expand_dims(tf.reshape(seq_tensor, seq_shape), -1)

            if self._opts._batchgenmode.startswith('one_hot'):
                seq_tensor.set_shape([self._opts._depth, self._opts._windowlength, 1])
            elif self._opts._batchgenmode.startswith('embed'):
                seq_tensor.set_shape([self._opts._embeddingdim, self._opts._windowlength, 1])

            #label = tf.reshape(label, label_shape)
            label.set_shape([self._opts._nclasses])

            with tf.Session() as check_data_sess:
                # initialize everything:
                check_data_sess.run(tf.global_variables_initializer())
                check_data_sess.run(tf.local_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=check_data_sess)
                #print(self.batchgen.class_dict)

                for _ in range(10000000000):

                    seqx, labelx = check_data_sess.run([seq_tensor, label])
                    seqx = np.asarray(seqx)
                    labelx = np.asarray(labelx)
                    print(seqx.shape)
                    print(labelx.shape)
                    print(np.argmax(labelx, axis=0))

            # gracefully shut down the queue
            coord.request_stop()
            coord.join(threads)

    def input_pipeline(self, file_paths, valid_mode=False):
        """
        Construct an input pipeline for training or validation, depending on the passed filepaths,
        NOTE: THE VALIDATION MODE IS ONLY IMPORTANT AT GRAPH CONSTRUCTION TIME
        :param file_paths:  [TFrecords files]
        :param valid_mode: FALSE: training, TRUE: validation mode
        :return: inputs for the specified VALIDATION MODE
        """
        print('%d files found' % len(file_paths))

        #set epochs to 1 in validation mode:
        epochs = self._opts._numepochs if not valid_mode else 1

        filename_queue = tf.train.string_input_producer(file_paths, num_epochs=epochs, shuffle=False, seed=None,
                                                        capacity=100, shared_name=None, name='fileQueue', cancel_op=None)

        id = 'valid' if valid_mode else 'train'

        reader = tf.TFRecordReader()

        with tf.name_scope('input_pipeline_%s' % id):
            _, serialized_batch = reader.read(filename_queue)

            features = tf.parse_single_example(
                serialized_batch, features={
                    'windowlength': tf.FixedLenFeature([], tf.int64),
                    'depth': tf.FixedLenFeature([], tf.int64),
                    'label_classes': tf.FixedLenFeature([], tf.int64),
                    'seq_raw': tf.FixedLenFeature([], tf.string),
                    'label_raw': tf.FixedLenFeature([], tf.string),
                }
            )
            if self._opts._batchgenmode.startswith('embed'):
                seq_tensor = tf.cast(tf.decode_raw(features['seq_raw'], tf.float32), tf.float32)
            elif self._opts._batchgenmode.startswith('one_hot'):
                seq_tensor = tf.cast(tf.decode_raw(features['seq_raw'], tf.float64), tf.float32)
            label = tf.cast(tf.decode_raw(features['label_raw'], tf.float64), tf.float32)

            # tf.Print(seq_tensor, [seq_tensor])
            # tf.Print(label, [label])

            windowlength = tf.cast(features['windowlength'], tf.int32)
            depth = tf.cast(features['depth'], tf.int32)

            #TODO check if we neeed this
            n_classes = tf.cast(features['label_classes'], tf.int32)

            seq_shape = tf.stack([depth, windowlength])
            #struc_shape = tf.stack([windowlength])
            label_shape = [n_classes]

            seq_tensor = tf.expand_dims(tf.reshape(seq_tensor, seq_shape), -1)
            #structure_label = tf.reshape(structure_label, struc_shape)

            if self._opts._batchgenmode.startswith('one_hot'):
                seq_tensor.set_shape([self._opts._depth, self._opts._windowlength, 1])
            elif self._opts._batchgenmode.startswith('embed'):
                seq_tensor.set_shape([self._opts._embeddingdim, self._opts._windowlength, 1])

            #label = tf.reshape(label, label_shape)
            label.set_shape([self._opts._nclasses])

            time.sleep(10)
            garbage_labels = 'depracted'
            structure_labels = 'depracted'

            # get a batch generator and shuffler:
            batch, labels = tf.train.shuffle_batch([seq_tensor, label],
                                                   batch_size=self._opts._batchsize, # save 4 spots for the garbage
                                                   num_threads=self._opts._numthreads,
                                                   capacity=500 * self._opts._numthreads, #should this match the filesize of example files?
                                                   min_after_dequeue=50 * self._opts._numthreads,
                                                   enqueue_many=False,
                                                   )

            return batch, labels, garbage_labels, structure_labels

    def model(self, seq_input, valid_mode=False):
        """
            construct the trainings OR evaluation graph, depending on valid_mode the variable sharing is
            initialized
        """
        self.summary_collection = ['train']
        name_suffix = '_train'
        if valid_mode:
            self.summary_collection = ['valid']
            name_suffix = '_valid'

        with tf.variable_scope('model%s' % name_suffix) as vs:
            tl.layers.set_name_reuse(True)

            seq_in_layer = tl.layers.InputLayer(seq_input, name='seq_input_layer%s' % name_suffix)
########################################################################################################################
#                                                   Encoder                                                            #
########################################################################################################################
            print('[*] ENCODER')
            with tf.variable_scope('encoder') as vs:
                with tf.variable_scope('embedding') as vs:
                    embedding = tl.layers.Conv2dLayer(seq_in_layer,
                                                  act=customlayers.prelu,
                                                  shape=[20, 1, 1, 64],
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                                  W_init_args={},
                                                  b_init=tf.constant_initializer(value=0.1),
                                                  b_init_args = {},
                                                  name='1x1%s' % name_suffix)
                    embedding = tl.layers.BatchNormLayer(embedding, decay=0.9, epsilon=1e-05,
                                                   is_train=self.is_train,
                                                   name='batchnorm_layer%s' % name_suffix)
                    output_shape = embedding.outputs.get_shape().as_list()
                    embedding.outputs = tf.reshape(embedding.outputs, shape=[self._opts._batchsize, output_shape[2], output_shape[3]])
                    helpers._add_var_summary(embedding.outputs, 'conv', collection=self.summary_collection)

                resnet = customlayers.resnet_block(embedding, channels=[64, 128],  pool_dim=2, is_train=self.is_train,
                                                   name='res1')
                resnet = customlayers.resnet_block(resnet, channels=[128, 256], pool_dim=2, is_train=self.is_train,
                                                   name='res2')
                resnet = customlayers.resnet_block(resnet, channels=[256, 256], pool_dim=2, is_train=self.is_train,
                                                   name='res3')
                resnet = customlayers.resnet_block(resnet, channels=[256, 256], pool_dim=2, is_train=self.is_train,
                                                   name='res4')
                resnet = customlayers.resnet_block(resnet, channels=[256, 256], pool_dim=2, is_train=self.is_train,
                                                   name='res5')
                resnet = customlayers.resnet_block(resnet, channels=[256, 256], pool_dim=None, is_train=self.is_train,
                                                   name='res6')
                resnet = customlayers.resnet_block(resnet, channels=[256, 256], pool_dim=3, is_train=self.is_train,
                                                   name='res7')
                resnet = customlayers.resnet_block(resnet, channels=[256, 256], pool_dim=2, is_train=self.is_train,
                                                   name='res8')
                resnet = customlayers.resnet_block(resnet, channels=[256, 256], pool_dim=None, is_train=self.is_train,
                                                   name='res9')
                resnet = customlayers.resnet_block(resnet, channels=[256, 512], pool_dim=2, is_train=self.is_train,
                                                   name='res10')
                encoder = customlayers.resnet_block(resnet, channels=[512, 512], pool_dim=2, is_train=self.is_train,
                                                   name='res11')
                self.encoder = encoder #put the encoder in an attribute for easy access
                print('Final shape: ' + str(encoder.outputs.get_shape().as_list()))


########################################################################################################################
#                                                   Classifier                                                         #
########################################################################################################################
            print('[*] CLASSIFIER')
            with tf.variable_scope('classifier') as vs:
#                with tf.variable_scope('dense1') as vs:
#                    flat = tl.layers.FlattenLayer(encoder, name='flatten')
#                    fc = tl.layers.DenseLayer(flat,
#                                              n_units=512,
#                                              #n_units=1024,
#                                              act=customlayers.prelu,
#                                              name='fc'
#                                              )
#                    dropout = tl.layers.DropoutLayer(fc,
#                                                     keep=0.5,
#                                                     is_train=self.is_train,
#                                                     is_fix=True,
#                                                     name='dropout')
#                with tf.variable_scope('dense2') as vs:
#                    fc = tl.layers.DenseLayer(dropout,
#                                              n_units=1024,
#                                              act=customlayers.prelu,
#                                              name='fc'
#                                              )
#                    dropout = tl.layers.DropoutLayer(fc,
#                                                     keep=0.5,
#                                                     is_train=self.is_train,
#                                                     is_fix=True,
#                                                     name='dropout')
#                    self.graph_no_dense = dropout  # now also fc
#
#                with tf.variable_scope('outlayer_1') as vs:
#                    classifier1 = tl.layers.DenseLayer(dropout,
#                                                      n_units=self._opts._nclasses,
#                                                      act=customlayers.prelu,
#                                                      name='fc')
#
#                with tf.variable_scope('outlayer_2') as vs:
#                    classifier2 = tl.layers.DenseLayer(dropout,
#                                                       n_units=self._opts._nclasses,
#                                                       act=customlayers.prelu,
#                                                       name='fc')

                with tf.variable_scope('out1x1_1') as vs:
                    classifier1 = tl.layers.Conv1dLayer(encoder,
                                                act=customlayers.prelu,
                                                        shape=[1, 512, self._opts._nclasses],  # 32 features for each 5x5 patch
                                                stride=1,
                                                padding='SAME',
                                                W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                                W_init_args={},
                                                b_init=tf.constant_initializer(value=0.1),
                                                b_init_args={},
                                                name='1x1_layer')
                    classifier1.outputs = tf.reshape(classifier1.outputs, [self._opts._batchsize, self._opts._nclasses])

                with tf.variable_scope('out1x1_2') as vs:
                    classifier2 = tl.layers.Conv1dLayer(encoder,
                                                act=customlayers.prelu,
                                                        shape=[1, 512, self._opts._nclasses],  # 32 features for each 5x5 patch
                                                stride=1,
                                                padding='SAME',
                                                W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                                W_init_args={},
                                                b_init=tf.constant_initializer(value=0.1),
                                                b_init_args={},
                                                name='1x1_layer')
                    classifier2.outputs = tf.reshape(classifier2.outputs, [self._opts._batchsize, self._opts._nclasses])

                    # this output is of shape [batch, 1, classes]
                with tf.variable_scope('outlayer_concat') as vs:
                    classifier = customlayers.StackLayer([classifier1, classifier2], axis=-1) # along the channels
########################################################################################################################
#                                                   Garbage Detector                                                   #
########################################################################################################################
            print('[*] GARBAGE_DETECTOR')
            with tf.variable_scope('garbage_detec') as vs:
                flat = tl.layers.FlattenLayer(encoder, name='flatten')
                garbage_detector = tl.layers.DenseLayer(flat,
                                                n_units=64,  # we keep this binary.
                                                act=customlayers.prelu,
                                                name='fc')
                dropout = tl.layers.DropoutLayer(garbage_detector,
                                                keep=0.5,
                                                is_train=self.is_train,
                                                is_fix=True,
                                                name='dropout')

            with tf.variable_scope('garbage_detec2') as vs:
                garbage_detector = tl.layers.DenseLayer(dropout,
                                                n_units=2,  # we keep this binary.
                                                act=customlayers.prelu,
                                                name='fc')

            if valid_mode:
                classifier.outputs = tf.Print(classifier.outputs, [classifier.outputs.get_shape(),
                                                               classifier.outputs, classifier.outputs
                                                               ], message='outVALID') if self._opts._debug else classifier.outputs
                self.eval_graph_complete = classifier
                return classifier, garbage_detector
            else:
                classifier.outputs = tf.Print(classifier.outputs, [classifier.outputs.get_shape(),
                                                               classifier.outputs, classifier.outputs
                                                               ], message='out') if self._opts._debug else classifier.outputs
                self.graph_complete = classifier
                return classifier, garbage_detector

    def get_loss(self, raw_logits, labels, valid_mode=False):
        """ add a loss to the current graph """
        name_suffix = '_train'

        if valid_mode:
            name_suffix = '_valid'
        with tf.variable_scope('loss%s' % name_suffix) as vs:

            # take an argmax to get the channel with the larget activations in each position:
            with tf.variable_scope('raw_logits') as vs:
                if not valid_mode:
                    single_raw_logits = tf.reduce_max(raw_logits, axis=-1, keep_dims=False)
                else:
                    # take the mean in valid
                    # TODO: add variance as a measure for certainity
                    single_raw_logits = tf.reduce_mean(raw_logits, axis=-1, keep_dims=False)

            # first get the logits from the outlayer
            sigmoid_logits = tf.nn.sigmoid(single_raw_logits, name='logits')


            # positives
            positive_predictions = tf.cast(tf.greater(sigmoid_logits, 0.5), dtype=tf.float32)
            true_positive_predictions = tf.multiply(positive_predictions, labels)
            # negatives
            negative_predictions = tf.cast(tf.less(sigmoid_logits, 0.5), dtype=tf.float32)
            negative_labels = tf.cast(tf.equal(labels, 0), dtype=tf.float32)
            true_negative_predictions = tf.multiply(negative_predictions, negative_labels)
            false_negative_predictions = tf.multiply(negative_labels, labels)
            false_positive_predictions = tf.multiply(positive_predictions, negative_labels)
            # stats
            nr_pred_positives = tf.reduce_sum(positive_predictions)
            nr_true_positives = tf.reduce_sum(true_positive_predictions)
            nr_true_negatives = tf.reduce_sum(true_negative_predictions)
            nr_false_positives = tf.reduce_sum(false_positive_predictions)
            nr_false_negatives = tf.reduce_sum(false_negative_predictions)
            tpr = tf.divide(nr_true_positives, tf.reduce_sum(labels))
            fdr = tf.divide(nr_false_positives, nr_pred_positives)
            fpr = tf.divide(nr_false_positives, tf.reduce_sum(negative_labels))
            tnr = tf.divide(nr_true_negatives, tf.reduce_sum(negative_labels))

            # accuracy
            f1_score = tf.divide(nr_true_positives*2,
                                 tf.add(tf.add(2*nr_true_positives, nr_false_negatives), nr_false_positives))

            tf.summary.scalar('TPR', tpr, collections=self.summary_collection)
            tf.summary.scalar('FPR', fpr, collections=self.summary_collection)
            tf.summary.scalar('FDR', fdr, collections=self.summary_collection)
            tf.summary.scalar('TNR', tnr, collections=self.summary_collection)
            tf.summary.scalar('F1', f1_score, collections=self.summary_collection)
            tf.summary.scalar('avg_pred_positives', tf.divide(nr_pred_positives, self._opts._batchsize), collections=self.summary_collection)
            tf.summary.scalar('avg_true_positives', tf.divide(nr_true_positives, self._opts._batchsize), collections=self.summary_collection)

            # get the FALSE POSITIVE LOSS:
            fp_loss = tf.divide(nr_false_positives, self._opts._batchsize)
            tf.summary.scalar('fp_loss', fp_loss, collections=self.summary_collection)
            # get the TRUE POSITIVE LOSS
            tp_loss = tf.subtract(1.0, tpr)

            #get the balanced cross entropy:
            # class_sizes = np.asfarray(
            #     [self.batchgen.class_dict[key]['size'] if self.batchgen.class_dict[key]['size'] <= 1000 else 1000
            #      for key in self.batchgen.class_dict.keys()])
            class_sizes = np.asfarray(
                [self.batchgen.class_dict[key]['size'] for key in self.batchgen.class_dict.keys()])
            mean_class_size = np.mean(class_sizes)
            self.pos_weight = mean_class_size / class_sizes
            # config.maxClassInbalance prevents too large effective learning rates (i.e. too large gradients)
            assert self._opts._maxclassinbalance >= 1.0

            self.pos_weight = np.maximum(1.0, np.minimum(self._opts._maxclassinbalance, self.pos_weight))
            self.pos_weight = self.pos_weight.astype(np.float32)
            print(self.pos_weight)
            self.log_file.write(str(self.pos_weight))

            # tile the pos weigths:
            pos_weights = tf.reshape(tf.tile(self.pos_weight, multiples=[self._opts._batchsize]),
                                     [self._opts._batchsize, self._opts._nclasses])
            assert pos_weights.get_shape().as_list() == [self._opts._batchsize, self._opts._nclasses]

            # get the FOCAL LOSS WIT
            # focal_loss = customlayers.focal_loss(logits=sigmoid_logits, labels=labels, gamma=2,
            #                                      pos_weights=pos_weights, clips=[0.001, 1000.])
            focal_loss = customlayers.focal_lossIII(prediction_tensor=single_raw_logits, target_tensor=labels,
                                                    weights=self.pos_weight,
                                                    gamma=2., epsilon=0.00001)
            fl_sum = tf.reduce_sum(focal_loss, name='focal_loss_sum')
            fl_mean = tf.reduce_mean(focal_loss, name='focal_loss_mean')

            ce_loss = tf.nn.weighted_cross_entropy_with_logits(logits=single_raw_logits, targets=labels,
                                                               pos_weight=self.pos_weight)
            ce_mean = tf.reduce_mean(ce_loss, name='celoss_mean')

            #get the l2 loss on weigths of conv layers and dense layers
            l2_loss = 0
            for w in tl.layers.get_variables_with_name('W_conv1d', train_only=True, printable=False):
                l2_loss += tf.contrib.layers.l2_regularizer(1e-4)(w)
            for w in tl.layers.get_variables_with_name('W_conv2d', train_only=True, printable=False):
                l2_loss += tf.contrib.layers.l2_regularizer(1e-4)(w)
            for w in tl.layers.get_variables_with_name('W', train_only=True, printable=False):
                l2_loss += tf.contrib.layers.l2_regularizer(1e-4)(w)

            """
            We add up all loss functions
            """
            #loss = fl_mean + l2_loss
            loss = ce_mean + l2_loss

            tf.summary.scalar('loss_total', loss, collections=self.summary_collection)
            tf.summary.scalar('loss_l2', l2_loss, collections=self.summary_collection)
            tf.summary.scalar('loss_1-tp', tp_loss, collections=self.summary_collection)
            #tf.summary.scalar('loss_ssce', ce_mean, collections=self.summary_collection)
            tf.summary.scalar('loss_focal_mean', fl_mean, collections=self.summary_collection)
            tf.summary.scalar('loss_focal_sum', fl_sum, collections=self.summary_collection)
            tf.summary.scalar('loss_CE', ce_mean, collections=self.summary_collection)

            return loss, f1_score

    def get_opt(self, loss, vars=[], adam=False):
        """ add an optimizer to the current graph """

        # only optimize the dense layers:
        # train_params = tl.layers.get_variables_with_name('dense', train_only=True, printable=True)

        # only optimize the last 5 layers:
        # train_params = tl.layers.get_variables_with_name('train', train_only=True, printable=True)[-8:]

        if adam:
            if vars:
                opt = tf.train.AdamOptimizer(learning_rate=self._opts._learningrate,
                                             beta1=0.9, beta2=0.999,
                                             epsilon=self._opts._epsilon,
                                             use_locking=False, name='Adam').minimize(loss, var_list=vars)

            else:
                opt = tf.train.AdamOptimizer(learning_rate=self._opts._learningrate, beta1=0.9, beta2=0.999,
                                             epsilon=self._opts._epsilon, use_locking=False, name='Adam').minimize(loss)
            #var_list=train_params)
        else:
            if vars:
                opt = tf.train.AdagradOptimizer(learning_rate=self._opts._learningrate,
                                                initial_accumulator_value=0.1,
                                                use_locking=False, name='Adagrad').minimize(loss, var_list=vars)
            #var_list=train_params)
            else:
                opt = tf.train.AdagradOptimizer(learning_rate=self._opts._learningrate,
                                                initial_accumulator_value=0.1,
                                                use_locking=False, name='Adagrad').minimize(loss)
        return opt

    def feed_dict(self):
        """
        Currently depracted as we run the dropout layers in "is_fix" mode.
        :return: {}
        """
        if self.is_train:
            feed_dict = {
                # self.training_bool: self.is_train, #the training bool decides which queue we use and if droppout is applied or not.
            }
            #feed_dict.update(self.graph_complete.all_drop) #switch dropout ON
        else:
            #dp_dict = tl.utils.dict_to_one(self.graph_complete.all_drop)
            feed_dict = {
                # self.training_bool: self.is_train, #the training bool decides which queue we use and if droppout is applied or not.
            }
            #feed_dict.update(dp_dict) #switch dropout OFF

        return {}

    def initialize_helpers(self):
        """
        initialize the model. Means: call all graph constructing ops
        """
        if self._opts._allowsoftplacement == 'True':
            config = tf.ConfigProto(allow_soft_placement=True)
        else:
            config = tf.ConfigProto(allow_soft_placement=False)

        # allow growth to surveil the consumed GPU memory
        config.gpu_options.allow_growth = True
        # open a session:
        self.session = tf.Session(config=config)

        self.log_file.write('Initialized Batch_Generator with MODE: %s\n' % self._opts._batchgenmode)
        self.batchgen = helpers.BatchGenerator(self._opts, self._kmer2vec_embedding, self._kmer2id)

        self.log_file.write('Initialized ROC_tracker\n')
        self.ROCtracker = helpers.RocTracker(self._opts)

    def guarantee_initialized_variables(self, session, list_of_variables=None):
        if list_of_variables is None:
            list_of_variables = tf.all_variables()
        uninitialized_variables = list(tf.get_variable(name) for name in
                                       session.run(tf.report_uninitialized_variables(list_of_variables)))
        session.run(tf.initialize_variables(uninitialized_variables))
        return uninitialized_variables

    def twerkformePLZZZZZZ(self, restore_whole=True):
        """
        Run training and validation ops.
        :return:
        """
        # get a graph
        train_graph = tf.Graph()
        with train_graph.as_default():
            self.initialize_helpers()

            # define the filenames for validation and training:
            train_filenames = glob.glob(os.path.join(self._opts._batchesdir,
                                                     '*train_batch_%s_*' % str(self._opts._windowlength)))
            print(train_filenames[0])

            with tf.device('/gpu:{}'.format(self._opts._gpu)):
                # graph for training:
                train_batch, train_labels, _, _ = self.input_pipeline(train_filenames, valid_mode=False)
                classifier, _ = self.model(train_batch, valid_mode=False)

                train_raw_logits = classifier.outputs

                train_loss, train_acc = self.get_loss(raw_logits=train_raw_logits, labels=train_labels,
                                                      valid_mode=False)

                #train_vars = [p for p in classifier.all_params if 'classifier' in p.name]

                opt = self.get_opt(train_loss, vars=[])
                #opt = self.get_opt(train_loss, vars=train_vars)

                self.session.run(tf.global_variables_initializer())
                self.session.run(tf.local_variables_initializer())

                # restore the model for training
                if self._opts._restore:
                    if restore_whole:
                        classifier = self.load_model_weights(classifier, session=self.session,
                                                                name='Classifier') #THIS RUNS THE SESSION INTERNALLY
                    else:
                        classifier = self.load_conv_weights_npz(classifier, session=self.session,
                                                            name='Classifier') #THIS RUNS THE SESSION INTERNALLY
                        # print(classifier)

                train_summaries = tf.summary.merge_all(key='train')

            self.saver = tf.train.Saver()

            #get time and stats_collector
            start_time = time.time()
            stats_collector = []

            # get the queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=self.session)

            # time.sleep(30)

            # get log file and writers:
            self.train_writer = tf.summary.FileWriter(self._opts._summariesdir + '/train', self.session.graph)
            self.eval_writer = tf.summary.FileWriter(self._opts._summariesdir + '/valid')

            self.log_file.write('Starting TRAINING...\n')

            for step in range(1, self._opts._numsteps):

                if step % 200 == 0:
                    _, loss, out, summary, acc, labels = self.session.run([opt,
                                                                           train_loss,
                                                                           train_raw_logits,
                                                                           train_summaries,
                                                                           train_acc,
                                                                           train_labels],
                                                                          feed_dict=self.feed_dict())
                    stats_collector.append((loss, acc))
                    self.train_writer.add_summary(summary, step)
                    losses = [stat[0] for stat in stats_collector]
                    accuracies = [stat[1] for stat in stats_collector]
                    av_loss = sum(losses) / len(losses)
                    av_accuracy = sum(accuracies) / len(accuracies)
                    stats_collector = []

                    self.log_file.write('Step %d: Av.loss = %.2f (%.3f sec)\n' % (step, av_loss,
                                                                                  time.time() - start_time))
                    #print(self.log_file)
                    self.log_file.write('Step %d: Av.accuracy = %.2f (%.3f sec)\n' % (step, av_accuracy,
                                                                                      time.time() - start_time))
                    self.log_file.flush()

                    #predicted_labels = np.argmax(out, axis=1)
                    #true_labels = np.argmax(labels, axis=1)

                    #self.log_file.write('predicted:    ' + str(predicted_labels) + '\n' )
                    #self.log_file.write('true:         ' + str(true_labels) + '\n' )

                if step % 2000 == 0:
                    # save the model in nais model snapshots like in model book
                    self.save_model(classifier, self.session, step=step, name='Classifier')
                    print('DONE')
                    self.eval_while_train(step, 2000)
                    #if step == 1000:
                    #self.valid_graph_initialized = True

                else:
                    _, loss, acc, labels = self.session.run([opt, train_loss, train_acc, train_labels],
                                                            feed_dict=self.feed_dict())
                    stats_collector.append((loss, acc))

                    # print labels for debugging
                    #print(np.argmax(labels, axis=1))

            # gracefully shut down the queue
            coord.request_stop()
            coord.join(threads)

    def eval_while_train(self, step=1, eval_steps=200):
        """
        restore the last checkpoint in the config_dict and reload the model weights IN A DIFFERENT SESSION TO THE TRAIN SESSION.
        The validation is performed on the whole validation set. metrics (TP, FP, etc) are collected to calculate the ROC curves
        :return:
        """
        # get a graph:
        eval_graph = tf.Graph()
        self.is_train = False
        # bs_train = self._opts._batchsize
        # self._opts._batchsize = 1

        valid_filenames = glob.glob(os.path.join(self._opts._batchesdir,
                                                 '*valid_batch_%s_*' % str(self._opts._windowlength)))
        print('Found %d validation files.' % len(valid_filenames))

        # get a new session:
        if self._opts._allowsoftplacement == 'True':
            config = tf.ConfigProto(allow_soft_placement=True)
        else:
            config = tf.ConfigProto(allow_soft_placement=False)
        # allow growth to survey the consumed GPU memory
        config.gpu_options.allow_growth=True
        with eval_graph.as_default():
            with tf.device('/gpu:{}'.format(self._opts._gpu)): # TODO fix this as gpu0 is always requested.
                with tf.Session(config=config) as sess:
                    if self.valid_graph_initialized: # first time we validate TODO: implement this smarter...
                        tl.layers.set_name_reuse(enable=True)
                    # graph for evaluation:
                    valid_batch, valid_labels, _, _ = self.input_pipeline(valid_filenames, valid_mode=True)
                    infer_classifier, _ = self.model(valid_batch, valid_mode=True)

                    labelss = tf.argmax(valid_labels, axis=1)
                    assert labelss.get_shape().as_list() == [self._opts._batchsize]


                    valid_raw_logits = infer_classifier.outputs

                    valid_sigmoid_logits = tf.sigmoid(valid_raw_logits, name='Sigmoid_logits')
                    # reduce mean:
                    valid_sigmoid_logits = tf.reduce_mean(valid_sigmoid_logits, axis=-1, keep_dims=False)


                    valid_loss, valid_acc = self.get_loss(raw_logits=valid_raw_logits, labels=valid_labels,
                                                          valid_mode=True)

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())

                    # restore the model weights
                    self.load_complete_model_eval(infer_classifier, sess, name='Classifier')

                    valid_summaries = tf.summary.merge_all(key='valid')

                    eval_coord = tf.train.Coordinator()
                    eval_threads = tf.train.start_queue_runners(coord=eval_coord, sess=sess)

                    average_acc = []

                    step = step
                    try:
                        while not eval_coord.should_stop():
                            step += 1
                            # control everything with the coordinator
                            if eval_coord.should_stop():
                                break

                            summary, loss, outlayer, \
                            acc, labels, sigmoid_logits, _ = sess.run([valid_summaries,
                                                                    valid_loss,
                                                                    valid_raw_logits,
                                                                    valid_acc,
                                                                    valid_labels,
                                                                    valid_sigmoid_logits,
                                                                    labelss
                                                                    ],
                                                                   feed_dict=self.feed_dict()
                                                                   )
                            self.eval_writer.add_summary(summary, step)
                            predicted_labels = np.argmax(outlayer, axis=1)

                            true_labels = np.argmax(labels, axis=1)

                            # pass the predictions to the ROC tracker:
                            self.ROCtracker.update(sigmoid_logits=sigmoid_logits, true_labels=labels)

                            average_acc.append(acc)

                    except tf.errors.OutOfRangeError:
                        average_acc = sum(average_acc)/len(average_acc)
                        self.log_file.write('[*] Finished validation after %s steps'
                                            ' with av.acc of %s' % (str(step), str(average_acc)))
                        self.log_file.flush()
                    finally:
                        # when done ask the threads to stop
                        eval_coord.request_stop()

                    eval_coord.join(eval_threads)
                    sess.close()

        self.ROCtracker.calc_and_save(self.log_file)

        # delete all layers from the set_keep dict
        #tl.layers.set_keep['_layers_name_list'] = [l for l in tl.layers.set_keep['_layers_name_list'] if not 'valid' in l]
        self.is_train = True
        # self._opts._batchsize = bs_train

    def infer_single(self, seq):
        """
        Initialize the model in inference mode. Model needs to be restored to use this op.
        """
        with tf.Graph().as_default():
            self.initialize_helpers()

            # graph for inference:
            self.is_train = False
            self._opts._batchsize = 1
            input_seq_node = tf.placeholder([self._opts._batchsize, self._opts._depth, self._opts._windowlength, 1])
            inference_net, _ = self.model(input_seq_node, valid_mode=False)
            inf_sigmoid_logits = tf.nn.sigmoid(inference_net.outputs)

            self.session.run(tf.global_variables_initializer())
            # restore all the weights
            self.load_model_weights(network=inference_net, session=self.session)
            #self.load_complete_model_eval(session=self.session, path=self._opts._restorepath)

            encoded_seq = self.batchgen._encode_single_seq(seq)
            # expand it to batch dim and channels dim:
            assert len(encoded_seq.shape) == 2
            encoded_seq = np.reshape(encoded_seq, [1, encoded_seq.shape[0], encoded_seq.shape[1], 1])

            # run the session and retrieve the predicted classes
            logits = self.session.run(inf_sigmoid_logits, feed_dict={input_seq_node: encoded_seq})
            # logits is a 2D array of shape [1, nclasses]
            thresholded_logits = np.where(logits >= 0.5)
            # get the IDs of the classes:
            predicted_IDs = thresholded_logits[1].as_list()

            # get the corresponding classes from the EC_file:
            predicted_classes = [self.batchgen.id_to_class[_id] for _id in predicted_IDs]
            self.log_file.write('Predicted classes:\n')
            for i, _class in enumerate(predicted_classes):
                self.log_file.write('%s:\t%f\n' % (_class, logits[0, predicted_IDs[i]]))

    def infer_from_example(self):
        """
        Infer all sequences in the VALID set and calculate baseline metrics on max. Scores for GAIA.
        """
        # TODO: to be written
        pass

    def get_saliency_map(self):
        """
        We walk over the sequence and try to fit a saliency mask, to see where important sequence parts are
        :return:
        """
        # TODO: To be written
        pass

    def generate_embedding(self, embedding_dims=1024, reduce_dims=False):
        """
        This function generates a Protein embedding based on the validation dataset.
        In detail it builds an inference graph, loads a pretrained model and stores the features of the
        last dense layer in a embdding matrix.
        The protein-name of each sequence passed through the net is stored in a dictionary and converted to
        integers , which allows later lookup.
        """

        # get an embedding dir:
        embedding_dir = os.path.join(self._opts._restorepath, 'embedding%d' % embedding_dims)
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        with tf.Graph().as_default():
            # we build the graph for inference, without loss ops and optimizers
            self.is_train = False
            self._opts._batchsize = 50 # TODO edit this here
            self.initialize_helpers()

            # get the line nr of the valid_data,
            # add the batch percentage and determine the number of batches in an epoch

            input_seq_node = tf.placeholder(dtype=tf.float32,
                                            shape=[self._opts._batchsize, self._opts._depth, self._opts._windowlength, 1])

            inference_net, _, _ = self.model(input_seq_node, valid_mode=False)

            # load the pretrained model
            self.session.run(tf.global_variables_initializer())
            self.load_model_weights(inference_net, session=self.session, name='Classifier')

            # initialize the embedding with UNK token, the dict is updated automatically for UNK
            self.embedding = np.zeros([1, 1024])

            for i in range(self.batchgen.eval_batch_nr): # this is calculated when we initialize batchgen

                batch = self.batchgen.generate_valid_batch(include_garbage=True)
                # run the session and retrieve the embedded_batch
                embed_batch = self.session.run(self.graph_no_dense.outputs, feed_dict={input_seq_node: batch})

                # Add the batch to the complete embedding:
                self.embedding = np.concatenate([self.embedding, embed_batch], axis=0)

            # get the total number of sequeces we embedded:
            embedded_seqs = self.embedding.shape[0]
            original_features = self.embedding.shape[1] # TODO we don't really need this

        if reduce_dims:
            # now perform a dimensionality reduction (PCA) to get the n most important features:
            pca = decomposition.PCA(n_components=embedding_dims)
            pca.fit(self.embedding)
            fitted_embedding = pca.transform(self.embedding)

        else:
            assert embedding_dims == 1024
            fitted_embedding = self.embedding

        tf.reset_default_graph()

        g = tf.Graph()

        with g.as_default():
            # not get a different session and feed the damn np.Tensor into a nice tf.Tensor with tensorboard support.
            with tf.Session() as embedding_sess:
                embedding_node = tf.placeholder(dtype=tf.float32, shape=[embedded_seqs,
                                                                         embedding_dims])

                # do the tensorboard stuff
                # Do a mock op for the checkpoint:
                embedding_Var = tf.get_variable('ProteinEmbedding', [embedded_seqs, embedding_dims])
                #embedding_Tensor = tf.multiply(embedding_node, mockop, name='ProteinEmbedding')
                embedding_Var = tf.assign(embedding_Var, embedding_node)

                # now do a mockOP:
                mockOP = tf.multiply(embedding_Var, tf.ones([embedded_seqs, embedding_dims]))

                # do the damn ops for tensorboard:
                config = projector.ProjectorConfig()
                embedding = config.embeddings.add()
                embedding.tensor_name = embedding_Var.name
                # Link this tensor to its metadata file (e.g. labels).
                embedding.metadata_path = os.path.join(embedding_dir, 'metadata.tsv')

                summary_writer = tf.summary.FileWriter(embedding_dir)
                projector.visualize_embeddings(summary_writer, config)

                # initialize Vars and get a Saver and a writer
                embedding_sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()

                # run it:
                _ = embedding_sess.run(mockOP,
                                       feed_dict={embedding_node: fitted_embedding}) # TODO edit here

                saver.save(embedding_sess, os.path.join(embedding_dir, "ProteinEmbedding.ckpt"), 1)

            # now save the metadata as tsv:
            with open(os.path.join(embedding_dir, 'metadata.tsv'), "w") as out_fobj:
                for key in self.batchgen.embedding_dict:
                    line = [','.join(self.batchgen.embedding_dict[key]['labels']), key,]
                    line = '\t'.join(line)
                    line += '\n'
                    out_fobj.write(line)

    def init_for_machine_infer(self, batchsize=1):
        """
        Initialize the model in inference mode. Model needs to be restored to use this op.
        :param batchsize: int
        Define size of batches used in machine_infer
        """
        with tf.Graph().as_default():
            self.initialize_helpers()

            # graph for inference:
            self.is_train = False
            self._opts._batchsize = batchsize

            self.input_seq_node = tf.placeholder(
                shape=[self._opts._batchsize, self._opts._depth, self._opts._windowlength, 1],
                dtype=tf.float32)
            inference_net, inference_garbage_detec = self.model(self.input_seq_node, valid_mode=False)

            double_sigmoid_logits = tf.nn.sigmoid(inference_net.outputs)

            self.mean_inf_sigmoid_logits, self.var_inf_sigmoid_logits = tf.nn.moments(double_sigmoid_logits,
                                                                            axes=[2])

            self.inf_garbage_logit = tf.nn.softmax(inference_garbage_detec.outputs, dim=1)

            self.session.run(tf.global_variables_initializer())
            # restore all the weight
            _ = self.load_model_weights(inference_net, session=self.session,
                                        name='Classifier')  # THIS RUNS THE SESSION INTERNALLY

    def machine_infer(self, seq):
        """
        Infers on one batch and calculates logits and their variance
        :return: ndarray([self.opts._batchsize, self.opts._windowlength],
                ndarray([self.opts._batchsize, self.opts._windowlength]
            Two numpy arrays, that hold the logits of each class for each input sequence
            and the variance of the logits of each class for each input sequence
        :param seq: ndarray([self.opts._batchsize, self.opts._depth, self.opts._windowlength])
            A numpy array that contains a batch of one-hot encoded sequences that are zeropadded to self.opots._windowlength
        """

        seq = np.expand_dims(seq, -1)

        # run the session and retrieve the predicted classes
        class_logits, class_variance \
            = self.session.run([self.mean_inf_sigmoid_logits, self.var_inf_sigmoid_logits],
                                     feed_dict={self.input_seq_node: seq})

        return class_logits, class_variance