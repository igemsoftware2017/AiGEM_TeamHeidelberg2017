import helpers
import customlayers
import tensorlayer as tl
import tensorflow as tf
import time, os, glob
import string
import json
import wget
import numpy as np
import re
from tensorflow.contrib.tensorboard.plugins import projector
from collections import OrderedDict
from sklearn import decomposition
np.set_printoptions(threshold=np.inf) #for debug


class DeeProtein():
    """The main class for the DeeProtein module.

    This class is used to set up the DeeProtein module. It allows training, inference and validation of models.

    Attributes:
      _opts: The OptionHandler of holding the flags specified in the config_dict.
      reader: A `tf.RecordsReader` used for the input pipeline.
      saver: A `tf.train.Saver` object used to save the model to checkpoints.
      opt: A `tf.train.Optimizer`, the optimizer of the model.
      is_train: A `bool` defining the state of the model true: train, false: validation.
      batchgen: A `helpers.BatchGenerator`, holds the class_dict.
      ROCtracker: A `helpers.ROCtracker` that calculates the metrics for the calidation.
    """
    def __init__(self, optionhandler, inference=False):
        self._opts = optionhandler
        self._opts.write_dict()
        self.reader = ''
        self.saver = ''
        self.opt = ''
        self.is_train = True
        self.batchgen = ''
        self.ROCtracker = ''
        if not inference:
            self.log_file = open(self._opts._summariesdir + '/model.log', "w", 1)
        else:
            self.log_file = open(self._opts._summariesdir + '/inference.log', "w", 1)
        self.valid_graph_initialized = False

    def restore_model_from_checkpoint(self, ckptpath, session):
        """Restores the model from checkpoint in `Session`. This function is deprected, please
           use the regular restore ops from npz-files.

        Args:
          ckptpath: A `str`. The path to the checkpoint file.
          session: The `tf.session` in which to restore the model.
        """
        # Restore variables from disk.
        self.log_file.write('[*] Loading checkpoints from %s\n' % ckptpath)
        self.saver.restore(session, ckptpath)
        self.log_file.write('[*] Restored checkpoints!\n')

    def save_model(self, network, session, step, name='DeeProtein'):
        """Saves the model into .npz and ckpt files.
        Save the dataset to a file.npz so the model can be reloaded. The model is saved in the
        checkpoints folder under the directory specified in the config dict under the
        summaries_dir key.

        Args:
          network: `tl.layerobj` holding the network.
          session: `tf.session` object from which to save the model.
          step: `int32` the global step of the training process.
          name: `str` The name of the network-part that is to save.
        """
        # save model as dict:
        param_save_dir = os.path.join(self._opts._summariesdir,
                                      'checkpoint_saves/')
        # everything but the outlayers
        conv_vars = [var for var in network.all_params
                     if 'dense' and 'outlayer' not in var.name]

        if not os.path.exists(param_save_dir):
            os.makedirs(param_save_dir)
        if conv_vars:
            tl.files.save_npz_dict(conv_vars,
                                   name=os.path.join(param_save_dir,
                                                     '%s_conv_part.npz' % name),
                                   sess=session)
        tl.files.save_npz_dict(network.all_params,
                               name=os.path.join(param_save_dir,
                                                 '%s_complete.npz' % name),
                               sess=session)

        # save also as checkpoint
        ckpt_file_path = os.path.join(param_save_dir, '%s.ckpt' % name)
        self.saver.save(session, ckpt_file_path, global_step=step)

    def save_params(self, network, session, ignore=None):
        """Save the network into parameter specific files.
        This function saves all parameters of the inference network and stores them as single Parameter files.

        Args:
          network: `tl.layer` Object holding the network.
          session: `tf.Session` the tensorflow session of whcih to save the model.
        """
        FILENAME_CHARS = string.ascii_letters + string.digits + '_'

        def _var_name_to_filename(var_name):
            chars = []
            for c in var_name:
                if c in FILENAME_CHARS:
                    chars.append(c)
                elif c == '/':
                    chars.append('_')
            return ''.join(chars)

        param_save_dir = os.path.join(os.path.join(self._opts._summariesdir, 'checkpoints'), 'inference_params')
        if not os.path.exists(param_save_dir):
            os.makedirs(param_save_dir)

        if ignore:
            remove_vars_compiled_re = re.compile(ignore)
        else:
            remove_vars_compiled_re = None

        manifest = OrderedDict()
        for p in network.all_params:
            name = p.name.rstrip(':0')
            if (ignore and
                    re.match(remove_vars_compiled_re, name)) or name == 'global_step':
                continue
            var_filename = _var_name_to_filename(name)
            manifest[name] = {'filename': var_filename, 'shape': p.get_shape().as_list()}
            with open(os.path.join(param_save_dir, var_filename), 'wb') as f:
                f.write(p.eval(session=session).tobytes())

        manifest_fpath = os.path.join(param_save_dir, 'manifest.json')
        with open(manifest_fpath, 'w') as f:
            f.write(json.dumps(manifest, indent=2, sort_keys=False))
        self.log_file.write('[*] Saved inference parameters!\n')

    def download_weights(self):
        """Download the weigths to restore DeeProtein from the 2017 iGEM wiki.

        Returns:
          A path to the dir containing the downloaded weigths.
        """
        url = 'https://zenodo.org/record/1035806/files/DeeProtein_weigths_ResNet30_886.tar.gz'
        curr_wd = os.getcwd()
        if not os.path.exists(self._opts._restorepath):
            os.mkdir(self._opts._restorepath)
        os.chdir(self._opts._restorepath)
        zip = wget.download(url)
        helpers.untar(zip)
        os.chdir(curr_wd)
        return os.path.join(self._opts._restorepath, 'DeeProtein_weights/')

    def load_conv_weights_npz(self, network, session, name='DeeProtein'):
        """Loads the model up to the last convolutional layer.
        Load the weights for the convolutional layers from a pretrained model.
        Automatically uses the path specified in the config dict under restore_path.

        Args:
          network: `tl.layer` Object holding the network.
          session: `tf.Session` the tensorflow session of whcih to save the model.
          name: `str`, name for the currect network to load. Although optional if multiple
            models are restored, the files are identified by name (optional).
        Returns:
          A tl.Layer object of same size as input, holding the updated weights.
        """
        # check if filepath exists:
        file = os.path.join(self._opts._restorepath, '%s_conv_part.npz' % name)
        self.log_file.write('[*] Loading %s\n' % file)
        if not tl.files.file_exists(file):
            self.log_file.write('[*] Loading %s FAILED. File not found.\n' % file)
            self.log_file.write('Trying to download weights from iGEM-HD-2017.\n')
            weights_dir = self.download_weights()
            file = os.path.join(weights_dir, '%s_conv_part.npz' % name)
            if not tl.files.file_exists(file):
                self.log_file.write('[*] Download weights from iGEM-HD-2017 FAILED. ABORTING.\n')
                exit(-1)
            else:
                self.log_file.write('Download successful.\n')
                pass
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

    def load_model_weights(self, network, session, name='DeeProtein'):
        """Load the weights for the convolutional layers from a pretrained model.
        If include outlayer is set to True, the outlayers are restored as well,
        otherwise the network is restored without outlayers.

        Args:
          network: `tl.layer` Object holding the network.
          session: `tf.Session` the tensorflow session of whcih to save the model.
          name: `str`, name for the currect network to load. Although optional if multiple
            models are restored, the files are identified by name (optional).
        Returns:
          A tl.Layer object of same size as input, holding the updated weights.
        """
        # check if filepath exists:
        file = os.path.join(self._opts._restorepath, '%s_complete.npz' % name)
        if not tl.files.file_exists(file):
            self.log_file.write('[*] Loading %s FAILED. File not found.\n' % file)
            if self._opts._nclasses == 886:
                self.log_file.write('[*] Suitable weigths found on iGEM-Servers.\n')
                self.log_file.write('Trying to download weights from iGEM-HD-2017.\n')
                weights_dir = self.download_weights()
                file = os.path.join(weights_dir, '%s_conv_part.npz' % name)
                if not tl.files.file_exists(file):
                    self.log_file.write('[*] Download weights from iGEM-HD-2017 FAILED. ABORTING.\n')
                    exit(-1)
                else:
                    self.log_file.write('Download successful.\n')
                    pass
            else:
                self.log_file.write('[*] No suitable weights on Servers. ABORTING.\n')
                exit(-1)

        # custom load_ckpt op:
        d = np.load(file)
        params = [val[1] for val in sorted(d.items(), key=lambda tup: int(tup[0]))]
        tl.files.assign_params(session, params, network)
        self.log_file.write('[*] Restored model weights!\n')
        print('[*] Restored model weights!\n')
        return network

    def load_complete_model_eval(self, network, session, name='DeeProtein'):
        """Restores the complete model from its latest save (.npz) for evaluation.
        This function is to be used in the evaluation mode, not to restore pretrained models.

        Args:
          network: `tl.layer` Object holding the network.
          session: `tf.Session` the tensorflow session of whcih to save the model.
          name: `str`, name for the currect network to load. Although optional if multiple
            models are restored, the files are identified by name (optional).
        Returns:
          A `tl.Layer object` of same size as input, holding the updated weights.
        """
        # the directory where we saved our checkpoints:
        file = os.path.join(self._opts._summariesdir,
                            'checkpoint_saves/%s_complete.npz' % name)
        # check if filepath exists:
        if tl.files.file_exists(file):
            # custom load_ckpt op:
            d = np.load(file)
            params = [val[1] for val in sorted(d.items(), key=lambda tup: int(tup[0]))]
            tl.files.assign_params(session, params, network)
            self.log_file.write('[*] Restored model for inference!\n')
            return network
        else:
            self.log_file.write('[*] Loading %s FAILED. File not found.\n' % file)
            return False

    def check_data(self, tfrecords_filename):
        """Checks a specified tf.Records file for coreect dataformat.
        Check if the data format in the example files is correct. Prints the shape of the data
        stored in a tf.Records file.

        Args
          tfrecords_filename: `str`, the path to the `tf.records` file to check.
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
        """Check the dataformat and content of a tf.records file.
        This function performs a comprehensive check up on the shapes and also on the
        values of the data stored in a tf.Records file.

        Args
          tfrecords_filename: `str`, the path to the `tf.records` file to check.
          valid_mode: `bool`, holds True if model is in valid mode (defaults to True).
        """
        filename_queue = tf.train.string_input_producer(file_paths,
                                                        num_epochs=None,
                                                        shuffle=False, seed=None,
                                                        capacity=10, shared_name=None,
                                                        name='fileQueue', cancel_op=None)
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
        """Set up an input pipeline for reading tf.records files.
        Construct an input pipeline for training or validation, depending on the passed
        filepaths.
        Note: The validation mode is only important at graph construction time.

        Args:
          file_paths: `str` TFrecords file paths.
          valid_mode: `bool`, FALSE: training, TRUE: validation mode
        Returns:
          batch: A `Tensor` holding the batch of one-hot encoded sequences
            in the format [b, 20, windowlength, 1].
          labels: A `Tensor` holding the batch of one-hot encoded labels
            in the format [b, n_classes, windowlength, 1].
          garbage_labels: Depracted.
          structure_label: Depracted.
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
            if self._opts._batchgenmode.startswith('one_hot'):
                seq_tensor = tf.cast(tf.decode_raw(features['seq_raw'], tf.float64),
                                     tf.float32)
            label = tf.cast(tf.decode_raw(features['label_raw'], tf.float64), tf.float32)
            windowlength = tf.cast(features['windowlength'], tf.int32)
            depth = tf.cast(features['depth'], tf.int32)
            n_classes = tf.cast(features['label_classes'], tf.int32)
            seq_shape = tf.stack([depth, windowlength])
            label_shape = [n_classes]

            seq_tensor = tf.expand_dims(tf.reshape(seq_tensor, seq_shape), -1)

            if self._opts._batchgenmode.startswith('one_hot'):
                seq_tensor.set_shape([self._opts._depth, self._opts._windowlength, 1])
            elif self._opts._batchgenmode.startswith('embed'):
                seq_tensor.set_shape([self._opts._embeddingdim, self._opts._windowlength, 1])
            label.set_shape([self._opts._nclasses])

            time.sleep(10)
            garbage_labels = 'depracted'
            structure_labels = 'depracted'

            # get a batch generator and shuffler:
            batch, labels = \
                            tf.train.shuffle_batch([seq_tensor, label],
                                        batch_size=self._opts._batchsize,
                                        # save 4 spots for the garbage
                                        num_threads=self._opts._numthreads,
                                        capacity=500 * self._opts._numthreads,
                                        min_after_dequeue=50 * self._opts._numthreads,
                                        enqueue_many=False)

            return batch, labels, garbage_labels, structure_labels

    def model(self, seq_input, valid_mode=False):
        """Build the graph for the model.
        Constructs the trainings or evaluation graph, depending on valid_mode.

        Args:
          seq_input: A `Tensor` holding the batch of one-hot encoded sequences of
            shape [b, 20, windowlength, 1].
          valid_mode: `bool`, FALSE: training, TRUE: validation mode

        Returns:
          A `Tensor` holding the raw activations of the model. Of shape [b, n_classes, 2]
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
                    embedding.outputs = tf.reshape(embedding.outputs,
                                                   shape=[self._opts._batchsize,
                                                          output_shape[2],
                                                          output_shape[3]])
                    helpers._add_var_summary(embedding.outputs,
                                             'conv', collection=self.summary_collection)

                resnet = customlayers.resnet_block(embedding, channels=[64, 128],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res1', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[128, 256],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res2', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res3', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res4', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res5', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=3, is_train=self.is_train,
                                                   name='res6', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res7', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res8', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res9', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res10', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res11', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res12', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res13', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res14', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res15', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res16', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res17', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res18', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res19', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res20', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res21', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res22', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res23', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res24', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res25', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res26', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 256],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res27', summary_collection=self.summary_collection)
                resnet = customlayers.resnet_block(resnet, channels=[256, 512],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res28', summary_collection=self.summary_collection)
                encoder = customlayers.resnet_block(resnet, channels=[512, 512],
                                                    pool_dim=2, is_train=self.is_train,
                                                    name='res29', summary_collection=self.summary_collection)
                self.encoder = encoder #store the encoder in an attribute for easy access
                print('Final shape: ' + str(encoder.outputs.get_shape().as_list()))


########################################################################################################################
#                                                   Classifier                                                         #
########################################################################################################################
            print('[*] CLASSIFIER')
            with tf.variable_scope('classifier') as vs:
                with tf.variable_scope('out1x1_1') as vs:
                    classifier1 = tl.layers.Conv1dLayer(encoder,
                                                act=customlayers.prelu,
                                                        shape=[1, 512, self._opts._nclasses],
                                                stride=1,
                                                padding='SAME',
                                                W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                                W_init_args={},
                                                b_init=tf.constant_initializer(value=0.1),
                                                b_init_args={},
                                                name='1x1_layer')
                    classifier1.outputs = tf.reshape(classifier1.outputs,
                                                     [self._opts._batchsize, self._opts._nclasses])

                with tf.variable_scope('out1x1_2') as vs:
                    classifier2 = tl.layers.Conv1dLayer(encoder,
                                                act=customlayers.prelu,
                                                shape=[1, 512, self._opts._nclasses],
                                                stride=1,
                                                padding='SAME',
                                                W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                                W_init_args={},
                                                b_init=tf.constant_initializer(value=0.1),
                                                b_init_args={},
                                                name='1x1_layer')
                    classifier2.outputs = tf.reshape(classifier2.outputs,
                                                     [self._opts._batchsize, self._opts._nclasses])

                    # this output is of shape [batch, 1, classes]
                with tf.variable_scope('outlayer_concat') as vs:
                    classifier = customlayers.StackLayer([classifier1, classifier2], axis=-1) # along the channels
########################################################################################################################
#                                       Garbage Detector (Currently out of use.)                                       #
########################################################################################################################
            print('[*] GARBAGE_DETECTOR')
            with tf.variable_scope('garbage_detec') as vs:
                flat = tl.layers.FlattenLayer(encoder, name='flatten')
                garbage_detector = tl.layers.DenseLayer(flat,
                                                n_units=64,
                                                act=customlayers.prelu,
                                                name='fc')
                dropout = tl.layers.DropoutLayer(garbage_detector,
                                                keep=0.5,
                                                is_train=self.is_train,
                                                is_fix=True,
                                                name='dropout')

            with tf.variable_scope('garbage_detec2') as vs:
                garbage_detector = tl.layers.DenseLayer(dropout,
                                                n_units=2,
                                                act=customlayers.prelu,
                                                name='fc')

            if valid_mode:
                classifier.outputs = tf.Print(classifier.outputs, [classifier.outputs.get_shape(),
                                                               classifier.outputs, classifier.outputs],
                                              message='outVALID') if self._opts._debug else classifier.outputs
                return classifier, garbage_detector
            else:
                classifier.outputs = tf.Print(classifier.outputs, [classifier.outputs.get_shape(),
                                                               classifier.outputs, classifier.outputs],
                                              message='out') if self._opts._debug else classifier.outputs
                return classifier, garbage_detector

    def get_loss(self, raw_logits, labels, valid_mode=False):
        """Add the loss ops to the current graph.

        Args:
          raw_logits: A `Tensor` holding the activations from the network.
          labels: A `Tensor` holding the one hot encoded ground truth.
          valid_mode: A `bool`, define the model in trainings mode or validation mode.

        Returns:
          loss: A `Tensor` object holding the loss as scalar.
          f1_score: A `Tensor` object holding the F1 score.
        """
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
            tf.summary.scalar('avg_pred_positives', tf.divide(nr_pred_positives, self._opts._batchsize),
                              collections=self.summary_collection)
            tf.summary.scalar('avg_true_positives', tf.divide(nr_true_positives, self._opts._batchsize),
                              collections=self.summary_collection)

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
            self.log_file.write("[*] Initialized loss with posweights: \n")
            self.log_file.write(str(self.pos_weight))

            # tile the pos weigths:
            pos_weights = tf.reshape(tf.tile(self.pos_weight, multiples=[self._opts._batchsize]),
                                     [self._opts._batchsize, self._opts._nclasses])
            assert pos_weights.get_shape().as_list() == [self._opts._batchsize, self._opts._nclasses]

            # get the FOCAL LOSS
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
            loss = fl_mean + l2_loss

            tf.summary.scalar('loss_total', loss, collections=self.summary_collection)
            tf.summary.scalar('loss_l2', l2_loss, collections=self.summary_collection)
            tf.summary.scalar('loss_1-tp', tp_loss, collections=self.summary_collection)
            tf.summary.scalar('loss_focal_mean', fl_mean, collections=self.summary_collection)
            tf.summary.scalar('loss_focal_sum', fl_sum, collections=self.summary_collection)
            tf.summary.scalar('loss_CE', ce_mean, collections=self.summary_collection)

            return loss, f1_score

    def get_opt(self, loss, vars=[], adam=False):
        """Adds an optimizer to the current computational graph.

        Args:
          loss: A `Tensor` 0d, Scalar - The loss to minimize.
          vars: A `list` holding all variables to optimize. If empty all Variables are optmized.
          adam: A `bool` defining whether to use the adam optimizer or not. Defaults to False

        Returns:
          A tf.Optimizer.
        """
        if adam:
            if vars:
                opt = tf.train.AdamOptimizer(learning_rate=self._opts._learningrate,
                                             beta1=0.9, beta2=0.999,
                                             epsilon=self._opts._epsilon,
                                             use_locking=False, name='Adam').minimize(loss, var_list=vars)

            else:
                opt = tf.train.AdamOptimizer(learning_rate=self._opts._learningrate, beta1=0.9, beta2=0.999,
                                             epsilon=self._opts._epsilon, use_locking=False, name='Adam').minimize(loss)
        else:
            if vars:
                opt = tf.train.AdagradOptimizer(learning_rate=self._opts._learningrate,
                                                initial_accumulator_value=0.1,
                                                use_locking=False, name='Adagrad').minimize(loss, var_list=vars)
            else:
                opt = tf.train.AdagradOptimizer(learning_rate=self._opts._learningrate,
                                                initial_accumulator_value=0.1,
                                                use_locking=False, name='Adagrad').minimize(loss)
        return opt

    def feed_dict(self):
        """Get a feed_dict to run the session.

        Depracted as we run the dropout layers in "is_fix" mode.

        Returns:
          An empty dict.
        """
        return {}

    def initialize_helpers(self):
        """Initialize the model and call all graph constructing ops.

        This function is a wrapper for all initialization ops in the model.
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
        self.batchgen = helpers.BatchGenerator(self._opts)

        self.log_file.write('Initialized ROC_tracker\n')
        self.ROCtracker = helpers.RocTracker(self._opts)

    def train(self, restore_whole=True):
        """Start the training process for the model.

        Runs training and validation ops with preferences specified in the config_dict.

        Args:
          restore_whole: A `bool` defining wheter to restore the complete model (including the outlayers -> True) or
            the model without the classification-layers (False).
        """
        # get a graph
        train_graph = tf.Graph()
        with train_graph.as_default():
            self.initialize_helpers()

            # define the filenames for validation and training:
            train_filenames = glob.glob(os.path.join(self._opts._batchesdir,
                                                     '*train_batch_%s_*' % str(self._opts._windowlength)))
            if self._opts._gpu == 'True':
                device = '/gpu:0'
            else:
                device = '/cpu:0'
            with tf.device(device):
                # graph for training:
                train_batch, train_labels, _, _ = self.input_pipeline(train_filenames, valid_mode=False)
                classifier, _ = self.model(train_batch, valid_mode=False)

                train_raw_logits = classifier.outputs

                train_loss, train_acc = self.get_loss(raw_logits=train_raw_logits, labels=train_labels,
                                                      valid_mode=False)

                opt = self.get_opt(train_loss, vars=[])

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

                train_summaries = tf.summary.merge_all(key='train')

            self.saver = tf.train.Saver()

            #get time and stats_collector
            start_time = time.time()
            stats_collector = []

            # get the queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=self.session)

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
                    self.log_file.write('Step %d: Av.accuracy = %.2f (%.3f sec)\n' % (step, av_accuracy,
                                                                                      time.time() - start_time))
                    self.log_file.flush()

                if step % 2000 == 0:
                    # save the model in nais model snapshots like in model book
                    self.save_model(classifier, self.session, step=step, name='Classifier')
                    self.eval_while_train(step, 2000)
                else:
                    _, loss, acc, labels = self.session.run([opt, train_loss, train_acc, train_labels],
                                                            feed_dict=self.feed_dict())
                    stats_collector.append((loss, acc))

            # gracefully shut down the queue
            coord.request_stop()
            coord.join(threads)

    def validate(self):
        """
        This Funtion runs the evaluation mode only. Model is restored from the restore path specified in the
        config_dict. Then validation ops are run and Metrics are calculated.
        """
        eval_graph = tf.Graph()
        self.is_train = False

        valid_filenames = glob.glob(os.path.join(self._opts._batchesdir,
                                                 '*valid_batch_%s_*' % str(self._opts._windowlength)))

        # get a graph
        if self._opts._allowsoftplacement == 'True':
            config = tf.ConfigProto(allow_soft_placement=True)
        else:
            config = tf.ConfigProto(allow_soft_placement=False)
        # allow growth to survey the consumed GPU memory
        config.gpu_options.allow_growth=True
        with eval_graph.as_default():

            self.initialize_helpers()
            if self._opts._gpu == 'True':
                device = '/gpu:0'
            else:
                device = '/cpu:0'
            with tf.device(device):
                with tf.Session(config=config) as self.session:
                    if self.valid_graph_initialized:
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

                    self.eval_writer = tf.summary.FileWriter(self._opts._summariesdir + '/valid')

                    self.session.run(tf.global_variables_initializer())
                    self.session.run(tf.local_variables_initializer())

                    # restore the model weights
                    infer_classifier = self.load_model_weights(infer_classifier, session=self.session,
                                                         name='Classifier') #THIS RUNS THE SESSION INTERNALLY

                    valid_summaries = tf.summary.merge_all(key='valid')

                    eval_coord = tf.train.Coordinator()
                    eval_threads = tf.train.start_queue_runners(coord=eval_coord, sess=self.session)

                    average_acc = []

                    step = 0
                    try:
                        while not eval_coord.should_stop():
                            step += 1
                            # control everything with the coordinator
                            if eval_coord.should_stop():
                                break

                            summary, loss, outlayer, \
                            acc, labels, sigmoid_logits, _ = self.session.run([valid_summaries,
                                                                       valid_loss,
                                                                       valid_raw_logits,
                                                                       valid_acc,
                                                                       valid_labels,
                                                                       valid_sigmoid_logits,
                                                                       labelss
                                                                       ],
                                                                      feed_dict=self.feed_dict()
                                                                      )
                            # pass the predictions to the ROC tracker:
                            self.ROCtracker.update(sigmoid_logits=sigmoid_logits, true_labels=labels)
                            self.eval_writer.add_summary(summary, step)

                            average_acc.append(acc)

                    except tf.errors.OutOfRangeError:
                        average_acc = sum(average_acc)/len(average_acc)
                        self.log_file.write('[*] Finished validation'
                                            ' with av.acc of %s' % str(average_acc))
                        self.log_file.flush()
                    finally:
                        # when done ask the threads to stop
                        eval_coord.request_stop()

                    eval_coord.join(eval_threads)
                    # do a safe on the inference parameters
                    #self.save_params(infer_classifier, session=self.session, ignore=None)
                    self.session.close()

        self.ROCtracker.calc_and_save(self.log_file)

        # set train flag back to true
        self.is_train = True

    def eval_while_train(self, step=1, eval_steps=200):
        """Run the validation during the training process.
        This OP restores the model for evaluation during the training process from the latest .npz save.
        This process happens in a different session and different graph than the training.
        The validation is performed on the whole validation set metrics
        like TP, FP, etc are collected to calculate the ROC curves.

        Args:
          step: A `int32` defining the step at which validatoin was initialized.
          eval_steps: A `int32` defining the number of validation steps to perform
            eval_steps*batchsize should exceed the number of samples in the valid dataset.
        """
        # get a graph:
        eval_graph = tf.Graph()
        self.is_train = False

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
            if self._opts._gpu == 'True':
                device = '/gpu:0'
            else:
                device = '/cpu:0'
            with tf.device(device):
                with tf.Session(config=config) as sess:
                    if self.valid_graph_initialized:
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
                    # do a safe on the inference parameters
                    self.save_params(infer_classifier, ignore=None)
                    sess.close()

        self.ROCtracker.calc_and_save(self.log_file)

        # set train flag back to true
        self.is_train = True

    def generate_embedding(self, embedding_dims=512, reduce_dims=False):
        """Generate a protein-embedding based from a trained model.
        This function generates a Protein embedding based on the validation dataset.
        In detail it builds an inference graph, loads a pretrained model and stores the features of the
        last dense layer in a embdding matrix.
        The protein-name of each sequence passed through the net is stored in a dictionary and converted to
        integers , which allows later lookup.

        Args:
          embedding_dims: A `int32` defining the dimensions of the protein embedding
            The dimensions of the resulting embedding should be equal to the out_dims of the network.
          reduce_dims: A `bool` defining wheter to reduce the dimensions via PCA prior to dumping the embedding.
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
                                            shape=[self._opts._batchsize,
                                                   self._opts._depth, self._opts._windowlength, 1])

            inference_net, _ = self.model(input_seq_node, valid_mode=False)

            # load the pretrained model
            self.session.run(tf.global_variables_initializer())
            self.load_model_weights(inference_net, session=self.session, name='Classifier')

            # initialize the embedding with UNK token, the dict is updated automatically for UNK
            self.embedding = np.zeros([1, 512])

            for i in range(self.batchgen.eval_batch_nr): # this is calculated when we initialize batchgen

                batch = self.batchgen.generate_valid_batch(include_garbage=False)
                # run the session and retrieve the embedded_batch
                embed_batch = self.session.run(self.encoder.outputs, feed_dict={input_seq_node: batch})

                # reshape to 2D:
                embed_batch = np.reshape(embed_batch, [self._opts._batchsize, embed_batch.shape[-1]])

                print(embed_batch.shape)

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
            assert embedding_dims == 512
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
                embedding.tensor_name = 'ProteinEmbedding'
                # Link this tensor to its metadata file (e.g. labels).
                embedding.metadata_path = os.path.join(embedding_dir, 'metadata.tsv')

                summary_writer = tf.summary.FileWriter(embedding_dir)
                projector.visualize_embeddings(summary_writer, config)

                # initialize Vars and get a Saver and a writer
                embedding_sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()

                # run it:
                _ = embedding_sess.run(mockOP,
                                       feed_dict={embedding_node: fitted_embedding})

                saver.save(embedding_sess, os.path.join(embedding_dir, "ProteinEmbedding.ckpt"), 1)

            # now save the metadata as tsv:

            # load the mapping from json:
            mapping_dict = '/net/data.isilon/igem/2017/data/swissprot_with_EC/swissprot_with_EC.csvGO_EC_mapping.p'

            with open(mapping_dict) as jsonf:
                mapping = json.load(jsonf)

            with open(os.path.join(embedding_dir, 'metadata.tsv'), "w") as out_fobj:
                header = '\t'.join(['GO(s)', 'EC(s)', 'name', 'ID'])
                header += '\n'
                out_fobj.write(header)
                for n, key in enumerate(self.batchgen.embedding_dict.keys()):
                    # get the mapping from GO to EC:
                    EC = set()
                    for go in self.batchgen.embedding_dict[key]['labels']:
                        try:
                            ec_map = mapping[go]
                            EC.update(ec_map)
                        except KeyError:
                            ec_map = ['no_EC_assigned']
                            EC.update(ec_map)
                    EC_list = list(EC)
                    line = [','.join(self.batchgen.embedding_dict[key]['labels']), ','.join(EC_list), key, str(n)]
                    line = '\t'.join(line)
                    line += '\n'
                    out_fobj.write(line)
