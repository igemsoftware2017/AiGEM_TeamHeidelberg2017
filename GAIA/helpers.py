import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import re, os
import random
import math
import json
import matplotlib.pyplot as plt
import sklearn.metrics
from seaborn import barplot, set_style
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict


class OptionHandler():
    def __init__(self, config_dict):
        #self._usefp16 = config_dict['usefp16']
        self.config = config_dict
        self._name = config_dict['model_name']
        self._thresholdnum = config_dict['threshold_num']
        self._gpu = config_dict['gpu']
        self._allowsoftplacement = config_dict['allow_softplacement']
        self._numepochs = config_dict['num_epochs']
        self._numsteps = config_dict['num_steps']
        self._kmer2vec_embedding = config_dict['kmer2vec_embedding']
        self._kmer2vec_kmerdict = config_dict['kmer2vec_kmerdict']
        self._kmer2vec_kmercounts = config_dict['kmer2vec_kmercounts']
        self._kmersize = config_dict['kmer_size']
        self._nkmers = config_dict['n_kmers']
        self._embeddingdim = config_dict['embedding_dim']
        self._depth = config_dict['depth']
        self._structuredims = config_dict['structure_dims']
        self._traindata = config_dict['train_data']
        self._validdata = config_dict['valid_data']
        self._batchesdir = config_dict['batches_dir']
        self._inferencedata = config_dict['inference_data']
        self._inferencemode = True if config_dict['inference_mode'] == 'True' else False
        self._labels = config_dict['labels']
        self._nclasses = config_dict['n_classes']
        self._topk = config_dict['topk']
        self._classbalancing = True if config_dict['class_balancing'] == 'True' else False
        self._maxclassinbalance = config_dict['maxclass_inbalance']
        self._dropoutrate = config_dict['dropoutrate']
        self._learningrate = config_dict['learning_rate']
        self._epsilon = config_dict['epsilon']
        self._batchsize = config_dict['batch_size']
        self._batchgenmode = config_dict['batchgen_mode']
        self._windowlength = config_dict['window_length']
        self._minlength = config_dict['min_length']
        self._numthreads = config_dict['num_threads'] #TODO ASSERT THIS NUMBER!!!!!!!!
        self._restorepath = config_dict['restore_path']
        self._restore = True if config_dict['restore'] == 'True' else False
        self._debug = True if config_dict['debug'] == 'True' else False
        self._ecfile = config_dict['EC_file']
        self._summariesdir = config_dict['summaries_dir'] # for tensorboard
        self._summariesdir = self._summariesdir + '_{l}_{n}_{w}_{g}_{b}_{lr}_{e}'.format(g=self._batchgenmode,
                                                                                   w=self._windowlength,
                                                                               n=self._nclasses,
                                                                        b=self._batchsize,
                                                                        lr=self._learningrate,
                                                                        e=self._epsilon,
                                                                        l=self._labels)
        self._seqfile = config_dict['seqfile']
        self._survivalpop = config_dict['survival_pop']
        self._generations = config_dict['generations']
        self._systematic = config_dict['systematic']
        self._muts_per_gen = config_dict['muts_per_gen']
        self._decrease_muts_after_gen = config_dict['decrease_muts_after_gen']

        if not os.path.exists(self._summariesdir):
            os.makedirs(self._summariesdir)
        if not os.path.exists(self._batchesdir):
            os.makedirs(self._batchesdir)

    def write_dict(self):
        """
        Store the config_dict to disc in the save_dir
        :return:
        """
        with open(os.path.join(self._summariesdir, 'config_dict.JSON'), "w") as config_dict:
            json.dump(self.config, config_dict)


class RocTracker():
    def __init__(self, optionhandler):
        self._opts = optionhandler
        self.metrics_path = os.path.join(self._opts._summariesdir, 'metrics')
        if not os.path.exists(self.metrics_path):
            os.mkdir(self.metrics_path)
        self.metrics_file = open(os.path.join(self.metrics_path, 'metrics.csv'), "w")
        self.roc_score = []
        self.roc_labels = []
        self.pred_positives_sum = np.zeros(self._opts._nclasses)
        self.actual_positives_sum = np.zeros(self._opts._nclasses)
        self.true_positive_sum = np.zeros(self._opts._nclasses)
        self.num_calculations = 0

    def update(self, sigmoid_logits, true_labels):
        """
        update the ROC tracker, with the predictions on one batch made during validation
        """
        # threshold this thing
        # we consider a class "predicted" if it's sigmoid activation is higher than 0.5 (predicted labels)
        batch_predicted_labels = np.greater(sigmoid_logits, 0.5)
        batch_predicted_labels = batch_predicted_labels.astype(float)


        batch_pred_pos = np.sum(batch_predicted_labels, axis=0) #sum up along the batch dim, keep the channels
        batch_actual_pos = np.sum(true_labels, axis=0) #sum up along the batch dim, keep the channels
        # calculate the true positives:
        batch_true_pos = np.sum(np.multiply(batch_pred_pos, batch_actual_pos), axis=0)

        # and update the counts
        self.pred_positives_sum += batch_pred_pos #what the model said
        self.actual_positives_sum += batch_actual_pos #what the labels say
        self.true_positive_sum += batch_true_pos # where labels and model predictions>0.5 match

        assert len(self.true_positive_sum) == self._opts._nclasses

        # add the predictions to the roc_score tracker
        self.roc_score.append(sigmoid_logits)
        self.roc_labels.append(true_labels)

    def calc_and_save(self, logfile):
        """
        Calculate the ROC curve with AUC value for the collected test values (roc_scores, roc_labels).
        Writes everything to files, plots curves and resets the Counters afterwards.
        """
        self.metrics_file = open(os.path.join(self.metrics_path, 'metrics.csv'), "w")

        self.num_calculations += 1

        # concat score and labels along the batchdim -> a giant test batch
        self.roc_score = np.concatenate(self.roc_score, axis=0)
        self.roc_labels = np.concatenate(self.roc_labels, axis=0)

        # get the total number of seqs we tested on:
        logfile.write('[*] Calculating metrics\n')
        test_set_size = self.roc_labels.shape[0]

        # do the calculations
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=np.reshape(self.roc_labels, (-1)),
                                                         y_score=np.reshape(self.roc_score, (-1)))
        auc = sklearn.metrics.auc(fpr, tpr)

        precision_arr, recall_arr, thresholds = sklearn.metrics.precision_recall_curve(
            y_true=np.reshape(self.roc_labels, (-1)), probas_pred=np.reshape(self.roc_score, (-1))) # micro-average PR curve

        # now save everything to the metrics.csv
        # metrics = [fpr, tpr, auc,
        #            thresholds, precision_arr, recall_arr, thresholds]
        # metrics_as_str = []
        # for m in metrics:
        #     if isinstance(m, np.ndarray):
        #         m_str = ','.join(str(e) for e in m.tolist())
        #     else:
        #         m_str = str(m)
        #     metrics_as_str.append(m_str)
        #
        # line = ';'.join(metrics_as_str)
        # line += '\n'
        # self.metrics_file.write(line)
        #self.metrics_file.flush()

        # write get the max, min and avg scores for each class:
        # determine the scores for the labels
        scores = self.roc_score * self.roc_labels

        mean_scores = np.mean(scores, axis=0)
        assert mean_scores.shape[0] == self._opts._nclasses
        max_scores = np.amax(scores, axis=0)
        assert max_scores.shape[0] == self._opts._nclasses
        min_scores = np.amin(scores, axis=0)
        assert min_scores.shape[0] == self._opts._nclasses

        self.metrics_file.write(str(mean_scores) + '\n')
        self.metrics_file.write(str(max_scores) + '\n')
        self.metrics_file.write(str(min_scores) + '\n')

        self.metrics_file.close()

        # get printable metrics (for log file)
        precision_class = self.true_positive_sum / np.maximum(1, self.pred_positives_sum) # where predPositives_sum == 0, tp_sum is also 0
        recall_class = self.true_positive_sum / np.maximum(1, self.actual_positives_sum) # where actualPositives_sum == 0, tp_sum is also 0
        precision = np.sum(self.true_positive_sum) / np.sum(self.pred_positives_sum)
        recall = np.sum(self.true_positive_sum) / np.sum(self.actual_positives_sum)
        f1 = 2*precision*recall / (precision + recall)
        logfile.write("[*] Tested on %d seqs, "
                      "precision %.2f%%, "
                      "recall %.2f%%, "
                      "F1 %.2f%%\n" % (test_set_size, precision, recall, f1))
        logfile.flush()



        #plot ROC:
        plot_simple_curve(x=fpr, y=tpr, title=self._opts._name + '_ROC_curve',
                          legend=self._opts._name + ' (AUC = %0.4f)' % auc,
                          xname='False positive rate', yname='True positive rate',
                          filename=os.path.join(self.metrics_path, self._opts._name + '.roc_%d' % self.num_calculations))


        # PR curve
        plot_simple_curve(x=recall_arr, y=precision_arr,
                          title=self._opts._name + ' PR curve', legend=self._opts._name,
                          xname='Recall', yname='Precision',
                          filename=os.path.join(self.metrics_path, self._opts._name + '.precision_%d' % self.num_calculations))

        # reset the stats-collectors:
        self.roc_score = []
        self.roc_labels = []
        self.pred_positives_sum = np.zeros(self._opts._nclasses)
        self.actual_positives_sum = np.zeros(self._opts._nclasses)

        logfile.write('[*] Done testing.\n')


class StratifiedCounterDict(dict):
    def __missing__(self, key):
        self[key] = {'tp': 0,
                     'pred_p': 0,
                     }
        return self[key]


class BatchGenerator():
    def __init__(self, optionhandler, kmer2vec_embedding, kmer2id):
        self._opts = optionhandler
        self.mode = self._opts._batchgenmode # one of ['window', 'bigbox', 'dynamic']
        self._kmer2vec_embedding = kmer2vec_embedding
        self._kmer2id = kmer2id
        self.inferencedata = open(self._opts._inferencedata, 'r')
        self.traindata = open(self._opts._traindata, 'r')
        self.validdata = open(self._opts._validdata, 'r')
        self.AA_to_id = {}
        self.id_to_AA = {}
        self.class_dict = OrderedDict()
        self.id_to_class = OrderedDict()
        self._get_class_dict()
        self.embedding_dict = OrderedDict()
        # determine the number of batches for eval from lines in the validdata and the garbagepercentage
        self.garbage_percentage = 0.2
        self.garbage_count = 0 # a counter for generated garbage sequences
        self.eval_batch_nr = int(_count_lines(self._opts._validdata) * (1 + self.garbage_percentage) //
                              self._opts._batchsize)
        print('Initialized Batchgen with   batchsize: %d,   numeval_batches: %d at'
              '                            garbage_percentage: %f' % (self._opts._batchsize,
                                                                      self.eval_batch_nr,
                                                                      self.garbage_percentage))
        self.batches_per_file = 10000
        self.epochs = 2000
        self.curr_epoch = 0
        self.label_enc = OneHotEncoder(n_values=self._opts._nclasses, sparse=False)
        self.AA_enc = 'where we put the encoder for the AAs'

        if self.mode.startswith('one_hot'):
            AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',]

            self.AA_enc = OneHotEncoder(n_values=self._opts._depth, sparse=False)

            if 'physchem' in self.mode:
                _hydro = [1.8, 2.5, -3.5, -3.5, 2.8,
                          -0.4, -3.2, 4.5, -3.9, 3.8,
                          1.9, -3.5, -1.6, -3.5, -4.5,
                          -0.8, -0.7, 4.2, -0.9, -1.3]
                _molarweight = [89.094, 121.154, 133.104, 147.131, 165.192,
                                75.067, 155.156, 131.175, 146.189, 131.175,
                                149.208, 132.119, 115.132, 146.146, 174.203,
                                105.093, 119.119, 117.148, 204.228, 181.191]
                _is_polar = lambda aa: 1 if aa in ['DEHKNQRSTY'] else 0
                _is_aromatic = lambda aa: 1 if aa in ['FWY'] else 0
                _has_hydroxyl = lambda aa: 1 if aa in ['ST'] else 0 #should we add TYR??
                _has_sulfur = lambda aa: 1 if aa in ['CM'] else 0

                for i, aa in enumerate(AAs):
                    self.AA_to_id[aa]  = {'id': len(self.AA_to_id),
                                          'hydro': _hydro[i],
                                          'molweight': _molarweight[i],
                                          'pol': _is_polar(aa),
                                          'arom': _is_aromatic(aa),
                                          'sulf': _has_sulfur(aa),
                                          'OH': _has_hydroxyl(aa)}
            else:
                for aa in AAs:
                    self.AA_to_id[aa] = len(self.AA_to_id)
                # get the inverse:
                self.id_to_AA = {}
                for aa, id in self.AA_to_id.items():
                    self.id_to_AA[id] = aa
                self.id_to_AA[42] = '_'

    def _get_class_dict(self):
        with open(self._opts._ecfile, "r") as ec_fobj:
            for line in ec_fobj:
                fields = line.strip().split()
                if fields[1].endswith('.csv'):  #TODO delete this when error is fixed
                    fields[1] = fields[1].rstrip('.csv')

                if self._opts._labels == 'EC':
                    self.class_dict[fields[1]] = {'id': len(self.class_dict),
                                                  'size': int(fields[0]),
                                                  }
                if self._opts._labels == 'GO':
                    self.class_dict[fields[1].split('_')[1]] = {'id': len(self.class_dict),
                                                                'size': int(fields[0]),
                                                                }

        # get a reverse dict:
        for key in self.class_dict.keys():
            self.id_to_class[self.class_dict[key]['id']] = key

    def _update_embedding_dict(self, name, labels):
        """
        Update the embedding dict for new entries. This is used on the fly as we perform inference
        batchgen
        """
        if len(self.embedding_dict) == 0:
            # add UNK token the first time this method is called
            self.embedding_dict['UNK'] = {}
            self.embedding_dict['UNK']['labels'] = ['UNK']
            self.embedding_dict['UNK']['id'] = 0 # we save 0 for the UNK token

        # check if the key (= name) is already in the dict:
        if name not in self.embedding_dict:
            assert len(self.embedding_dict) > 0
            self.embedding_dict[name] = {}
            self.embedding_dict[name]['labels'] = labels
            self.embedding_dict[name]['id'] = len(self.embedding_dict)
        else:
            print(name)
            print('WARNING: Overwrote value in embedding dict. Check infile for redundant sequences!')

    def _csv_EC_decoder(self, in_csv, encoded_labels=True):
        line = in_csv.readline()
        fields = line.strip().split(';')
        name = fields[0]
        seq = fields[1]
        if fields[2].endswith('.csv'): #TODO assert this
            fields[2] = fields[2].rstrip('.csv')
        if self._opts._labels == 'EC':
            EC_str = fields[2] #TODO assert this
            if encoded_labels:
                EC_CLASS = 0 if self._opts._inferencemode else self.class_dict[EC_str]['id']
                label = [[EC_CLASS]] # we need a 2D array
            else:
                label = EC_str
        elif self._opts._labels == 'GO':
            GO_str = fields[2] #TODO assert this
            GOs = 0 if self._opts._inferencemode else fields[2].split(',') #TODO assert this
            if encoded_labels:
                label = [[self.class_dict[go]['id']] for go in GOs] # returns a 2D array
            else:
                label = GOs
        # TODO add an assertion for mode
        return name, seq, label

    def _seq2tensor(self, seq):
        """
        Does what you think it does.

        As for now we Fix the considered length to 200AA position,
        yielding a tensor of shape:
        [100, 196, 1]
        """
        if self.mode == 'one_hot_padded':
            # first check if the sequence fits in the box:
            if len(seq) <= self._opts._windowlength:
                seq_matrix = np.ndarray(shape=(len(seq)), dtype=np.int32)
            # if sequence does not fit we clip it:
            else:
                seq_matrix = np.ndarray(shape=(self._opts._windowlength), dtype=np.int32)
            for i in range(len(seq_matrix)):
                seq_matrix[i] = self.AA_to_id[seq[i]]
            start_pos = 0 #because our sequence sits at the beginning of the box
            length = len(seq_matrix)  #true length (1 based)
            # now encode the sequence in one-hot
            oh_seq_matrix = np.reshape(self.AA_enc.fit_transform(np.reshape(seq_matrix, (1, -1))), (len(seq_matrix), 20))
            # pad the sequence to the boxsize:
            npad = ((0, self._opts._windowlength-length), (0, 0))
            padded_seq_matrix = np.pad(oh_seq_matrix, pad_width=npad, mode='constant', constant_values=0)
            padded_seq_matrix = np.transpose(padded_seq_matrix)
            del oh_seq_matrix, seq_matrix

            # seq_matrix = np.ndarray(shape=(self._opts._windowlength), dtype=np.int32)
            # for i in range(len(seq_matrix)):
            #     seq_matrix[i] = self.AA_to_id[seq[i]]
            # start_pos = 0 #because our sequence sits at the beginning of the box
            # length = len(seq_matrix)  #true length (1 based)
            # now encode the sequence in one-hot
            # oh_seq_matrix = np.reshape(self.AA_enc.fit_transform(np.reshape(seq_matrix, (1, -1))), (len(seq_matrix), self._opts._depth))
            # pad the sequence to the boxsize:
            # npad = ((0, self._opts._windowlength-length), (0, 0))
            # padded_seq_matrix = np.pad(oh_seq_matrix, pad_width=npad, mode='constant', constant_values=0)
            # padded_seq_matrix = np.transpose(padded_seq_matrix)
            # del oh_seq_matrix, seq_matrix

            return padded_seq_matrix, start_pos, length #true length 1 based


        elif self.mode == 'one_hot_padded_physchem':
            # TODO implement this shit
            pass

        else:
            print("Error: MODE must be of ['one_hot_padded', 'one_hot_padded_physchem']")

    def _encode_single_seq(self, seq, desired_label=None):
        """
        Encode single sequence.
        """
        seq_matrix, start_pos, length = self._seq2tensor(seq)
        # look up the label in the class_dict:
        if desired_label:
            desired_label_ID = self.class_dict[desired_label]['id']

            # encode label one_hot:
            oh_label = self.label_enc.fit_transform([[desired_label_ID]]) # of shape [1, n_classes]
            return oh_label, seq_matrix, start_pos, length

        else:
            return seq_matrix

    def _process_csv(self, queue, return_name=True, encode_labels=True):
        """
        pls infer from name.
        """
        name, seq, label = self._csv_EC_decoder(queue, encoded_labels=encode_labels)
        seq_matrix, start_pos, end_pos = self._seq2tensor(seq)
        if return_name:
            return name, label, seq_matrix, start_pos, end_pos
        else:
            return label, seq_matrix, start_pos, end_pos

    def generate_garbage_sequence(self, return_name=False):
        """
        Generates a sequence full of garbage, e.g. a obviously non functional sequence.
        :return:
        """
        modes = ['complete_random', 'pattern', 'same']
        mode = modes[random.randint(0, 2)]
        self.garbage_count += 1

        # get the length of the protein
        length = random.randint(175, self._opts._windowlength-10) #enforce padding

        if mode == 'pattern':
            #print('pattern')
            # Generate a repetitive pattern of 5 AminoAcids to generate the prot
            # get a random nr of AAs to generate the pattern:
            AA_nr = random.randint(2, 5)
            # get an index for each AA in AA_nr
            idxs = []
            for aa in range(AA_nr):
                idx_found = False
                while not idx_found:
                    aa_idx = random.randint(0, 19)
                    if not aa_idx in idxs:
                        idxs.append(aa_idx)
                        idx_found = True
            reps = math.ceil(length/AA_nr)
            seq = reps * idxs
            length = len(seq)

        elif mode == 'complete_random':
            # print('complete_random')
            seq = []
            for aa in range(length):
                # get an idx for every pos in length:
                idx = random.randint(0, 19)
                seq.append(idx)

        elif mode == 'same':
            # print('ONE')
            AA = random.randint(0, 19)
            seq = length * [AA]

        label = np.zeros([self._opts._nclasses])
        label = np.expand_dims(label, axis=0)
        garbage_label = np.asarray([1])
        garbage_label = np.expand_dims(garbage_label, axis=0)
        oh_seq_matrix = np.reshape(self.AA_enc.fit_transform(np.reshape(seq, (1, -1))), (len(seq), 20))
        # pad the sequence to the boxsize:
        npad = ((0, self._opts._windowlength-length), (0, 0))
        padded_seq_matrix = np.pad(oh_seq_matrix, pad_width=npad, mode='constant', constant_values=0)
        padded_seq = np.transpose(padded_seq_matrix)
        if return_name:
            # return a sequence ID to identify the generated sequence
            # generate a "random" name
            name = 'g%d' % self.garbage_count
            return name, padded_seq, label, garbage_label
        else:
            return padded_seq, label, garbage_label

    def generate_random_data_batch(self):
        seq_tensor_batch = tf.random_normal([self._opts._batchsize, self._opts._embeddingdim, self._opts._windowlength, 1])

        label_batch = [np.random.randint(1,self._opts._nclasses) for _ in range(self._opts._batchsize)]
        index_batch = [tf.constant(label) for label in label_batch]
        label_tensor = tf.stack(index_batch)
        onehot_labelled_batch = tf.one_hot(indices=tf.cast(label_tensor, tf.int32),
                                           depth=self._opts._nclasses)
        return seq_tensor_batch, onehot_labelled_batch

    def generate_single_seq_batch(self, seq, desired_label):
        """
        Generate a batch from a single sequence. (means, the whole batch is the same seq) we need
        this as the network is made for a certain batchsize. If we restore the model we need to keep this batchsize constant.
        """
        seq_tensors = []
        label_batch = []
        for _ in range(self._opts._batchsize):
            oh_label, seq_tensor, _, _ = self._encode_single_seq(seq, desired_label)
            label_batch.append(oh_label)
            seq_tensors.append(seq_tensor)
        batch_tensor = np.expand_dims(np.stack(seq_tensors, axis=0), axis=-1)
        label_tensor = np.concatenate(label_batch, axis=0)
        # drop the first dimension

        return batch_tensor, label_tensor

    def generate_inference_batch(self):
        """
        Generates a batch to infer the labels for sequences, as everything is fed into the same graph,
        we use the same kind of preprocessing and basically the same function as generate_batch but on
        another file.
        :return: batch, labels (empty), positions
        """
        seq_tensors = []
        label_batch = []
        positions = np.ndarray([self._opts._batchsize, 2])
        lengths = np.ndarray([self._opts._batchsize])
        in_csv = self.inferencedata
        for i in range(self._opts._batchsize):
            try:
                """ Note that this is not shuffled! """
                ECclass, seq_tensor, start_pos, end_pos = self._process_csv(in_csv, return_name=False,
                                                                      encode_labels=True)
                label_batch.append(ECclass)
                seq_tensors.append(seq_tensor)
            except IndexError: # catches error from csv_decoder
                # reopen the file:
                in_csv.close()
                # TODO: implement file shuffling when we reopen the file
                self.inferencedata = open(self._opts._inferencedata, 'r')
                in_csv = self.inferencedata
                """ redo """
                ECclass, seq_tensor, start_pos, end_pos = self._process_csv(in_csv, return_name=False,
                                                                            encode_labels=True)
                label_batch.append(ECclass)
                seq_tensors.append(seq_tensor)

                positions[i, 0] = start_pos
                positions[i, 1] = end_pos
                lengths[i] = end_pos

        batch = np.stack(seq_tensors, axis=0)

        if 'spp' in self.mode:
            return batch, label_batch, lengths
        if 'padded' in self.mode:
            return batch, label_batch, lengths
        else:
            return batch, label_batch, positions

    def generate_batch(self, is_train):
        """
            generate batches to train the model:
            as we use the sparse softmax ce, we DO NOT NEED TO ONE HOT ENCODE OUR LABELS!
        """
        seq_tensors = []
        label_batch = []
        positions = np.ndarray([self._opts._batchsize, 2])
        lengths = np.ndarray([self._opts._batchsize])
        if is_train:
            in_csv = self.traindata
        else:
            in_csv = self.validdata
        for i in range(self._opts._batchsize):
            try:
                """ Note that this is not shuffled! """
                ECclass, seq_tensor, start_pos, end_pos = self._process_csv(in_csv, return_name=False,
                                                                            encode_labels=True)
                label_batch.append(ECclass)
                seq_tensors.append(seq_tensor)
            except IndexError: # catches error from csv_decoder
                # reopen the file:
                in_csv.close()
                # TODO: implement file shuffling when we reopen the file
                if is_train:
                    self.traindata = open(self._opts._traindata, 'r')
                    in_csv = self.traindata
                else:
                    self.validdata = open(self._opts._validdata, 'r')
                    in_csv = self.validdata
                """ redo """
                ECclass, seq_tensor, start_pos, end_pos = self._process_csv(in_csv, return_name=False,
                                                                            encode_labels=True)
                label_batch.append(ECclass)
                seq_tensors.append(seq_tensor)

                positions[i, 0] = start_pos
                positions[i, 1] = end_pos
                lengths[i] = end_pos

        batch = np.stack(seq_tensors, axis=0)

        if 'spp' in self.mode:
            return batch, label_batch, lengths
        if 'padded' in self.mode:
            return batch, label_batch, lengths
        else:
            return batch, label_batch, positions

    def generate_valid_batch(self, include_garbage=False):
        """
        Generates a batch to infer the labels for sequences, as everything is fed into the same graph,
        we use the same kind of preprocessing and basically the same function as generate_batch but on
        another file.
        :return: batch, labels (empty), positions
        """
        seq_tensors = []
        in_csv = self.validdata
        if not include_garbage:
            for i in range(self._opts._batchsize):
                try:
                    """ Note that this is not shuffled! """
                    name, label, seq_tensor, _, _ = self._process_csv(in_csv, return_name=True,
                                                                      encode_labels=False)
                    self._update_embedding_dict(name, label)
                    seq_tensors.append(seq_tensor)
                except IndexError: # catches error from csv_decoder
                    # reopen the file:
                    in_csv.close()
                    # TODO: implement file shuffling when we reopen the file
                    self.validdata = open(self._opts._validdata, 'r')
                    in_csv = self.validdata
                    """ redo """
                    name, label, seq_tensor, _, _ = self._process_csv(in_csv, return_name=True,
                                                                      encode_labels=False)
                    self._update_embedding_dict(name, label)
                    seq_tensors.append(seq_tensor)

        #
        elif include_garbage:
            num_garbage = math.ceil(self._opts._batchsize * self.garbage_percentage)
            for i in range(self._opts._batchsize - num_garbage):
                try:
                    """ Note that this is not shuffled! """
                    name, label, seq_tensor, _, _ = self._process_csv(in_csv, return_name=True,
                                                                      encode_labels=False)
                    self._update_embedding_dict(name, label)
                    seq_tensors.append(seq_tensor)
                except IndexError: # catches error from csv_decoder
                    # reopen the file:
                    in_csv.close()
                    # TODO: implement file shuffling when we reopen the file
                    self.validdata = open(self._opts._validdata, 'r')
                    in_csv = self.validdata
                    """ redo """
                    name, label, seq_tensor, _, _ = self._process_csv(in_csv, return_name=True,
                                                                      encode_labels=False)
                    self._update_embedding_dict(name, label)
                    seq_tensors.append(seq_tensor)

            for i in range(num_garbage):
                name, seq_tensor, _, _ = self.generate_garbage_sequence(return_name=True)
                label = 'garbage'
                self._update_embedding_dict(name, label)
                seq_tensors.append(seq_tensor)

        batch = np.stack(seq_tensors, axis=0)
        batch = np.expand_dims(batch, axis=-1)
        return batch


class TFrecords_generator():
    def __init__(self, optionhandler):
        self._opts = optionhandler
        self.label_enc = OneHotEncoder(n_values=self._opts._nclasses, sparse=False)
        self.AA_enc = 'where we put the encoder for the AAs'
        self.mode = self._opts._batchgenmode # one of ['window', 'bigbox', 'dynamic']
        self._kmer2vec_embedding = 'kmer2vec_embedding'
        self._kmer2id = {}
        self.inferencedata = open(self._opts._inferencedata, 'r')
        self.traindata = open(self._opts._traindata, 'r')
        self.validdata = open(self._opts._validdata, 'r')
        self.AA_to_id = {}
        self.class_dict = {}
        self._get_class_dict()
        self.structure_dict = {}
        self.examples_per_file = 10000
        self.epochs = self._opts._numepochs
        self.curr_epoch = 0
        self.writer = 'where we put the writer'

        # get the structure_dict
        structure_forms = ['UNORDERED', 'HELIX', 'STRAND', 'TURN']
        assert len(structure_forms) == self._opts._structuredims-1
        for s in structure_forms:
            self.structure_dict[s] = len(self.structure_dict) + 1 #serve the 0 for NO INFORMATION
        #self.structure_enc = OneHotEncoder(n_values=self._opts._structuredims, sparse=False)

        if self.mode.startswith('one_hot'):
            AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',]

                   #'X']
            self.AA_enc = OneHotEncoder(n_values=self._opts._depth, sparse=False)
            if 'physchem' in self.mode:
                _hydro = [1.8, 2.5, -3.5, -3.5, 2.8,
                          -0.4, -3.2, 4.5, -3.9, 3.8,
                          1.9, -3.5, -1.6, -3.5, -4.5,
                          -0.8, -0.7, 4.2, -0.9, -1.3]
                _molarweight = [89.094, 121.154, 133.104, 147.131, 165.192,
                                75.067, 155.156, 131.175, 146.189, 131.175,
                                149.208, 132.119, 115.132, 146.146, 174.203,
                                105.093, 119.119, 117.148, 204.228, 181.191]
                _is_polar = lambda aa: 1 if aa in ['DEHKNQRSTY'] else 0
                _is_aromatic = lambda aa: 1 if aa in ['FWY'] else 0
                _has_hydroxyl = lambda aa: 1 if aa in ['ST'] else 0 #should we add TYR??
                _has_sulfur = lambda aa: 1 if aa in ['CM'] else 0

                for i, aa in enumerate(AAs):
                    self.AA_to_id[aa]  = {'id': len(self.AA_to_id),
                                          'hydro': _hydro[i],
                                          'molweight': _molarweight[i],
                                          'pol': _is_polar(aa),
                                          'arom': _is_aromatic(aa),
                                          'sulf': _has_sulfur(aa),
                                          'OH': _has_hydroxyl(aa)}
            else:
                for aa in AAs:
                    self.AA_to_id[aa] = len(self.AA_to_id)
                # get the inverse:
                self.id_to_AA = {}
                for aa, id in self.AA_to_id.items():
                    self.id_to_AA[id] = aa
                self.id_to_AA[42] = '_'

        elif self.mode.startswith('embed'):
            with open(self._opts._kmer2vec_kmerdict, "r") as vocab_file:
                for line in vocab_file:
                    fields = line.strip().split()
                    fields[0] = fields[0].strip('\'b')
                    self._kmer2id[fields[0]] = len(self._kmer2id)
            #print(self._kmer2id)

    def _get_class_dict(self):
        with open(self._opts._ecfile, "r") as ec_fobj:
            for line in ec_fobj:
                fields = line.strip().split()
                if fields[1].endswith('.csv'):  #TODO delete this when error is fixed
                    fields[1] = fields[1].rstrip('.csv')

                if self._opts._labels == 'EC':
                    self.class_dict[fields[1]] = {'id': len(self.class_dict),
                                                  'size': int(fields[0]),
                                                  }
                if self._opts._labels == 'GO':
                    self.class_dict[fields[1].split('_')[1]] = {'id': len(self.class_dict),
                                                  'size': int(fields[0]),
                                                  }

    def _csv_EC_decoder(self, in_csv):
        line = in_csv.readline()
        fields = line.strip().split(';')
        name = fields[0]
        seq = fields[1]
        if self._opts._labels == 'EC':
            if fields[3].endswith('.csv'):
                fields[3] = fields[3].rstrip('.csv')
            EC_str = fields[3]
            EC_CLASS = 0 if self._opts._inferencemode else self.class_dict[EC_str]['id']
            label = [[EC_CLASS]] # we need a 2D array
        elif self._opts._labels == 'GO':
            GO_str = fields[2]
            GOs = 0 if self._opts._inferencemode else fields[2].split(',')
            if GOs[0].endswith('.csv'):
                GOs = [go.rstrip('.csv') for go in GOs]
            label = [[self.class_dict[go]['id']] for go in GOs] # returns a 2D array
        # TODO add an assertion for mode
        structure_str = fields[3]
        return name, seq, label, structure_str

    def _seq2tensor(self, seq):
        """
        Does what you think it does.

        As for now we Fix the considered length to 200AA position,
        yielding a tensor of shape:
        [100, 196, 1]
        """
        if self.mode == 'one_hot_padded':
            # first check if the sequence fits in the box:
            if len(seq) <= self._opts._windowlength:
                seq_matrix = np.ndarray(shape=(len(seq)), dtype=np.int32)
            # if sequence does not fit we clip it:
            else:
                seq_matrix = np.ndarray(shape=(self._opts._windowlength), dtype=np.int32)
            for i in range(len(seq_matrix)):
                seq_matrix[i] = self.AA_to_id[seq[i]]
            start_pos = 0 #because our sequence sits at the beginning of the box
            length = len(seq_matrix)  #true length (1 based)
            # now encode the sequence in one-hot
            oh_seq_matrix = np.reshape(self.AA_enc.fit_transform(np.reshape(seq_matrix, (1, -1))), (len(seq_matrix), 20))
            # pad the sequence to the boxsize:
            npad = ((0, self._opts._windowlength-length), (0, 0))
            padded_seq_matrix = np.pad(oh_seq_matrix, pad_width=npad, mode='constant', constant_values=0)
            padded_seq_matrix = np.transpose(padded_seq_matrix)
            del oh_seq_matrix, seq_matrix

            # seq_matrix = np.ndarray(shape=(self._opts._windowlength), dtype=np.int32)
            # for i in range(len(seq_matrix)):
            #     seq_matrix[i] = self.AA_to_id[seq[i]]
            # start_pos = 0 #because our sequence sits at the beginning of the box
            # length = len(seq_matrix)  #true length (1 based)
            # now encode the sequence in one-hot
            # oh_seq_matrix = np.reshape(self.AA_enc.fit_transform(np.reshape(seq_matrix, (1, -1))), (len(seq_matrix), self._opts._depth))
            # pad the sequence to the boxsize:
            # npad = ((0, self._opts._windowlength-length), (0, 0))
            # padded_seq_matrix = np.pad(oh_seq_matrix, pad_width=npad, mode='constant', constant_values=0)
            # padded_seq_matrix = np.transpose(padded_seq_matrix)
            # del oh_seq_matrix, seq_matrix

            return padded_seq_matrix, start_pos, length #true length 1 based

        elif self.mode == 'one_hot_padded_physchem':
            # TODO implement this shit
            pass

        elif self.mode.startswith('embed'):
            k = self._opts._kmersize
            # split the sequence into words
            frame_words = [seq[start:start + k]
                           for start in range(0, len(seq))]

            # determine the vector of IDs to lookup simultaneously in the embedding
            frame_ids = [self._kmer2id[w] for w in frame_words if len(w) == k]

            seq_matrix = np.zeros(shape=(self._opts._windowlength), dtype=np.int32)
            for i in range(len(seq_matrix)):
                try:
                    seq_matrix[i] = frame_ids[i]
                except IndexError: #means no more frames
                    pass


            start_pos = 0
            # pad the sequence to 1000
            length = len(frame_words) if len(frame_words) <= self._opts._windowlength else self._opts._windowlength
            # npad = ((0, self._opts._windowlength-length))
            # padded_seq_matrix = np.pad(seq_matrix, pad_width=npad, mode='constant', constant_values=0)
            padded_seq_matrix = seq_matrix
            del seq_matrix
            return padded_seq_matrix, start_pos, length

        else:
            print("Error: MODE must be of ['one_hot_padded', 'one_hot_padded_physchem', 'embed']")

    def _get_structure(self, structure_str, seq_length):
        """
        Construct a One Hot Encoded Tensor with height = self._structure_dims, width = self._windowlength
        :param structure_str: str
        the entry in the swissprot csv corresponding to the FT fields in the swissprot textfile download
        Example format:
        [('TURN', '11', '14'), ('HELIX', '19', '27'), ('STRAND', '32', '36'), ('HELIX', '45', '54'),
        ('STRAND', '59', '69'), ('STRAND', '72', '80'), ('HELIX', '86', '96'), ('HELIX', '99', '112'),
        ('HELIX', '118', '123'), ('HELIX', '129', '131'), ('HELIX', '134', '143'), ('STRAND', '146', '149'),
        ('HELIX', '150', '156'), ('STRAND', '157', '159'), ('HELIX', '173', '182'), ('STRAND', '186', '189'),
        ('HELIX', '192', '194'), ('HELIX', '199', '211'), ('STRAND', '216', '221'), ('HELIX', '226', '239'),
        ('STRAND', '242', '246'), ('HELIX', '272', '275'), ('HELIX', '277', '279'), ('STRAND', '283', '285')]
        :return:
        """
        # if there is info about the structure:
        if structure_str != '[]':
            # get an array of len length:
            structure = np.ones([seq_length])
            # modify the structure str:
            # TODO: Improve the super ugly hack with a proper regex
            structure_str = re.sub('[\'\[\]\(]', '', structure_str)
            structure_features = [j.strip(', ').split(', ') for j in structure_str.strip(')').split(')')]

            for ft in structure_features:
                # get the ID for the ft:
                id_to_write = self.structure_dict[ft[0]]
                start = int(ft[1])
                end = int(ft[2])
                for i in range(start, end+1):
                    structure[i] = id_to_write
            # encode it One-Hot:

            #oh_structure_matrix = np.reshape(self.structure_enc.fit_transform(np.reshape(structure, [1, -1])),
                                             #[len(structure), self._opts._structuredims])
            # now pad it up to windowlength:
            # npad = ((0, self._opts._windowlength-seq_length), (0, 0))
            # padded_structure_matrix = np.pad(oh_structure_matrix, pad_width=npad, mode='constant', constant_values=0)
            # padded_structure_matrix = np.transpose(padded_structure_matrix)
            npad = ((0, self._opts._windowlength-seq_length))
            padded_structure_matrix = np.pad(structure, pad_width=npad, mode='constant', constant_values=0)
            #assert padded_structure_matrix.shape[1] == self._opts._windowlength

        else:
            # return only zeros if there is no information about the structure
            padded_structure_matrix = np.zeros([self._opts._windowlength])

        return padded_structure_matrix

    def _process_csv(self, queue):
        """
        pls infer from name.
        """
        _, seq, labels, structure_str = self._csv_EC_decoder(queue)
        seq_matrix, start_pos, length = self._seq2tensor(seq)
        structure_tensor = self._get_structure(structure_str, length)
        # encode the label one_hot:
        oh_label_tensor = self.label_enc.fit_transform(labels) # of shape [1, n_classes]
        classes = oh_label_tensor.shape[0]
        # open an array full of zeros to add the labels to
        oh_labels = np.zeros(self._opts._nclasses)
        for c in range(classes):
            oh_labels += oh_label_tensor[c]

        oh_labels = np.expand_dims(oh_labels, axis=0)

        return oh_labels, seq_matrix, structure_tensor, start_pos, length

    def generate_garbage_sequence(self):
        """
        Generates a sequence full of garbage, e.g. a obviously non functional sequence.
        :return:
        """
        modes = ['complete_random', 'pattern', 'same']
        mode = modes[random.randint(0, 2)]

        # get the length of the protein
        length = random.randint(175, self._opts._windowlength-1)

        if mode == 'pattern':
            #print('pattern')
            # Generate a repetitive pattern of 5 AminoAcids to generate the prot
            # get a random nr of AAs to generate the pattern:
            AA_nr = random.randint(2, 5)
            # get an index for each AA in AA_nr
            idxs = []
            for aa in range(AA_nr):
                idx_found = False
                while not idx_found:
                    aa_idx = random.randint(0, 19)
                    if not aa_idx in idxs:
                        idxs.append(aa_idx)
                        idx_found = True
            reps = math.ceil(length/AA_nr)
            seq = reps * idxs
            length = len(seq)

        elif mode == 'complete_random':
            # print('complete_random')
            seq = []
            for aa in range(length):
                # get an idx for every pos in length:
                idx = random.randint(0, 19)
                seq.append(idx)

        elif mode == 'same':
            # print('ONE')
            AA = random.randint(0, 19)
            seq = length * [AA]

        label = np.zeros([self._opts._nclasses])
        label = np.expand_dims(label, axis=0)
        garbage_label = np.asarray([1])
        garbage_label = np.expand_dims(garbage_label, axis=0)
        oh_seq_matrix = np.reshape(self.AA_enc.fit_transform(np.reshape(seq, (1, -1))), (len(seq), 20))
        # pad the sequence to the boxsize:
        npad = ((0, self._opts._windowlength-length), (0, 0))
        padded_seq_matrix = np.pad(oh_seq_matrix, pad_width=npad, mode='constant', constant_values=0)
        padded_seq = np.transpose(padded_seq_matrix)
        return padded_seq, label, garbage_label

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def example_to_TFrecords(self, is_train, garbage_percentage=0.2, structure=True):
        include_garbage = False if garbage_percentage == 0 else True

        # determine how many files we need to write:
        if is_train:
            length_data_set = _count_lines(self._opts._traindata)
            batch_files_name = os.path.basename(self._opts._traindata) + 'train_batch_{}'.format(str(self._opts._windowlength))
            print(batch_files_name)
            in_csv = self.traindata

        else:
            length_data_set = _count_lines(self._opts._validdata)
            batch_files_name = os.path.basename(self._opts._validdata) + 'valid_batch_{}'.format(str(self._opts._windowlength))
            print(batch_files_name)
            in_csv = self.validdata

        files_to_write = np.int32(np.ceil(length_data_set*(1+garbage_percentage)*2
                                          / float(self.examples_per_file))) # write every thing twice

        for n in range(1, files_to_write+1):
            file_path = os.path.join(self._opts._batchesdir, batch_files_name) + '_' + str(n)
            self.writer = tf.python_io.TFRecordWriter(file_path)

            if structure:

                for i in range(self.examples_per_file):
                    if include_garbage and  i % int(1/garbage_percentage) == 0:
                        # print("garbage_seq")
                        seq_tensor, label, garbage_label = self.generate_garbage_sequence()
                        structure_label = np.zeros([self._opts._windowlength])

                        assert seq_tensor.shape == (self._opts._depth, self._opts._windowlength), "%s" % str(seq_tensor.shape)
                        assert label.shape == (1, self._opts._nclasses)
                        # convert the features to a raw string:
                        seq_raw = seq_tensor.tostring()
                        label_raw = label.tostring()
                        garbage_label_raw = garbage_label.tostring()
                        structure_label_raw = structure_label.tostring()

                        example = tf.train.Example(
                            features=tf.train.Features(feature={
                                'windowlength': self._int64_feature(self._opts._windowlength),
                                'structure_depth': self._int64_feature(self._opts._structuredims),
                                'depth': self._int64_feature(self._opts._depth),
                                'label_classes': self._int64_feature(self._opts._nclasses),
                                'seq_raw': self._bytes_feature(seq_raw),
                                'label_raw': self._bytes_feature(label_raw),
                                'garbage_label_raw': self._bytes_feature(garbage_label_raw),
                                'structure_label_raw': self._bytes_feature(structure_label_raw),
                            }))
                        self.writer.write(example.SerializeToString())
                    else:
                        # print("validseq")
                        try:
                            oh_labels, seq_tensor, structure_label, _, _ = self._process_csv(in_csv)

                        except IndexError: # catches error from csv_decoder -> reopen the file:
                            in_csv.close()
                            if is_train:
                                self.traindata = open(self._opts._traindata, 'r')
                                in_csv = self.traindata
                            else:
                                self.validdata = open(self._opts._validdata, 'r')
                                in_csv = self.validdata
                            oh_labels, seq_tensor, structure_label, _, _ = self._process_csv(in_csv)

                        garbage_label = np.asarray([0]) # NOT garbage
                        garbage_label = np.expand_dims(garbage_label, axis=0)

                        assert seq_tensor.shape == (self._opts._depth, self._opts._windowlength)
                        assert oh_labels.shape == (1, self._opts._nclasses)
                        # convert the features to a raw string:
                        seq_raw = seq_tensor.tostring()
                        label_raw = oh_labels.tostring()
                        garbage_label_raw = garbage_label.tostring()
                        structure_label_raw = structure_label.tostring()

                        example = tf.train.Example(
                            features=tf.train.Features(feature={
                                'windowlength': self._int64_feature(self._opts._windowlength),
                                'structure_depth': self._int64_feature(self._opts._structuredims),
                                'depth': self._int64_feature(self._opts._depth),
                                'label_classes': self._int64_feature(self._opts._nclasses),
                                'seq_raw': self._bytes_feature(seq_raw),
                                'label_raw': self._bytes_feature(label_raw),
                                'garbage_label_raw': self._bytes_feature(garbage_label_raw),
                                'structure_label_raw': self._bytes_feature(structure_label_raw),
                            }))
                        self.writer.write(example.SerializeToString())

            elif not structure:

                for i in range(self.examples_per_file):
                    if include_garbage and  i % int(1/garbage_percentage) == 0:
                        # print("garbage_seq")
                        assert seq_tensor.shape == (self._opts._depth, self._opts._windowlength), "%s" % str(seq_tensor.shape)
                        assert label.shape == (1, self._opts._nclasses)
                        # convert the features to a raw string:
                        seq_raw = seq_tensor.tostring()
                        label_raw = label.tostring()

                        example = tf.train.Example(
                            features=tf.train.Features(feature={
                                'windowlength': self._int64_feature(self._opts._windowlength),
                                'depth': self._int64_feature(self._opts._depth),
                                'label_classes': self._int64_feature(self._opts._nclasses),
                                'seq_raw': self._bytes_feature(seq_raw),
                                'label_raw': self._bytes_feature(label_raw),
                            }))
                        self.writer.write(example.SerializeToString())
                    else:
                        # print("validseq")
                        try:
                            oh_labels, seq_tensor, _, _, _ = self._process_csv(in_csv)

                        except IndexError: # catches error from csv_decoder -> reopen the file:
                            in_csv.close()
                            if is_train:
                                self.traindata = open(self._opts._traindata, 'r')
                                in_csv = self.traindata
                            else:
                                self.validdata = open(self._opts._validdata, 'r')
                                in_csv = self.validdata
                            oh_labels, seq_tensor, _, _, _ = self._process_csv(in_csv)

                        assert seq_tensor.shape == (self._opts._depth, self._opts._windowlength)
                        assert oh_labels.shape == (1, self._opts._nclasses)
                        # convert the features to a raw string:
                        seq_raw = seq_tensor.tostring()
                        label_raw = oh_labels.tostring()

                        example = tf.train.Example(
                            features=tf.train.Features(feature={
                                'windowlength': self._int64_feature(self._opts._windowlength),
                                'depth': self._int64_feature(self._opts._depth),
                                'label_classes': self._int64_feature(self._opts._nclasses),
                                'seq_raw': self._bytes_feature(seq_raw),
                                'label_raw': self._bytes_feature(label_raw),
                            }))
                        self.writer.write(example.SerializeToString())
            self.writer.close()

    def embed_and_to_TFrecords(self, is_train):
        """
        Process the dataset to embedded sequences, and wirte the sequences as TF records.
        :return:
        """
        # ensure _batches_dir is correct:
        assert self._opts._batchesdir.endswith('embed/')

        # create batchesdir if not exists
        if not os.path.exists(self._opts._batchesdir):
            os.mkdir(self._opts._batchesdir)

        # collect the file for train/valid
        if is_train:
            length_data_set = _count_lines(self._opts._traindata)
            batch_files_name = os.path.basename(self._opts._traindata) + 'train_batch_{}'.format(str(self._opts._windowlength))
            print(batch_files_name)
            in_csv = self.traindata

        else:
            length_data_set = _count_lines(self._opts._validdata)
            batch_files_name = os.path.basename(self._opts._validdata) + 'valid_batch_{}'.format(str(self._opts._windowlength))
            print(batch_files_name)
            in_csv = self.validdata

        # "placeholder" for length:
        length_node = tf.placeholder(dtype=tf.int32)

        # load the embedding and construct a session to do the lookup!
        with tf.Session() as sess:
            with tf.variable_scope('kmer2vec') as vs:
                embedding = tf.get_variable('embedding',
                                                shape=[self._opts._nkmers,
                                                       self._opts._embeddingdim],
                                                trainable=False)
                #embedding_saver = tf.train.Saver({"w_out": embedding})
                embedding_saver = tf.train.Saver({"n_emb": embedding})
                embedding_saver.restore(sess, tf.train.latest_checkpoint(
                                                  self._opts._kmer2vec_embedding))

            with tf.variable_scope('process_sequence') as vs:
                sequence_node = tf.placeholder(tf.int32, shape=[self._opts._windowlength])

                # slice the sequence and extract the words to be embedded:
                #true_seq = sequence_node[:length_node]
                # look it up
                embedded_sequence = tf.transpose(tf.nn.embedding_lookup(embedding, sequence_node))
                embedded_sequence = tf.reshape(embedded_sequence, [self._opts._embeddingdim, self._opts._windowlength])

            # determine how many files we need to write:
            files_to_write = np.int32(np.ceil(length_data_set * 2 / float(self.examples_per_file))) # write every thing twice

            for n in range(1, files_to_write+1):
                file_path = os.path.join(self._opts._batchesdir, batch_files_name) + '_' + str(n)
                self.writer = tf.python_io.TFRecordWriter(file_path)

                for _ in range(self.examples_per_file):

                    try:
                        oh_labels, seq_tensor, _, length = self._process_csv(in_csv)

                    except IndexError: # catches error from csv_decoder -> reopen the file:
                        in_csv.close()
                        if is_train:
                            self.traindata = open(self._opts._traindata, 'r')
                            in_csv = self.traindata
                        else:
                            self.validdata = open(self._opts._validdata, 'r')
                            in_csv = self.validdata
                        oh_labels, seq_tensor, _, length = self._process_csv(in_csv)

                    feed_dict = {sequence_node: seq_tensor, length_node: length}

                    embedded_seq = sess.run(embedded_sequence, feed_dict=feed_dict) # will return a list
                    embedded_seq = np.asarray(embedded_seq)

                    self._embed_to_TFrecords(embedded_seq, oh_labels)
            self.writer.close()

    def _embed_to_TFrecords(self, embedded_seq, oh_label):

        assert embedded_seq.shape == (self._opts._embeddingdim, self._opts._windowlength)
        assert oh_label.shape == (1, self._opts._nclasses)

        #print(np.argmax(oh_label, axis=1))

        # convert the features to a raw string:
        seq_raw = embedded_seq.tostring()
        label_raw = oh_label.tostring()

        example = tf.train.Example(
            features=tf.train.Features(feature={
                'windowlength': self._int64_feature(self._opts._windowlength),
                'depth': self._int64_feature(self._opts._embeddingdim),
                'label_classes': self._int64_feature(self._opts._nclasses),
                'seq_raw': self._bytes_feature(seq_raw),
                'label_raw': self._bytes_feature(label_raw),
            }))

        self.writer.write(example.SerializeToString())

    def produce_train_valid(self):
        if self.mode.startswith('one_hot'):
            #self.example_to_TFrecords(is_train=True, garbage_percentage=0, structure=False)
            self.example_to_TFrecords(is_train=False, garbage_percentage=0, structure=False)

        elif self.mode.startswith('embed'):
            self.embed_and_to_TFrecords(is_train=True)
            #self.embed_and_to_TFrecords(is_train=False)


def plot_histogram(log_file, save_dir):
    count_dict = {}
    with open(log_file, "r") as in_fobj:
        for line in in_fobj:
            pred_labels = line.strip().split()
            for label in pred_labels:
                try:
                    count_dict[label] += 1
                except KeyError:
                    count_dict[label] = 0
    bars = [count_dict[label] for label in count_dict.keys()]
    labels = [label for label in count_dict.keys()]
    set_style("whitegrid")
    fig, ax = plt.subplots()
    ax = barplot(x=bars, y=labels)
    fig.save(os.path.join(save_dir, 'negative_test.png'))


def plot_simple_curve(x, y, title, legend, xname, yname, filename):
    plt.ioff()
    fig = plt.figure()
    plt.title(title)
    plt.plot(x, y, color="red", lw=2, label=legend)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.legend(loc="lower right")
    plt.savefig(filename+".svg")
    plt.savefig(filename+".png")
    plt.close(fig)


def _count_lines(file_path):
    count = 0
    with open(file_path, "r") as fobj:
        for line in fobj:
            count += 1
    return count


def _add_var_summary(var, name, collection=None):
    """ attaches a lot of summaries to a given tensor"""
    with tf.name_scope(name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean, collections=collection)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev, collections=collection)
            tf.summary.scalar('max', tf.reduce_max(var), collections=collection)
            tf.summary.scalar('min', tf.reduce_min(var), collections=collection)
            tf.summary.histogram('histogram', var, collections=collection)


def _variable_on_cpu(name, shape, initializer, trainable):
    """ Helper function to get a variable stored on cpu"""
    with tf.device('/cpu:0'): #TODO will this work?
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    #dtf.add_to_collection('CPU', var)
    return var


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
