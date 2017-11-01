import numpy as np
import random
import os

os.environ['MPLCONFIGDIR'] = "."  # set this to something reasonable not mounted
# CAUTION: it is possible that this script caches a lot of data into this directory,
# better not choose one mounted via network ;-)


import pprint
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.colors as mplcolors
from datetime import datetime
import pickle
from scipy.special import expit
import sys
import random
import itertools
import json

plt.style.use(json.load(open('/net/data.isilon/igem/2017/scripts/clonedDeeProtein/DeeProtein/style.json', 'r')))
font = font_manager.FontProperties(fname='/net/data.isilon/igem/2017/data/cache/fonts/JosefinSans-Regular.tff')
monospaced = font_manager.FontProperties(fname='/net/data.isilon/igem/2017/data/cache/fonts/DroidSansMono-Regular.ttf')


cdict = {'red': [(0.0, 0.6056862745, 0.6256862745),
                 (0.5, 1.0, 1.0),
                 (1.0, 0.9725490196, 1.0)],

         'green': [(0.0, 0.0998039216, 0.1198039216),
                   (0.5, 1.0, 1.0),
                   (1.0, 0.6196078431, 1.0)],

         'blue': [(0.0, 0.1154901961, 0.1354901961),
                  (0.5, 1.0, 1.0),
                  (1.0, 0.1137254902, 1.0)]}

class Stats:
    """
    Holds statistics, saves them as pickle file, plots them as PNGs.

    """


    def __init__(self, savedir=None, goallabels=None, avoidslabels=None, goalsdict=None, avoidsdict=None, head=None,
                 pickledir=None):
        """
        Initialises the statistics either empty or from a picklefile.
        Args:
            savedir(str): directory to which the data is saved,
                not needed when initialising from a pickle file (optional).
            goallabels(list of str): names written to the plots for the goal classes,
                not needed when initialising from a pickle file (optional).
            avoidslabels(list of str): names written to the plots for the avoid classes,
                not needed when initialising from a pickle file (optional).
            goalsdict(list of int): keys for the class dict, specify the goal classes for highligthing,
                not needed when initialising from a pickle file (optional).
            avlidssdict(list of int): keys for the class dict, specify the avoids classes for highligthing,
                not needed when initialising from a pickle file (optional).
            head(str): Title of the written plots,
                not needed when initialising from a pickle file (optional).
            pickledir(str): directory from which the Stats object is restored,
                only needed when initialising from a pickle file (optional).
        """

        if pickledir:
            self.load(pickledir)
        else:
            self.score = []
            self.goals = [[] for _ in range(len(goallabels))]
            self.goals_var = [[] for _ in range(len(goallabels))]
            self.avoids = [[] for _ in range(len(avoidslabels))]
            self.avoids_var = [[] for _ in range(len(avoidslabels))]
            self.savedir = savedir
            self.goallabels = goallabels
            self.avoidslables = avoidslabels
            self.goalsdict = goalsdict
            self.avoidsdict = avoidsdict
            self.mutated = []
            self.head = head[1:]
            self.mutating = []
            self.mutatingpositions = []
            self.systematic = []
            self.blosum_score = []
            self.lenseq = 0
            self.mutated_aas = [0 for _ in range(1000)]
        plt.switch_backend('agg')

    def plot_all(self, f_width=10, f_height=8, res=80, name='stats.png'):
        """
        Plots the development of each goal score, avoid score and their variances,
        indicates wether mutations occur and wether thesystematic mode is used
        Args:
            f_width (float): specifies the width of the plot that is written
            f_height (float): specifies the height of the plot that is written
            res (float): specifies the resolution of the plot that is written
            name (str): specifies the name of plots that is written
        """

        time = range(len(self.score))

        plt.figure(1, figsize=(f_width, f_height), dpi=res, facecolor='w', edgecolor='k')
        ax = plt.subplot(111)
        ax.set_ylim(-0.1, 1.1)
        ax.plot(time, self.score, label='Score', zorder=1)
        ax.plot(time, self.blosum_score, label='Blosum Score')
        ax.plot(time, self.mutated, label='mutated')

        for goal in range(len(self.goals)):
            goallabel = 'goal: {}'.format(self.goallabels[goal])
            ax.plot(time, self.goals[goal], label=goallabel)

        for avoid in range(len(self.avoids)):
            avoidslabel = 'avoid: {}'.format(self.avoidslables[avoid])
            ax.plot(time, self.avoids[avoid], label=avoidslabel)

        for goal_var in range(len(self.goals)):
            goals_var_label = 'goal (var score): {}'.format(self.goallabels[goal_var])
            ax.plot(time, self.goals_var[goal_var], label=goals_var_label)

        for avoid_var in range(len(self.avoids)):
            avoid_var_label = 'avoid (var score): {}'.format(self.avoidslables[avoid_var])
            ax.plot(time, self.avoids_var[avoid_var], label=avoid_var_label)

        # not mutating
        ax.fill_between(time, -0.1, -0.06, where=np.asarray(self.mutating) == 0, facecolor='grey', alpha=0.3)
        # systematic
        ax.fill_between(time, -0.06, -0.02, where=np.asarray(self.systematic) == 1, facecolor='red', alpha=0.3)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.gcf().text(0.8, 0.2, 'Generation: {}'.format(len(self.mutated)-1))
        plt.ylabel('Score')
        plt.xlabel('Generation')
        plt.title(self.head)
        plt.savefig(os.path.join(self.savedir, name))
        plt.gcf().clear()

    def plotdistoverseq(self, f_width=16, f_height=8, res=200, name='hist.png'):
        """
        Plots how often a position was mutated in a bar plot, including backmutations
        Args:
            f_width (float): specifies the width of the plot that is written
            f_height (float): specifies the height of the plot that is written
            res (float): specifies the resolution of the plot that is written
            name (str): specifies the name of plots that is written
        """

        plt.figure(1, figsize=(f_width, f_height), dpi=res, facecolor='w', edgecolor='k')
        seqpos = range(self.lenseq)
        plt.bar(seqpos, height=self.mutatingpositions, width=1)
        plt.ylabel('Number of mutations')
        plt.xlabel('Sequence position')
        plt.title('Distribution of mutations')
        plt.savefig(os.path.join(self.savedir, name))
        plt.gcf().clear()

    def plotdisttooriginal(self, f_width=16, f_height=8, res=200, name='hist_rel.png'):
        """
        Plots how often a position was not the original residue
        Args:
            f_width (float): specifies the width of the plot that is written
            f_height (float): specifies the height of the plot that is written
            res (float): specifies the resolution of the plot that is written
            name (str): specifies the name of plots that is written
        """

        plt.figure(1, figsize=(f_width, f_height), dpi=res, facecolor='w', edgecolor='k')
        seqpos = range(self.lenseq)
        plt.bar(seqpos, height=self.mutated_aas[:self.lenseq-1], width=1)
        plt.ylabel('Number of mutations')
        plt.xlabel('Sequence position')
        plt.title('Distribution of mutations')
        plt.savefig(os.path.join(self.savedir, name))
        plt.gcf().clear()


    def update(self, classes, scores, mutated, mutating, seqs, systematic, classes_variance, blosum_score,
               mutated_aas):
        """
        Updates the statistics with a new set of values, dumps the current state of the statistics object as pickle
        Args:
            classes(ndarray): logits retrieved from the classifier
            garbage(garbage scores): depracted
            scores(list): Scores calculated by the scoring function
            mutated(float): Amount of the sequence that is mutated
            mutating(0 or 1): indicates wether a mutation in the best sequenced occured, compared to the last generation
            seqs(ndarray): one-hot encoded sequences with additional information in dimension 20 and 21 in the one-hot
                dimension (1)
            systematic(0 or 1): indicates wether systematic mode was active
            classes_variance(ndarray): variance between logits from the classifier
            blosum_score(float): sum of blosum62 matrix entries for the mutations in the best sequence
            mutated_aas(list): List of positions that are mutated in the current best sequences
        """

        best = np.argmax(scores)
        self.score.append(scores[best])
        self.mutated.append(mutated)
        self.mutating.append(mutating)
        self.lenseq = int(np.sum(seqs[0, :20, :]))

        bestgoals = [classes[best, goalid] for goalid in self.goalsdict]
        bestavoids = [classes[best, avoidid] for avoidid in self.avoidsdict]

        for goal in range(len(self.goals)):
            self.goals[goal].append(bestgoals[goal])

        for avoid in range(len(self.avoids)):
            self.avoids[avoid].append(bestavoids[avoid])

        bestgoals_var = [(expit(classes_variance[best, goalid])) for goalid in self.goalsdict]
        bestavoids_var = [(expit(classes_variance[best, avoidid])) for avoidid in self.avoidsdict]

        for goal in range(len(self.goals_var)):
            self.goals_var[goal].append(bestgoals_var[goal])

        for avoid in range(len(self.avoids_var)):
            self.avoids_var[avoid].append(bestavoids_var[avoid])

        self.mutatingpositions = seqs[best, 21, :]
        self.systematic.append(systematic)
        self.blosum_score.append(blosum_score[best])

        for el in mutated_aas:
            self.mutated_aas[el] += 1

        pickle.dump(
                [self.score,
                self.goals,
                self.avoids,
                self.savedir,
                self.goallabels,
                self.avoidslables,
                self.goalsdict,
                self.avoidsdict,
                self.mutated,
                self.head,
                self.mutating,
                self.mutatingpositions,
                self.systematic,
                self.goals_var,
                self.avoids_var,
                self.blosum_score,
                self.mutated_aas],
            open(os.path.join(self.savedir, 'stats.p'), 'wb'))


    def load(self, picklefile):
        """
        Loads picklefile and sets the attributes to the values from the picklefile.
        Args:
            picklefile(str): path to picklefile to restore from
        """

        pobj = open(picklefile, 'rb')
        unpickled = pickle.load(pobj)
        self.score = unpickled[0]
        self.goals = unpickled[3]
        self.avoids = unpickled[4]
        self.savedir = unpickled[5]
        self.goallabels = unpickled[6]
        self.avoidslables = unpickled[7]
        self.goalsdict = unpickled[8]
        self.avoidsdict = unpickled[9]
        self.mutated = unpickled[10]
        self.head = unpickled[11]
        self.mutating = unpickled[12]
        self.mutatingpositions = unpickled[13]
        self.systematic = unpickled[14]
        self.goals_var = unpickled[15]
        self.avoids_var = unpickled[16]
        self.blosum_score = unpickled[17]
        self.mutated_aas = unpickled[18]


class Memory:
    """
    Stores the information needed to prevent trying the same sequence multiple times. Calculates the next position,
    amino acid and backmutating position needed for a mutation
    """
    def __init__(self, startseq, seq2mutate, maxmut):
        """
        Initialises the memory with a startseq relative to which the number of mutations is calculated and a
        sequence2mutate, for which position amino acid and backmutating position are calculated
        Args:
            startseq(ndarray): one-hot encoded sequence,
                the original sequence that is used to count the number of mutations
            seq2mutate(ndarrray): one-hot encoded sequence, sequence in which mutations are introduced with the
                with the informations provided by the position() function
            maxmut(int): Maximum number of mutations that is allowed. If a new mutation would result in more mutations,
                a postion to backmutate is provided
        """

        self.startseq = startseq
        self.maxmut = maxmut
        self.seq2mutate = seq2mutate
        self.forward = []
        self.backward = []
        self.aa = 0
        self.break_here = False
        self.backmutate = np.sum(np.abs(seq2mutate[:20, :] - startseq[:20, :]))/2 >= maxmut

        if self.backmutate:
            self.new_tobackmutate()
        self.new_tomutate()

    def position(self):
        """
        Calculates all informations needed to introduce mutations in a function
        Returns:
            aa(int): amino acid, key from id2aa dict. The position is mutated to this residue
            pos(int): position in the sequence that is mutated
            backpos(int): position in the sequence that is backmutated if back is true
            back(bool): true, if backmutation is needed

        """
        aa = self.aa

        if self.backmutate:
            if self.aa == 19:
                self.aa = 0
                self.new_tobackmutate()
            else:
                self.aa += 1

            if self.backmutate:
                pos = self.forward[-1]
                backpos = self.backward[-1]
                back = True
                return aa, pos, backpos, back
        else:
            if self.aa == 19:
                self.aa = 0
                self.new_tomutate()
            else:
                self.aa += 1
            pos = self.forward[-1]
            back = False
            return aa, pos, 0.1, back

    def new_tobackmutate(self):
        """
        Randomly hooses a new postion to backmutate if needed,
            does not take a position that was already used for backmutation with the current mutation position before
        """

        if self.maxmut - len(self.backward) == 1:
            self.new_tomutate()
            self.backward = []
            self.new_tobackmutate()
            return

        else:
            rand = random.randint(1, self.maxmut - len(self.backward))
        locusback = -1
        while rand > 0:
            if np.argmax(self.seq2mutate[:20, locusback + 1]) != np.argmax(self.startseq[:20, locusback +1]): # mutated
                if locusback not in self.backward:
                    rand -= 1
            locusback += 1
        self.backward.append(locusback)

    def new_tomutate(self):
        """
        Randomly hooses a new postion to mutate,
            does not take a position that was already used for mutation before.
                If all mutation positions were tried before, the algorithm will not find something better as all
                combinations of mutations reachable within a single mutation and backmutation were tried.
        """

        if np.sum(self.seq2mutate[20, :]) - len(self.forward) == 1: # 1 # number of positions that are mutable
            self.break_here = True
            return
        else:
            rand = random.randint(1, np.sum(self.seq2mutate[20, :]) - len(self.forward))
        locus = -1
        while rand > 0:
            if self.seq2mutate[20, locus +1] ==1:
                if locus+1 not in self.forward:
                    rand -= 1
            locus += 1
        self.forward.append(locus)


class GeneticAlg:

    def __init__(self, optionhandler, classifier, TFrecordsgenerator):

        """
        Initializes the genetic algorithm with values from a gaia-file, builds the one-hot encoded sequences, the needed
        dicts and stores the provided objects. Initializes the classifier for machine inference

        Reads .gaia file:
        -----------------
        Line:
        1 (optional) starts with '>' contains title of the stats.png plot
        2 '[Float]>GO:[GO-term],[Float]>GO:[GO-term],[...]' contains goal GO-terms with their weight
        3 'Float]>GO:[GO-term],[Float]>GO:[GO-term],[...]' contains avoid GO-terms with their weights
        4 [Sequence to evolve]
        5 'Maxmut: [Integer]' contains maximum number of mutations
        6 'Notgarbage weight: [Float]' contains weight for not_garbage score
        7 'Garbage_weight: [Float]' cotains weight for garbage score

        Args:
          optionhandler(Optionhandler): Optionhandler from helpers.py that stores all relevant options
          classifier(DeeProtein): Classifier that is used to score the sequences. Needs to read a batch of
          one-hot encoded sequences as a three dimensional array
          TFrecordsgenerator(TFRecordsgenerator): from helpers.py, generates the aa2id and id2aa dict,
          the one-hot encoding
        """
        self.opts = optionhandler
        self.TFrecordsgenerator = TFrecordsgenerator
        sys.stderr.write('Working directory: \n{}\n\n\n'.format(self.opts._summariesdir))
        sys.stderr.flush()

        self.seqs = np.zeros([self.opts._batchsize, self.opts._depth + 2, self.opts._windowlength]) # 20: to mutate
        # (1) or not to mutate(0), 21: how often the position has been mutated

        self.classifier = classifier
        self.aa2id = TFrecordsgenerator.AA_to_id
        self.id2aa = TFrecordsgenerator.id_to_AA
        self.class_dict = TFrecordsgenerator.class_dict

        self.logfile = open(os.path.join(self.opts._summariesdir, 'logfile.txt'), 'w')
        self.currpath = os.path.join(self.opts._summariesdir, 'currfile.txt')

        print(os.path.join(self.opts._summariesdir, 'logfile.txt'))
        seqs_file = open(self.opts._seqfile, 'r')

        # save seqs_file
        write_seqs_file = open(os.path.join(self.opts._summariesdir, 'seqfile.txt'), 'w')
        write_seqs_file.write(seqs_file.read())
        seqs_file.seek(0)
        head = seqs_file.readline()
        if head.startswith('>'):
            str_goals = seqs_file.readline().strip().split(',')
        else:
            str_goals = head.strip().split(',')
        str_avoids = seqs_file.readline().strip().split(',')
        self.logfile.write('Read sequence file {}\n'.format(self.opts._seqfile))

        self.logfile.write('Goals: {}\n'.format(str_goals))
        self.logfile.write('Avoids: {}\n'.format(str_avoids))

        self.goal_weights = [float(_class.split('>')[0]) for _class in str_goals]
        self.avoids_weights = [float(_class.split('>')[0]) for _class in str_avoids]

        self.goals = [self.class_dict[_class.split('>')[1]]['id'] for _class in str_goals]
        self.avoids = [self.class_dict[_class.split('>')[1]]['id'] for _class in str_avoids]

        self.stats = Stats(self.opts._summariesdir, str_goals, str_avoids, self.goals, self.avoids, head)

        startseq = seqs_file.readline()

        maxmut_data = seqs_file.readline() # one integer, currently
        self.maxmut = int(maxmut_data.split(':')[1].strip())

        not_garbage_weight_data = seqs_file.readline()
        self.not_garbage_weight = float(not_garbage_weight_data.split(':')[1].strip())

        garbage_weight_data = seqs_file.readline()
        self.garbage_weight = float(garbage_weight_data.split(':')[1].strip())

        for j in range(len(startseq)):  # seq len dim
            if startseq[j].isupper():  # to mutate
                for i in range(self.seqs.shape[0]):  # batch dim
                    self.seqs[i, 20, j] = 1  # to mutate or not to mutate

        self.logfile.write('Maxmut: {}\n'.format(self.maxmut))

        self.logfile.write('Starting sequence:\n{}\n\n'.format(startseq.strip('\n')))

        self.startseq_chars = startseq.upper()

        oh_seq, _, _ = TFrecordsgenerator._seq2tensor(self.startseq_chars.strip('\n'))

        self.seqs[0, :20, :] = oh_seq
        for seqi in range(self.seqs.shape[0]-1):
            self.seqs[seqi+1] = np.copy(self.seqs[0, :, :])

        self.startseq = self.seqs[0, :, :]

        self.memory = None  # for saving parameters in systematic mode

        self.backmutating = False # for output

        weights = np.asarray([
            9.26, 1.13, 5.19, 5.78, 3.81, 7.34, 2.18, 6.03, 4.77, 7.47, 2.64, 4.20, 0.424, 4.30, 5.51, 2.51, 5.53, 7.01,
            0.139, 0.297]) # p/10, w/10, y/10
        #   A,    C,    D,    E,    F,    G,    H,    I,    K,    L,    M,    N,    P,    Q,    R,    S,    T,    V,
        #   W,    Y
        # Data from [GenScript](https://www.genscript.com/tools/codon-frequency-table)

        self.normalized_weights = weights / np.sum(weights)

        self.blosum62 = {
            'C':  {'C':9,  'S':-1, 'T':-1, 'P':-3, 'A':0,  'G':-3, 'N':-3, 'D':-3, 'E':-4, 'Q':-3, 'H':-3, 'R':-3, 'K':-3, 'M':-1, 'I':-1, 'L':-1, 'V':-1, 'F':-2, 'Y':-2, 'W':-2},
            'S':  {'C':-1, 'S':4,  'T':1,  'P':-1, 'A':1,  'G':0,  'N':1,  'D':0,  'E':0,  'Q':0,  'H':-1, 'R':-1, 'K':0,  'M':-1, 'I':-2, 'L':-2, 'V':-2, 'F':-2, 'Y':-2, 'W':-3},
            'T':  {'C':-1, 'S':1,  'T':4,  'P':1,  'A':-1, 'G':1,  'N':0,  'D':1,  'E':0,  'Q':0,  'H':0,  'R':-1, 'K':0,  'M':-1, 'I':-2, 'L':-2, 'V':-2, 'F':-2, 'Y':-2, 'W':-3},
            'P':  {'C':-3, 'S':-1, 'T':1,  'P':7,  'A':-1, 'G':-2, 'N':-1, 'D':-1, 'E':-1, 'Q':-1, 'H':-2, 'R':-2, 'K':-1, 'M':-2, 'I':-3, 'L':-3, 'V':-2, 'F':-4, 'Y':-3, 'W':-4},
            'A':  {'C':0,  'S':1,  'T':-1, 'P':-1, 'A':4,  'G':0,  'N':-1, 'D':-2, 'E':-1, 'Q':-1, 'H':-2, 'R':-1, 'K':-1, 'M':-1, 'I':-1, 'L':-1, 'V':-2, 'F':-2, 'Y':-2, 'W':-3},
            'G':  {'C':-3, 'S':0,  'T':1,  'P':-2, 'A':0,  'G':6,  'N':-2, 'D':-1, 'E':-2, 'Q':-2, 'H':-2, 'R':-2, 'K':-2, 'M':-3, 'I':-4, 'L':-4, 'V':0,  'F':-3, 'Y':-3, 'W':-2},
            'N':  {'C':-3, 'S':1,  'T':0,  'P':-2, 'A':-2, 'G':0,  'N':6,  'D':1,  'E':0,  'Q':0,  'H':-1, 'R':0,  'K':0,  'M':-2, 'I':-3, 'L':-3, 'V':-3, 'F':-3, 'Y':-2, 'W':-4},
            'D':  {'C':-3, 'S':0,  'T':1,  'P':-1, 'A':-2, 'G':-1, 'N':1,  'D':6,  'E':2,  'Q':0,  'H':-1, 'R':-2, 'K':-1, 'M':-3, 'I':-3, 'L':-4, 'V':-3, 'F':-3, 'Y':-3, 'W':-4},
            'E':  {'C':-4, 'S':0,  'T':0,  'P':-1, 'A':-1, 'G':-2, 'N':0,  'D':2,  'E':5,  'Q':2,  'H':0,  'R':0,  'K':1,  'M':-2, 'I':-3, 'L':-3, 'V':-3, 'F':-3, 'Y':-2, 'W':-3},
            'Q':  {'C':-3, 'S':0,  'T':0,  'P':-1, 'A':-1, 'G':-2, 'N':0,  'D':0,  'E':2,  'Q':5,  'H':0,  'R':1,  'K':1,  'M':0,  'I':-3, 'L':-2, 'V':-2, 'F':-3, 'Y':-1, 'W':-2},
            'H':  {'C':-3, 'S':-1, 'T':0,  'P':-2, 'A':-2, 'G':-2, 'N':1,  'D':1,  'E':0,  'Q':0,  'H':8,  'R':0,  'K':-1, 'M':-2, 'I':-3, 'L':-3, 'V':-2, 'F':-1, 'Y':2,  'W':-2},
            'R':  {'C':-3, 'S':-1, 'T':-1, 'P':-2, 'A':-1, 'G':-2, 'N':0,  'D':-2, 'E':0,  'Q':1,  'H':0,  'R':5,  'K':2,  'M':-1, 'I':-3, 'L':-2, 'V':-3, 'F':-3, 'Y':-2, 'W':-3},
            'K':  {'C':-3, 'S':0,  'T':0,  'P':-1, 'A':-1, 'G':-2, 'N':0,  'D':-1, 'E':1,  'Q':1,  'H':-1, 'R':2,  'K':5,  'M':-1, 'I':-3, 'L':-2, 'V':-3, 'F':-3, 'Y':-2, 'W':-3},
            'M':  {'C':-1, 'S':-1, 'T':-1, 'P':-2, 'A':-1, 'G':-3, 'N':-2, 'D':-3, 'E':-2, 'Q':0,  'H':-2, 'R':-1, 'K':-1, 'M':5,  'I':1,  'L':2,  'V':-2, 'F':0,  'Y':-1, 'W':-1},
            'I':  {'C':-1, 'S':-2, 'T':-2, 'P':-3, 'A':-1, 'G':-4, 'N':-3, 'D':-3, 'E':-3, 'Q':-3, 'H':-3, 'R':-3, 'K':-3, 'M':1,  'I':4,  'L':2,  'V':1,  'F':0,  'Y':-1, 'W':-3},
            'L':  {'C':-1, 'S':-2, 'T':-2, 'P':-3, 'A':-1, 'G':-4, 'N':-3, 'D':-4, 'E':-3, 'Q':-2, 'H':-3, 'R':-2, 'K':-2, 'M':2,  'I':2,  'L':4,  'V':3,  'F':0,  'Y':-1, 'W':-2},
            'V':  {'C':-1, 'S':-2, 'T':-2, 'P':-2, 'A':0,  'G':-3, 'N':-3, 'D':-3, 'E':-2, 'Q':-2, 'H':-3, 'R':-3, 'K':-2, 'M':1,  'I':3,  'L':1,  'V':4,  'F':-1, 'Y':-1, 'W':-3},
            'F':  {'C':-2, 'S':-2, 'T':-2, 'P':-4, 'A':-2, 'G':-3, 'N':-3, 'D':-3, 'E':-3, 'Q':-3, 'H':-1, 'R':-3, 'K':-3, 'M':0,  'I':0,  'L':0,  'V':-1, 'F':6,  'Y':3,  'W': 1},
            'Y':  {'C':-2, 'S':-2, 'T':-2, 'P':-3, 'A':-2, 'G':-3, 'N':-2, 'D':-3, 'E':-2, 'Q':-1, 'H':2,  'R':-2, 'K':-2, 'M':-1, 'I':-1, 'L':-1, 'V':-1, 'F':3,  'Y':7,  'W': 2},
            'W':  {'C':-2, 'S':-3, 'T':-3, 'P':-4, 'A':-3, 'G':-2, 'N':-4, 'D':-4, 'E':-3, 'Q':-2, 'H':-2, 'R':-3, 'K':-3, 'M':-1, 'I':-3, 'L':-2, 'V':-3, 'F':1,  'Y':2,  'W':11}}


    def len_seq(self, seq):
        seq_len = 0
        while np.sum(seq[:20, :], axis=0)[seq_len] == 1 and seq_len < seq.shape[1]:
            seq_len += 1
        return seq_len


    def mutated_seq(self, seq):
        """
        Performs the mutation on one sequence, guarantees that no more than self.maxmut mutations are introduced by
            mutating a mutated residue back to its wt amino acid when the number of mutations is too large
        Args:
            seq(ndarray): one-hot encoded sequence that is mutated

        Returns(ndarray): one-hot encoded sequence that was mutated

        """
        rand = random.random()
        aa = -1
        while rand > 0:
            rand -= self.normalized_weights[aa + 1]
            aa += 1

        locus = -1
        mutatables = np.sum(seq[20, :])

        tomutate = random.randint(1, mutatables)

        while tomutate > 0:  # go to random locus that is to mutate
            if seq[20, locus+1] == 1:
                tomutate -= 1
            locus += 1

        # check if the sequence can be mutated further away from the original:

        if np.sum(np.abs(seq[:20, :] - self.startseq[:20, :]))/2 >= self.maxmut:

            if np.argmax(seq[:20, locus]) != np.argmax(self.startseq[:20, locus]): # mutated, still original sequence
                seq[:20, locus] = np.zeros(20)
                seq[aa, locus] = 1  # now a random amino acid
                seq[21, locus] += 1

            else: # another locus has to be backmutated
                self.backmutating = True

                tobackmutate = random.randint(1, self.maxmut)
                locusback = -1
                while tobackmutate > 0:  # go to random locus that is to mutate
                    if np.argmax(seq[:20, locusback + 1]) != np.argmax(self.startseq[:20, locusback + 1]): # mutated
                        tobackmutate -= 1
                    locusback += 1

                seq[:20, locus] = np.zeros(20)
                seq[aa, locus] = 1  # now a random amino acid
                seq[21, locus] += 1

                seq[:20, locusback] = np.zeros(20)
                seq[np.argmax(self.startseq[:20, locusback]), locusback] = 1 # set one where the starting seq is one
                seq[21, locusback] += 1

        else: # there are mutations left, we can mutate it.
            seq[:20, locus] = np.zeros(20)
            seq[aa, locus] = 1  # now a random amino acid
            seq[21, locus] += 1

        return seq


    def mutates_seq(self, seqs2mutate):
        """
        Mutates a set of sequences, preserves the best (which is assumed to be in the first position)
        Args:
            seqs2mutate(ndarray): first dimension is the batch dimension, second and third are the one-hot encoding.
            Sequences to mutate

        Returns(ndarray): Mutated sequences in the same structure as seqs2mutate

        """
        mutatedseqs = np.ndarray(seqs2mutate.shape)
        for seq in range(seqs2mutate.shape[0]):
            if seq == 0: # assumes best element is in first place
                mutatedseqs[seq] = seqs2mutate[seq]
            else:
                mutatedseqs[seq] = self.mutated_seq(seqs2mutate[seq])

        return mutatedseqs


    def systematic_mutation(self, seqs2mutate):
        """
        Introduces mutations in seqs2mutate and keeps in memory, which mutations were already tried.
        Still random positions are chosen, but for each position all amino acids are tried systematically
        Preserves the best sequence. Never returns a sequence with more mutations than specified
        Args:
            seqs2mutate(ndarray): one-hot encoded sequences to mutate
        Returns
            seqs2mutate(ndarray): one-hot encoded mutated sequences
            continue_here(bool): if false, all mutations that can be reached by a single mutation we're tried,
                the algorithm will not find a better one
        """

        if not self.memory: # startseq, lenseq, seq2mutate, maxmut
            self.memory = Memory(self.startseq, seqs2mutate[0, :, :], self.maxmut)

        for seqid in range(1, seqs2mutate.shape[0]-1):

            aa, locus, locusback, backmutate = self.memory.position()

            # mutate
            seqs2mutate[seqid, :20, locus] = np.zeros(20)
            seqs2mutate[seqid, aa % 20, locus] = 1  # new value set
            seqs2mutate[seqid, 21, locus] += 1

            # backmutate
            if backmutate:
                self.backmutating = True
                seqs2mutate[seqid, :20, locusback] = np.zeros(20)
                seqs2mutate[seqid, np.argmax(self.startseq[:20, locusback]), locusback] = 1  # set one
                                                                                        # where the starting seq is one
                seqs2mutate[seqid, 21, locusback] += 1
        continue_here = not self.memory.break_here
        return seqs2mutate, continue_here


    def choose(self, seqs, scores, survivalpop=-1):
        """
        Takes a set of sequences and chooses the best subset according to the provided scores. The subset contains
        Args:
            seqs(ndarray): One-hot encoded sequences, the first dimension is the batch dimension
            scores(list of float): Scores, has to be in the same order as seqs
            survivalpop(list of int): Specifies how many instances of the best, second best, ... are chosen

        Returns:
            outseq(ndarray): One-hot encoded chosen sequences.
                Each sequence occurs multiple times as specified in survivalpop
            buff(ndarray): One-hot encoded input sequences sorted with the scores provided
        """

        if survivalpop == -1:
            survivalpop = self.opts._survivalpop

        assert scores.shape[0] == seqs.shape[0]
        assert len(survivalpop) <= seqs.shape[0]
        assert np.sum(survivalpop) == seqs.shape[0]

        outseqs = np.ndarray(seqs.shape)

        indices = np.array(range(scores.shape[0]))
        buff = sorted(indices, key=lambda x: scores[x], reverse=True)
        survival_indices = buff[:len(survivalpop)]

        ind = 0
        pop = 0
        for survivor_ind in survival_indices:
            for _ in range(survivalpop[pop]):
                outseqs[ind] = np.copy(seqs[survivor_ind])
                ind += 1
            pop += 1

        return outseqs, buff

    def translate_seq(self, seq, treshold=0.6):
        """
        Turns a one-hot encoded sequence into a string. Also supports encodings with a dimension for each amino acid, if
            they use the same aa2id dict. Writes the amino acid if the modus is greater than a threshold. If no amino
            acid is written for the position, an underscore is written.
        Args:
            seq(ndarray): one-hot encoded sequence to be translated
            treshold(float): threshold over which the amino acid is counted.

        Returns(str): The sequence in capital letters
        """

        outstr = ""
        for pos in range(seq.shape[1]):
            if seq[np.argmax(seq[:, pos]), pos] > treshold:
                outstr += self.id2aa[np.argmax(seq[:, pos])]
            else:
                outstr += '_'
        return outstr

    def write_currfile(self, text, gen=0):
        """
        Overwrites a file with current information on the sequence, adds the current time.
        Args:
            text(str): The information to be written, usually the same that is written to the logfile
            gen(int): The generation in which the algorithm is

        """

        with open(self.currpath, 'w') as currfile:
            currfile.write('{}\nGeneration: {}\n{}'.format(str(datetime.now())[:16],gen, text))
            currfile.flush()


    def score(self, seq, classes, classes_variance):
        """
        Calculates a score for a sequence using the logists retrieved from the classifier as well as their variance and
            other information on the sequence, like the blosum score. Weights for goal and avoid classes are taken from
            the gaia file, the blosum weight is normalized with the length of the sequence
        Args:
            seq(ndarray): Sequence to be scored in one-hot encoding
            classes(ndarrray): Mean logits for the sequence from the classifier.
            classes_variance(ndarray): Variance between the logits for the sequence from the classifier

        Returns(float): A score for the sequence

        """
        blosumweight = 2/(11*len(self.startseq_chars))

        scores = np.ndarray((self.seqs.shape[0]))
        blosum_scores = np.ndarray((self.seqs.shape[0]))

        for seq in range(scores.shape[0]):
            scores[seq] = 0
            for goal_id in range(len(self.goal_weights)):
                scores[seq] += self.goal_weights[goal_id] * classes[seq, self.goals[goal_id]] \
                               - (expit(classes_variance[seq, self.goals[goal_id]]))
            for avoid_id in range(len(self.avoids_weights)):
                scores[seq] -= self.avoids_weights[avoid_id] * classes[seq, self.avoids[avoid_id]] \
                               + (expit(classes_variance[seq, self.avoids[avoid_id]]))
            norm = np.sum(self.goal_weights)
            if norm == 0:
                norm = 1
            scores[seq] = scores[seq]/norm
            blosum_scores[seq] = blosumweight * self.blosumscore(self.seqs[seq])
        scores -= blosum_scores
        return scores, blosum_scores

    def blosumscore(self, seq):
        """
        Calculates the sum of the blosum62 matrix entries that correspond to the mutations in seq relative to
            self.starseq
        Args:
            seq(ndarray): contains the sequence that is evaluated

        Returns(float): The sum of the blosum62 entries for the mutations in the given sequence.
        """

        bscore = 0
        tr_seq = self.translate_seq(seq[:20, :])
        for i in range(len(self.startseq_chars) - 1):
                bscore += self.blosum62[tr_seq[i]][self.startseq_chars[i]]

        return bscore


    def mut_residues(self, seq1, seq2=None):
        """
        Compares two sequences and returns a string listing up the differences in amino acid composition of the
         two sequences
        Args:
            seq1 (str): Sequence that is compared to either the second sequence or the starting sequence.
            seq2 (str): Sequence to which the first sequence is compared to (optional).

        Returns (str): Comma seperated information on how many of each amino acid are in the first and in the second
        sequence:
        Mutated:
        <number of occurences of aa1 in sequence 1>, <number of occurences of aa2 in sequence 1>, ...
        to:
        <number of occurences of aa1 in sequence 2>, <number of occurences of aa2 in sequence 2>, ...

        """
        if not seq2:
            seq2 = self.startseq
        str1 = self.translate_seq(seq1[:20, :])
        str2 = self.translate_seq(seq2[:20, :])
        str1dict = {'A': 0,'C': 0,'D': 0,'E': 0,'F': 0,'G': 0,'H': 0,'I': 0,'K': 0,'L': 0,'M': 0,'N': 0,'P': 0,'Q': 0,
                    'R': 0,'S': 0,'T': 0,'V': 0,'W': 0,'Y': 0}
        str2dict = {'A': 0,'C': 0,'D': 0,'E': 0,'F': 0,'G': 0,'H': 0,'I': 0,'K': 0,'L': 0,'M': 0,'N': 0,'P': 0,'Q': 0,
                    'R': 0,'S': 0,'T': 0,'V': 0,'W': 0,'Y': 0}
        mutated = []

        for i in range(len(str1)):
            if str1[i] != str2[i]:
                mutated.append('{}{}{}'.format(str2[i], i+1, str1[i]))
                str1dict[str1[i]] += 1
                str2dict[str2[i]] += 1

        outstr = "Mutations:\n"
        for el in mutated:
            outstr += '{}, '.format(el)

        outstr += "\nMutated:\n"
        for el in str2dict:
            outstr += "{}: {}, ".format(el, str2dict[el])

        outstr += "\nto:\n"
        for el in str1dict:
            outstr += "{}: {}, ".format(el, str1dict[el])

        return outstr


    def evolve(self, generations=-1):
        """
        Coordinates the evolutionary improvement of a sequence.
        Mutates the sequences, scores the sequences, selects sequences. While that happens statistics are updated,
        the logfile is written.
        Args:
            generations (int): overwrites the number of generations specified in the opts object that is generated from
            the config.json file (optional).
        """
        self.logfile.write('start evolving.\n')
        self.logfile.flush()

        if generations == -1:
            generations = self.opts._generations

        classes, classes_variance = self.classifier.machine_infer(self.seqs[:, :20, :])

        self.logfile.write('classified.\n')
        self.logfile.flush()

        goal = [classes[0, goalid] for goalid in self.goals]
        avoiding = [classes[0, avoidid] for avoidid in self.avoids]
        curr_scores, blosum_score = self.score(None, classes,classes_variance)
        output = 'score: {}\nGoal: {}\nAvoiding: {}\n{}\n'.format(
            curr_scores[0], goal, avoiding,
            self.translate_seq(
                self.seqs[0, :20, :]
            )).replace("'", "").replace("[", "").replace("]", "")
        self.logfile.write(output)
        self.logfile.flush()
        self.write_currfile(output)
        curr_best = self.startseq

        decrease = 0
        systematic = 0
        muts_in_this_gen = 1

        self.logfile.write('start loop:\n')
        self.logfile.flush()
        for gen in range(generations):
            self.logfile.write('\n\n<<<<<<<<<<Generation {}:>>>>>>>>>>\n'.format(gen))
            self.logfile.flush()
            self.backmutating = False

            if self.opts._systematic == "True":
                if gen > 5 and np.sum(self.stats.mutating[-5:]) == 0:
                    self.seqs, cont = self.systematic_mutation(self.seqs)
                    systematic = 1
                    if not cont:
                        output = 'Systematically tried all possible mutations to the sequence that can be reached' \
                                           ' by a single mutation with the given ' \
                                           'parameters and this is the best:\n\n{}\n\n'.format(output)
                        self.logfile.write(output)
                        self.logfile.flush()
                        self.write_currfile(output, gen)
                        return
                else:
                    self.seqs = self.mutates_seq(self.seqs)
                    # reset memory as we were successful with the systematic approach
                    self.memory = None
                    systematic = 0

            else:
                muts_in_this_gen = self.opts._muts_per_gen - decrease
                first = True
                for _ in range(self.opts._muts_per_gen - decrease):
                    self.seqs = self.mutates_seq(self.seqs)
                    if first and gen % self.opts._decrease_muts_after_gen == 0 and gen != 0:
                        first = False
                        # debugfile.write('Gen: {}, muts_in_this_gen: {}\n'.format(gen, muts_in_this_gen))
                        if decrease < self.opts._muts_per_gen-1:
                            decrease += 1

            classes, classes_variance = self.classifier.machine_infer(self.seqs[:, :20, :])
            curr_scores, curr_blosum_score = self.score(None, classes, classes_variance)

            bestseq = self.seqs[np.argmax(curr_scores), :, :]
            mutated = np.sum(np.abs(self.startseq[:20, :] - bestseq[:20, :]))/2
            mutated = mutated / self.len_seq(self.startseq)

            mutating = 0 if np.sum(np.abs(curr_best[:20, :] - bestseq[:20, :])) == 0 else 1
            curr_best = bestseq

            mutated_aas = []
            for pos in range(self.seqs.shape[1]):
                if np.argmax(bestseq[:20, pos]) != np.argmax(self.startseq[:20, pos]):
                    mutated_aas.append(pos)

            self.stats.update(classes, curr_scores, mutated, mutating, self.seqs, systematic,
                              classes_variance, curr_blosum_score, mutated_aas)

            if gen % 100 == 0 and gen != 0:
                self.stats.plot_all()
                # self.stats.plotdistoverseq()

                # self.stats.plotdisttooriginal()

            self.seqs, indices = self.choose(self.seqs, curr_scores)

            best = []
            oldseq = None
            for seq in range(self.seqs.shape[0]):
                if not np.array_equal(self.seqs[seq], oldseq):
                    best.append(seq)
                oldseq = self.seqs[seq]

            first = True
            index = 0

            for seq in best:
                goal = [classes[indices[index], goalid] for goalid in self.goals]
                avoiding = [classes[indices[index], avoidid] for avoidid in self.avoids]

                mutated = np.sum(np.abs(self.startseq[:20, :] - self.seqs[seq, :20, :])) / 2
                mutated = mutated / self.len_seq(self.startseq)
                output = 'Systematic mode: {}\n' \
                         'Backmutating: {}\n' \
                         'Mutating: {}\n' \
                         'Mutations in this generation: {}\n' \
                         'score: {}\n' \
                         'Goal: {}\n' \
                         'Avoiding: {}\n' \
                         'Mutated: {}\n' \
                         '{}\n' \
                         '{}\n\n'.format(
                            systematic,
                            self.backmutating,
                            mutating,
                            muts_in_this_gen,
                            curr_scores[indices[index]],
                            goal, avoiding,
                            mutated,
                            self.translate_seq(self.seqs[seq, :20, :]),
                            self.mut_residues(self.seqs[seq, :, :])
                                        ).replace("'", "").replace("[", "").replace("]", "").replace("_", "")
                self.logfile.write(output)
                self.logfile.flush()
                if first:
                    self.write_currfile(output, gen)
                    first = False
        return

    def score_combinations_BINARY(self, ipath, max2combine):
        """
        Calculates the scores for all combinations possible for the mutation sets in the input file
        that have max2combine elements, dumps data as pickle, writes scores with all available information to logfile.
        Args:
            ipath (str): path to input file, in which mutations to be comined are specified:
            each line is treated as one unit in combination,
            if a line contains several mutations these always occur together. The delimiter between two mutations is ','
            A mutation starts with an integer that defines the position and ends with a char that defines the amino acid
            to which the position is mutated.
            max2combine (int): sets the limit on how many lines are combined together.
            Useful to decrease the number of possible combinations.

        """
        self.logfile.write('Scoring combinations from file {}.\n'.format(ipath))
        with open(ipath) as ifile:
            lines = ifile.readlines()
        to_comb = [line.strip().split(', ') for line in lines]
        self.logfile.write('To comb:\n')
        pprint.pprint(to_comb, self.logfile)

        combs = []
        flat_combs = []
        for r in range(min(len(to_comb), max2combine+1)):
           combs.extend(list(itertools.combinations(to_comb, r)))
        combs = list(itertools.chain(combs))
        self.logfile.write('combs: \n')
        pprint.pprint(combs, self.logfile)
        for idx in range(len(combs)):
            flat_combs.append([])
            for el in combs[idx]:
                for subel in el:
                    flat_combs[idx].append(subel)
        self.logfile.write('flat_combs: \n')
        pprint.pprint(flat_combs, self.logfile)

        self.logfile.write('Calculating {} combinations.\n'.format(len(flat_combs)))
        self.logfile.flush()
        data = []
        counter = 0
        for comb in flat_combs:
            if counter % 1000 == 0:
                self.logfile.write('Scored {} % of all combinations so far.\n'.format(counter/len(flat_combs)))
                self.logfile.flush()
            counter += 1
            mutated = self.mutate_specific(comb)
            # seq2classify = np.expand_dims(mutated, axis=0)
            #
            # pprint.pprint(seq2classify, self.logfile)
            # pprint.pprint(mutated.shape, self.logfile)
            # params = (seq2classify, 1)
            # pprint.pprint(len(params), self.logfile)
            # pprint.pprint(len(params[0]), self.logfile)
            # pprint.pprint(params[1], self.logfile)
            # self.logfile.write('done.')
            # self.logfile.flush()

            classes, classes_variance = self.classifier.machine_infer(np.expand_dims(mutated, axis=0))
            goal = []
            goal_var = []
            avoid = []
            avoid_var = []
            for goal_id in range(len(self.goal_weights)):
                goal.append(classes[0, self.goals[goal_id]])
                goal_var.append(classes_variance[0, self.goals[goal_id]])
            for avoid_id in range(len(self.avoids_weights)):
                avoid.append(classes[0, self.avoids[avoid_id]])
                avoid_var.append(classes_variance[0, self.avoids[avoid_id]])

            data.append((comb, goal, goal_var, avoid, avoid_var,
                                                           self.translate_seq(mutated)))
        self.logfile.write('Scored all combinations.\n')
        self.logfile.flush()
        data = sorted(data, key=lambda x: x[1][0], reverse=True)
        self.logfile.write('Sorted all combinations.\n')
        self.logfile.flush()
        pickle.dump(data, open(os.path.join(self.opts._summariesdir, 'comb.p'), 'wb'))
        self.logfile.write('Dumped data as pickle file.\n\n\nData:\n=====\n\n')
        self.logfile.flush()
        for entry in data:
            self.logfile.write('Mutations: {}\n'
                               'Goal scores: {}\n'
                               'Goal variances: {}\n'
                               'Avoid scores: {}\n'
                               'Avoid variances: {}\n'
                               'Sequence: \n{}\n\n'.format(
                entry[0], entry[1], entry[2], entry[3], entry[4], entry[5].replace('_', '')))

    def score_combinations(self, ipath, max2combine):
        """
        Calculates the scores for all combinations possible for the mutation sets in the input file
        that have max2combine elements, dumps data as pickle, writes scores with all available information to logfile.
        Args:
            ipath (str): path to input file, in which mutations to be comined are specified:
            each line is treated as one unit in combination,
            if a line contains several mutations these always occur together. The delimiter between two mutations is ','
            A mutation starts with an integer that defines the position and ends with a char that defines the amino acid
            to which the position is mutated.
            max2combine (int): sets the limit on how many lines are combined together.
            Useful to decrease the number of possible combinations.

        """
        self.logfile.write('Scoring combinations from file {}.\n'.format(ipath))
        with open(ipath) as ifile:
            lines = ifile.readlines()
        to_comb = [line.strip().split(', ') for line in lines]
        self.logfile.write('To comb:\n')
        pprint.pprint(to_comb, self.logfile)

        combs = []
        flat_combs = []
        for r in range(min(len(to_comb), max2combine+1)):
           combs.extend(list(itertools.combinations(to_comb, r)))
        combs = list(itertools.chain(combs))
        self.logfile.write('combs: \n')
        pprint.pprint(combs, self.logfile)
        for idx in range(len(combs)):
            flat_combs.append([])
            for el in combs[idx]:
                for subel in el:
                    flat_combs[idx].append(subel)
        self.logfile.write('flat_combs: \n')
        pprint.pprint(flat_combs, self.logfile)

        self.logfile.write('Calculating {} combinations.\n'.format(len(flat_combs)))
        self.logfile.flush()
        data = []
        counter = 0
        for comb in flat_combs:
            if counter % 1000 == 0:
                self.logfile.write('Scored {} % of all combinations so far.\n'.format(counter/len(flat_combs)))
                self.logfile.flush()
            counter += 1
            mutated = self.mutate_specific(comb)
            # seq2classify = np.expand_dims(mutated, axis=0)
            #
            # pprint.pprint(seq2classify, self.logfile)
            # pprint.pprint(mutated.shape, self.logfile)
            # params = (seq2classify, 1)
            # pprint.pprint(len(params), self.logfile)
            # pprint.pprint(len(params[0]), self.logfile)
            # pprint.pprint(params[1], self.logfile)
            # self.logfile.write('done.')
            # self.logfile.flush()

            classes, classes_variance = self.classifier.machine_infer(np.expand_dims(mutated, axis=0))
            goal = []
            goal_var = []
            avoid = []
            avoid_var = []
            for goal_id in range(len(self.goal_weights)):
                goal.append(classes[0, self.goals[goal_id]])
                goal_var.append(classes_variance[0, self.goals[goal_id]])
            for avoid_id in range(len(self.avoids_weights)):
                avoid.append(classes[0, self.avoids[avoid_id]])
                avoid_var.append(classes_variance[0, self.avoids[avoid_id]])

            data.append((comb, goal, goal_var, avoid, avoid_var,
                                                           self.translate_seq(mutated)))
        self.logfile.write('Scored all combinations.\n')
        self.logfile.flush()
        data = sorted(data, key=lambda x: x[1][0], reverse=True)
        self.logfile.write('Sorted all combinations.\n')
        self.logfile.flush()
        pickle.dump(data, open(os.path.join(self.opts._summariesdir, 'comb.p'), 'wb'))
        self.logfile.write('Dumped data as pickle file.\n\n\nData:\n=====\n\n')
        self.logfile.flush()
        for entry in data:
            self.logfile.write('Mutations: {}\n'
                               'Goal scores: {}\n'
                               'Goal variances: {}\n'
                               'Avoid scores: {}\n'
                               'Avoid variances: {}\n'
                               'Sequence: \n{}\n\n'.format(
                entry[0], entry[1], entry[2], entry[3], entry[4], entry[5].replace('_', '')))


    def mutate_specific(self, muts, seq=None):
        """
        Introduced a specific mutation in a given sequence or the starting sequence if no sequence is provided.
        Args:
            muts (list of str): A list of strings that specifiy the mutations that shall be made in the sequence,
                each element starts with an integer that specifies the position to mutatated and ands with a char that
                defines to which amino acid the position is mutated.
            seq (ndarray([20, sequence length]): sequence, in which the given mutations are introduced (optional).

        Returns (ndarray([20, sequence length]): sequence that has the specified mutations

        """

        if seq == None:
            seq = self.startseq[:20, :].copy()
        else:
            seq, _, _ = self.TFrecordsgenerator._seq2tensor(seq.strip('\n'))
        # sequence is now in one-hot encoding
        aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        for mut in muts:
            seq[:, int(mut[:-1])-1] = np.zeros(20) # reset
            seq[aas.index(mut[-1:]), int(mut[:-1])-1] = 1

        return seq


    def count_muts(self, seq):
        """
        Counts mutations compared to starting sequence
        Args:
            seq(ndarray([20, sequence length]): sequence that is compared to starting sequence

        Returns (int): number of mutations in sequence relative to starting sequence
        """
        return np.sum(np.abs(self.startseq[:20, :] - seq[:20, :])) / 2


    def randomize(self, until):
        """
        Works only with goal class!
        Calculates random mutants with x mutated amino acids and their scores, with x being each number from 0 to until
        dumps data as pickle and calls plotting functions.
        Args:
            until (int): specifies the maximum number of mutations the sequence shall have
        """

        lever = 100
        self.maxmut = 1000
        self.logfile.write('Randomizing mutations.\n')
        self.logfile.flush()

        data = []
        reduced = []
        seqs = [self.seqs.copy() for _ in range(lever)]

        for muts in range(until):
            self.logfile.write('{} out of {} mutations.\n'.format(muts, until))
            self.logfile.flush()
            data_one_mut = []
            reduced_one_mut = []
            for idx in range(seqs[0].shape[0]):
                for i in range(lever):
                    while self.count_muts(seqs[i][idx, :, :]) < muts:
                        seqs[i][idx, :, :] = self.mutated_seq(seqs[i][idx, :, :])

            # now we have 100 sequences that have exactly muts mutations
            data_one_mut.append(muts)
            data_one_mut.append(seqs)

            # classify:
            goal = []
            goal_var = []
            avoid = []
            avoid_var = []
            for i in range(lever):
                classes, classes_variance = self.classifier.machine_infer(seqs[i][:, :20, :], self.opts._batchsize)
                for idx in range(seqs[i].shape[0]):
                    for goal_id in range(len(self.goal_weights)):
                        goal.append(classes[idx, self.goals[goal_id]])
                        goal_var.append(classes_variance[idx, self.goals[goal_id]])
                    for avoid_id in range(len(self.avoids_weights)):
                        avoid.append(classes[idx, self.avoids[avoid_id]])
                        avoid_var.append(classes_variance[idx, self.avoids[avoid_id]])

            data_one_mut.append(goal)
            data_one_mut.append(goal_var)
            data_one_mut.append(avoid)
            data_one_mut.append(avoid_var)

            reduced_one_mut.append(muts)
            reduced_one_mut.append(np.mean(goal, axis=0))
            reduced_one_mut.append(np.var(goal, axis=0))
            reduced_one_mut.append(np.mean(goal_var, axis=0))
            reduced_one_mut.append(np.var(goal_var, axis=0))
            reduced_one_mut.append(np.mean(avoid, axis=0))
            reduced_one_mut.append(np.var(avoid, axis=0))
            reduced_one_mut.append(np.mean(avoid_var, axis=0))
            reduced_one_mut.append(np.var(avoid_var, axis=0))

            data.append(data_one_mut)
            reduced.append(reduced_one_mut)

        pickle.dump(data, open(os.path.join(self.opts._summariesdir, 'data.p'), 'wb'))
        pickle.dump(reduced, open(os.path.join(self.opts._summariesdir, 'reduced.p'), 'wb'))
        self.logfile.write('Dumped picklefile.\n')
        self.logfile.flush()

        plot_data(reduced, self.opts._summariesdir, self.logfile)


    def randomize_all(self, until):
        """
        Works with all classes!
        Calculates random mutants with x mutated amino acids and their scores, with x being each number from 0 to until
        dumps data as pickle and calls plotting functions.
        Args:
            until (int): specifies the maximum number of mutations the sequence shall have
        """

        lever = 100
        self.maxmut = 1000
        self.logfile.write('Randomizing mutations with all classes.\n')
        self.logfile.flush()

        data = []
        reduced = []
        seqs = [self.seqs.copy() for _ in range(lever)]

        for muts in range(until):
            self.logfile.write('{} out of {} mutations.\n'.format(muts, until))
            self.logfile.flush()
            data_one_mut = []
            reduced_one_mut = []
            for idx in range(seqs[0].shape[0]):
                for i in range(lever):
                    while self.count_muts(seqs[i][idx, :, :]) < muts:
                        seqs[i][idx, :, :] = self.mutated_seq(seqs[i][idx, :, :])

            # now we have 100 sequences that have exactly muts mutations
            data_one_mut.append(muts)
            data_one_mut.append(seqs)

            # classify:
            goal = []
            goal_var = []
            avoid = []
            avoid_var = []
            for i in range(lever):
                classes, classes_variance = self.classifier.machine_infer(seqs[i][:, :20, :])
                for idx in range(seqs[i].shape[0]):
                    for goal_id in range(len(self.goal_weights)):
                        goal.append(classes[idx, :])
                        goal_var.append(classes_variance[idx, :])

            data_one_mut.append(goal)
            data_one_mut.append(goal_var)

            reduced_one_mut.append(muts)
            reduced_one_mut.append(np.mean(goal, axis=0))
            reduced_one_mut.append(np.var(goal, axis=0))
            reduced_one_mut.append(np.mean(goal_var, axis=0))
            reduced_one_mut.append(np.var(goal_var, axis=0))

            data.append(data_one_mut)
            reduced.append(reduced_one_mut)

        pickle.dump(data, open(os.path.join(self.opts._summariesdir, 'data.p'), 'wb'))
        pickle.dump(reduced, open(os.path.join(self.opts._summariesdir, 'reduced.p'), 'wb'))
        self.logfile.write('Dumped pickle.\n')
        self.logfile.flush()

        to_highlight = ['GO:0004818', 'GO:0004827']
        highl = [self.class_dict[i]['id'] for i in to_highlight]
        plot_all_data(reduced, self.goals[0], self.opts._summariesdir, self.logfile, highl=highl,
                      highl_names=to_highlight)


    def walk(self, highl=[100000], highl_names=['Default'], f_width=10, f_height=8,
                          res=80, name='walk_data.png'):
        """
        Calculates all possible single mutants of a sequence, makes data available to plot, calls plotting functions
            and dumps pickle files
        Args:
            highl (list of int): passed to plotting functions, indicates which classes shall be highlighted
            highl_names (list of str): passed to plotting funcitons, speciefies a name for each element in highl
            f_width (float): passed to plotting functions, specifies the width of the plots that are written,
                except for plots that show position specific information
            f_height (float): passed to plotting functions, specifies the height of the plots that are written
            res (float): passed to plotting functions, specifies the resolution of the plots that are written
            name: passed to plotting functions, specifies the name of plots that are written.
                Prefixes or suffixes are added

        """

        data = []
        wt_classes, wt_classes_variance = self.classifier.machine_infer(self.seqs[:, :20, :])

        for pos in range(len(self.startseq_chars)):
            scores = []
            scores_mean = []
            scores_var = []
            variances = []
            variances_mean = []
            variances_var = []

            seqs = self.seqs.copy()
            for aa in range(20):
                seqs[aa, :20, pos] = np.zeros(20)  # reset
                seqs[aa, aa, pos] = 1
            classes, classes_variance = self.classifier.machine_infer(seqs[:, :20, :])
            scores.append(classes)
            variances.append(classes_variance)

            scores_mean.append(np.mean(classes, axis=0))
            scores_var.append(np.var(classes, axis=0))
            variances.append(np.mean(classes_variance, axis=0))
            variances_var.append(np.var(classes_variance, axis=0))

            datapoint = [scores, variances, scores_mean, scores_var, variances_mean, variances_var, self.startseq_chars]
            data.append(datapoint)

        pickle.dump(data, open(os.path.join(self.opts._summariesdir, 'walk_data.p'), 'wb'))
        pickle.dump(wt_classes[:, 0], open(os.path.join(self.opts._summariesdir, 'wt_scores.p'), 'wb'))
        to_highlight = ['GO:0004818', 'GO:0004827']
        highl = [self.class_dict[i]['id'] for i in to_highlight]
        plot_walk(data, self.goals[0], self.opts._summariesdir, self.logfile, wt_scores=wt_classes[0, :],
                  highl=highl, highl_names=to_highlight, name='walk_data.png')


def draw_heatmap(scores, goal_class, wt_scores, startseq_chars, wdir, logfile, f_width=50, f_height=4, res=300):
    """
    Draws a heatmap with the sequence on x-axis and amino acid to which a specific position is mutated on y-axis.
    The color indicates the score of the mutant defines by sequence position and amino acid
    Args:
        scores (ndarray([20, sequence length, number of classes]): holds the data that is plottet
        goal_class (int): Speciefies a class for which the data is plottet
        wt_scores (ndarray([number of classes]): holds the scores on the unchanged sequences to compare to
        startseq_chars (str): Specifies the sequence that is written on the bottom of the plot
        wdir (str): directory in which plots are written
        logfile (open writable file): file to which logs are written
        f_width (float): specifies the width of the plots that are written,
            except for plots that show position specific information
        f_height (float): specifies the height of the plots that are written
        res (float): specifies the resolution of the plots that are written

    """

    ccmap = mplcolors.LinearSegmentedColormap('by_cmap', cdict)
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    fig, ax = plt.subplots(figsize=(f_width, f_height), dpi=res)

    for idx in range(scores.shape[0]):
        for idy in range(scores.shape[1]):
            scores[idx, idy, goal_class] = scores[idx, idy, goal_class] - wt_scores[goal_class]


    pcm = ax.imshow(scores[:, :-1, goal_class], origin='lower',
                    aspect='auto',
                    cmap = ccmap)
    clb = fig.colorbar(pcm, ax=ax)
    clb.set_label('Mutant score', y=0.5)

    xtickvals = []
    xtickvalsminor = []
    ytickvals = []
    xticks = []
    xticksminor = []
    yticks = []

    for pos in range(scores.shape[1]-1):
        xtickvalsminor.append(startseq_chars[pos])
        xticksminor.append(pos)
        xtickvals.append(pos + 1) # 1-based counting for sequence position
        xticks.append(pos)
    for aa in range(20):
        ytickvals.append(aas[aa])
        yticks.append(aa)

   # plt.title('Predicted score for amino acid exchanges for label {}'.format(goal_class))
    plt.ylabel('Amino Acid')
    #plt.xlabel('Sequence Position')

    plt.tick_params(axis='y',
                    which='both',
                    left='off')
    plt.tick_params(axis='x',
                    which='major',
                    bottom='off',
                    labelbottom='on',
                    labelsize=4,
                    pad=15)
    ax.tick_params( axis='x',
                    which='minor',
                    bottom='off',
                    labelbottom='on')

    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels(xtickvals, minor=False, rotation='vertical')

    ax.set_xticklabels(xtickvalsminor, minor=True)
    ax.set_xticks(xticksminor, minor=True)

    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(ytickvals, minor=False, rotation=0)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    plt.savefig(os.path.join(wdir, 'walk_heatmap_goal-{}'.format(goal_class)))
    plt.close()


def plot_walk(data, goal, wdir, logfile, wt_scores=None, highl=[10000], highl_names=['Default'], aas = [0], f_width=5, f_height=4,
                          res=300, name='walk_data_HD.png'):

    """Plots graphs indicating how the scores change, when a specific position is mutated.
    Args:
        data (list): that stores all relevant data, either from method walk or from saved pickle
        goal (integer): Index of the goal in GeneticAlg.class_dict
        wdir (str): directory to write the plots in
        logfile (open writable file): file to write logs to
        wt_scores (ndarray([number_classes]): an array of scores to compare mutants to. Needed to write the 'sub'-plots
        highl (list of int): a list of indices from GeneticAlg.class_dict for classes that should be highlighted in plots
        highl_names (list of str): a list of Names for the classes that should be highlighted.
        aas (list): a list of amino acids to which the positions are mutated
        f_width (float): specifies the width of the plots that are written, except for plots that show position specific information
        f_height (float): specifies the height of the plots that are written
        res (float): specifies the resolution of the plots that are written
        name (str): specifies the name of plots that are written. Prefixes or suffixes are added
    """
    pprint.pprint(highl, logfile)
    pprint.pprint(goal, logfile)
    pprint.pprint(aas, logfile)

    plt.switch_backend('agg')
    number_classes = len(data[0][0][0][0])
    logfile.write('number of classes: {}\n'.format (number_classes))

    scores_mean = [[] for _ in range (number_classes)]
    scores_var = [[] for _ in range(number_classes)]
    variances_mean = [[] for _ in range(number_classes)]
    variances_var = [[] for _ in range(number_classes)]

    scores = [[[] for _ in range(number_classes)] for _ in range(20)]
    variances = [[[] for _ in range(number_classes)] for _ in range(20)]

    scores_heatmap = np.ndarray([20, len(data), number_classes])

    pos = 0
    for datapoint in data:
        for score in range(number_classes):
            scores_mean[score].append(datapoint[2][0][score])
            scores_var[score].append(datapoint[3][0][score])
            # variances_mean[score].append(datapoint[4][score])
            # variances_var[score].append(datapoint[5][score])
            for aa in range(20):
                scores[aa][score].append(datapoint[0][0][aa, score])
                # variances[aa][score].append(datapoint[1][aa, score])
                pprint.pprint(datapoint[0][0][aa][score], logfile)
                scores_heatmap[aa, pos, score] = datapoint[0][0][aa][score]
        pos += 1
    pos = range(len(data))

    draw_heatmap(scores_heatmap, goal, wt_scores=wt_scores, wdir=wdir, logfile=logfile, startseq_chars=data[0][-1])

    # mean plots:

    plt.figure(1, figsize=(f_width, f_height), dpi=res)
    ax = plt.subplot(111)
    # ax2 = ax.twinx()

    for i in range(number_classes):
        if i == goal:
            lower = np.asarray(scores_mean[i]) + np.asarray(scores_var[i]).tolist()
            upper = np.asarray(scores_mean[i]) - np.asarray(scores_var[i]).tolist()
            p1 = ax.plot(pos, scores_mean[i], label='Goal', zorder=1000, color='#9D1C20', linewidth=0.25)
            # p2 = ax.plot(pos, lower, label='Goal - sigma', color='#BB5651')
            # p3 = ax.plot(pos, upper, label='Goal + sigma', color='#BB5651')
            # ax.fill_between(pos, lower, upper, facecolor='#D89F9C')
        else:
            if i in highl:
                lower = np.asarray(scores_mean[i]) + np.asarray(scores_var[i]).tolist()
                upper = np.asarray(scores_mean[i]) - np.asarray(scores_var[i]).tolist()
                p4 = ax.plot(pos, scores_mean[i], label=highl_names[0] + ",\n" + highl_names[1], zorder=999,
                             color='#005493', linewidth=0.25)
                # p5 = ax.plot(pos, lower, label='Goal - sigma', color='#6698BE')
                # p6 = ax.plot(pos, upper, label='Goal + sigma', color='#6698BE')
                # ax.fill_between(pos, lower, upper, facecolor='#B2CBDD')
            else:
                p7 = ax.plot(pos, scores_mean[i], label='Alt', color='#009193', linewidth=0.25)


    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim([0, 1.2 * np.max(scores_mean)])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height * 0.9])

    try:
        axes = p1 + p4 +p7
    except:
        axes = p1 + p7

    legs = [a.get_label() for a in axes]
    ax.legend(axes, legs, loc=0, bbox_to_anchor=(1.0, 0.5), prop=font)

    ax.set_ylabel('Score', fontproperties=font)
    plt.xlabel('Mutations', fontproperties=font)
    plt.title('Comparison of different mutation rates and scores', fontproperties=font)
    plt.savefig(os.path.join(wdir, 'means{}'.format(name)))
    plt.gcf().clear()
    logfile.write('Plotted means.\n')
    logfile.flush()

    # aa specific plots:
    for aa in aas:
        plt.figure(1, figsize=(f_width, f_height), dpi=res)
        ax = plt.subplot(111)
        # ax2 = ax.twinx()

        for i in range(len(scores_mean)):
            if i == goal:
                lower = np.asarray(scores[aa][i]) + np.asarray(scores_var[i]).tolist()
                upper = np.asarray(scores[aa][i]) - np.asarray(scores_var[i]).tolist()
                p1 = ax.plot(pos, scores[aa][i], label='Goal', zorder=1000, color='#9D1C20', linewidth=0.25)
                # p2 = ax.plot(pos, lower, label='Goal - sigma', color='#BB5651')
                # p3 = ax.plot(pos, upper, label='Goal + sigma', color='#BB5651')
                # ax.fill_between(pos, lower, upper, facecolor='#D89F9C')
            else:
                if i in highl:
                    p4 = []
                    lower = np.asarray(scores[aa][i]) + np.asarray(scores_var[i]).tolist()
                    upper = np.asarray(scores[aa][i]) - np.asarray(scores_var[i]).tolist()
                    p4 = ax.plot(pos, scores[aa][i], label=highl_names[0] + ",\n" + highl_names[1], zorder=999,
                                 color='#005493', linewidth=0.25)
                   # p5 = ax.plot(pos, lower, label='Goal - sigma', color='#6698BE')
                   # p6 = ax.plot(pos, upper, label='Goal + sigma', color='#6698BE')
                   # ax.fill_between(pos, lower, upper, facecolor='#B2CBDD')
                else:
                    p7 = ax.plot(pos, scores[aa][i], label='Alt', color='#009193', linewidth=0.25)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # ax.set_ylim([0, 1.2 * np.max(scores_mean)])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height * 0.9])

        try:
            axes = p1 + p4 + p7
        except:
            axes = p1 + p7

        legs = [a.get_label() for a in axes]
        ax.legend(axes, legs, loc=0, bbox_to_anchor=(1.0, 0.5), prop=font)

        ax.set_ylabel('Score', fontproperties=font)
        plt.xlabel('Mutations', fontproperties=font)
        plt.title('Comparison of different mutation rates and scores', fontproperties=font)
        plt.savefig(os.path.join(wdir, 'means_{}.png'.format(aa)))
        plt.gcf().clear()
        logfile.write('Plottet for {}.\n'.format(aa))
        logfile.flush()

    if wt_scores is not None: # each subtracted with the wt_score

        plt.figure(2, figsize=(20, f_height), dpi=res)
        ax = plt.subplot(111)

        for i in range(number_classes):
            sub_mean = (np.asarray(scores_mean[i]) - np.asarray(wt_scores[i])).tolist()
            if i == goal:
                p1 = ax.plot(pos, sub_mean, label='Goal', zorder=1000, color='#9D1C20', linewidth=0.25)
            else:
                if i in highl:
                    p4 = ax.plot(pos, sub_mean, label=highl_names[0] + ",\n" + highl_names[1], zorder=999,
                                 color='#005493', linewidth=0.25)
                else:
                    p7 = ax.plot(pos, sub_mean, label='Alt', color='#B3DEDE', linewidth=0.25)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.9])
        for pos in range(len(data[0][-1])):
            plt.text(pos, 0, data[0][-1][pos], fontsize=4, fontproperties=monospaced)

        try:
            axes = p1 + p4 + p7
        except:
            axes = p1 + p7

        legs = [a.get_label() for a in axes]
        #ax.legend(axes, legs, loc=0, bbox_to_anchor=(1.0, 0.5), prop=font)
        ax.legend(axes, legs, loc=0, prop=font)
        ax.set_ylabel('Score', fontproperties=font)
        plt.xlabel('Mutations', fontproperties=font)
        plt.title('Comparison of different mutation rates and scores', fontproperties=font)
        plt.savefig(os.path.join(wdir, 'means_sub.png'))
        plt.gcf().clear()
        logfile.write('Plotted means.\n')
        logfile.flush()

        # aa specific plots:
        pos = range(len(data))

        for aa in aas:
            plt.figure(1, figsize=(f_width, f_height), dpi=res)
            ax = plt.subplot(111)
            for i in range(len(scores_mean)):
                pprint.pprint(scores[aa][i], logfile)
                #  logfile.write('wt_scores\n')
                #  pprint.pprint(wt_scores, logfile)
                sub_mean = (np.asarray(scores[aa][i]) - np.asarray(wt_scores[i])).tolist()
                #  logfile.write('submean\n')
                #  pprint.pprint(sub_mean, logfile)
                #  logfile.write('Pos:\n')
                #  pprint.pprint(pos, logfile)
                if i == goal:
                    p1 = ax.plot(pos, sub_mean, label='Goal', zorder=1000, color='#9D1C20', linewidth=0.25)
                else:
                    if i in highl:
                        p4 = ax.plot(pos, sub_mean, label=highl_names[0] + ",\n" + highl_names[1], zorder=999,
                                     color='#005493', linewidth=0.25)
                    else:
                        p7 = ax.plot(pos, sub_mean, label='Alt', color='#009193', linewidth=0.25)

            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            # ax.set_ylim([0, 1.2 * np.max(scores_mean)])

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height * 0.9])

            try:
                axes = p1 + p4 + p7
            except:
                axes = p1 + p7

            legs = [a.get_label() for a in axes]
            ax.legend(axes, legs, loc=0, bbox_to_anchor=(1.0, 0.5), prop=font)

            ax.set_ylabel('Score', fontproperties=font)
            plt.xlabel('Mutations', fontproperties=font)
            plt.title('Comparison of different mutation rates and scores', fontproperties=font)
            plt.savefig(os.path.join(wdir, 'means_sub_{}.png'.format(aa)))
            plt.gcf().clear()
            logfile.write('Plottet for {}.\n'.format(aa))
            logfile.flush()


    logfile.write('Plotted aas.\n')
    logfile.flush()


    logfile.write('Wrote plots.\n')
    logfile.flush()



def plot_all_data(data, goal, wdir, logfile, highl=[100000], highl_names=['Default'], f_width=10, f_height=8, res=80, name='data.png'):
    """
        Plots all data from randomize(). x-axis is the number of mutations the sequence has,
            y-axis is the score +/- standard deviation. Highlights the goal score and the highlight scores in different
            colors. Plots all other scores in a light color. Does not plot the standard deviation.
        Args:
            data(list of arrays): The basis on which the plot is calculated
            wdir(str): A directory to write the plots to
            logfile(open writable file): A file to write los in
            highl(list of int): A list of class indices in the class_dict to highlight
            highl_names(list of float): A list of names for the highlight classes
            f_width (float): specifies the width of the plots that are written, except for plots that show position specific information
            f_height (float): specifies the height of the plots that are written
            res (float): specifies the resolution of the plots that are written
            name (str): specifies the name of plots that are written. Prefixes or suffixes are added
        """

    pprint.pprint(highl_names, logfile)
    pprint.pprint(highl, logfile)
    pprint.pprint(goal, logfile)
    plt.switch_backend('agg')
    number_classes = len(data[0][1])
    muts = []
    goal_mean = [[] for _ in range(number_classes)]
    goal_var = [[] for _ in range(number_classes)]

    for datapoint in data:
        muts.append(datapoint[0])
        for score in range(number_classes):
            goal_mean[score].append(datapoint[1][score])
            goal_var[score].append((datapoint[2][score]))

    plt.figure(1, figsize=(f_width, f_height), dpi=res, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    # ax2 = ax.twinx()
    for i in range(len(goal_mean)):
        if i == goal:
            lower = np.asarray(goal_mean[i]) + np.asarray(goal_var[i]).tolist()
            upper = np.asarray(goal_mean[i]) - np.asarray(goal_var[i]).tolist()
            p1 = ax.plot(muts, goal_mean[i], label='Goal', zorder=1000, color='#9D1C20', linewidth=0.25)
            p2 = ax.plot(muts, lower, label='Goal - sigma', color='#BB5651')
            p3 = ax.plot(muts, upper, label='Goal + sigma', color='#BB5651')
            ax.fill_between(muts, lower, upper, facecolor='#D89F9C')
        else:
            if i in highl:
                p4 = []
                lower = np.asarray(goal_mean[i]) + np.asarray(goal_var[i]).tolist()
                upper = np.asarray(goal_mean[i]) - np.asarray(goal_var[i]).tolist()
                p4 = ax.plot(muts, goal_mean[i], label=highl_names[0] + ",\n" + highl_names[1], zorder=999, color='#005493', linewidth=0.25)
                p5 = ax.plot(muts, lower, label='Goal - sigma', color='#6698BE')
                p6 = ax.plot(muts, upper, label='Goal + sigma', color='#6698BE')
                ax.fill_between(muts, lower, upper, facecolor='#B2CBDD')
            else:
                p7 = ax.plot(muts, goal_mean[i], label='Alt', color='#009193', linewidth=0.25)
                # p8 = ax.plot(muts, lower, label='Goal - sigma', color='#66BDBE')
                # p9 = ax.plot(muts, upper, label='Goal + sigma', color='#66BDBE')
                # ax.fill_between(muts, lower, upper, facecolor='#B3DEDE')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim([0, 1.2 * np.max(goal_mean)])

    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.set_position([box.x0, box.y0, box.width*0.7, box.height*0.9])

    axes = p1 + p4 + p7
    legs = [a.get_label() for a in axes]
    ax.legend(axes, legs, loc=0, bbox_to_anchor=(1.0, 0.5), prop=font)
    # ax.legend(axes, legs, loc=0, prop=font)

    # plt.gcf().text(0.8, 0.2, 'Generation: {}'.format(len(self.mutated) - 1))
    ax.set_ylabel('Score', fontproperties=font)
    plt.xlabel('Mutations', fontproperties=font)
    plt.title('Comparison of different mutation rates and scores', fontproperties=font)
    plt.savefig(os.path.join(wdir, name))
    plt.gcf().clear()
    logfile.write('Wrote plots.\n')
    logfile.flush()


def plot_data(data, wdir, logfile, f_width=10, f_height=8, res=80, name='data.png'):
    """
    Plots data from randomize(). x-axis is the number of mutations the sequence has,
        y-axis is the score +/- standard deviation. Only plots the goal class with mean score and standard deviation
    Args:
        data(list of arrays): The basis on which the plot is calculated
        wdir(str): A directory to write the plots to
        logfile(open writable file): A file to write los in
        f_width (float): specifies the width of the plots that are written, except for plots that show position specific information
        f_height (float): specifies the height of the plots that are written
        res (float): specifies the resolution of the plots that are written
        name (str): specifies the name of plots that are written. Prefixes or suffixes are added

    """

    plt.switch_backend('agg')
    muts = []
    goal_mean = []
    goal_var = []
    goal_var_mean = []
    goal_var_var = []
    avoid_mean = []
    avoid_var = []
    avoid_var_mean = []
    avoid_var_var = []

    for datapoint in data:
        muts.append(datapoint[0])
        goal_mean.append(datapoint[1])
        goal_var.append((datapoint[2]))
        goal_var_mean.append(datapoint[3])
        goal_var_var.append(datapoint[4])

        avoid_mean.append(datapoint[5])
        avoid_var.append(datapoint[6])
        avoid_var_mean.append(datapoint[7])
        avoid_var_var.append(datapoint[8])

    plt.figure(1, figsize=(f_width, f_height), dpi=res, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    ax2 = ax.twinx()
    p1 = ax.plot(muts, goal_mean, label='Goal', zorder=1, color='#005493')
    lower = np.asarray(goal_mean) + np.asarray(goal_var).tolist()
    upper = np.asarray(goal_mean) - np.asarray(goal_var).tolist()
    p2 = ax.plot(muts, lower, label='Goal - sigma', color='#6698BE')
    p3 = ax.plot(muts, upper, label='Goal + sigma', color='#6698BE')
    ax.fill_between(muts, lower, upper, facecolor='#B2CBDD')

    p4 = ax2.plot(muts, goal_var_mean, label='Goal_var', color='#009193')

    lower = np.asarray(goal_var_mean) + np.asarray(goal_var_var).tolist()
    upper = np.asarray(goal_var_mean) - np.asarray(goal_var_var).tolist()
    p5 = ax2.plot(muts, lower, label='Goal_var - sigma', color='#66BDBE')
    p6 = ax2.plot(muts, upper, label='Goal_var + sigma', color='#66BDBE')
    ax2.fill_between(muts, lower, upper, facecolor='#B3DEDE')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    axes = p1 + p2 + p3 + p4 + p5 + p6
    legs = [a.get_label() for a in axes]
    ax.legend(axes, legs, loc=0, bbox_to_anchor=(1.4, 0.5))

    # plt.gcf().text(0.8, 0.2, 'Generation: {}'.format(len(self.mutated) - 1))
    ax.set_ylabel('Score')
    ax2.set_ylabel('Variance')
    plt.xlabel('Mutations')
    plt.title('Comparison of different mutation rates and scores')
    plt.savefig(os.path.join(wdir, name))
    plt.gcf().clear()
    logfile.write('Wrote plots.\n')
    logfile.flush()
