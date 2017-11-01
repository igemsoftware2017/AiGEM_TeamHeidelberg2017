## Prerequisities

- Python 3.5+
- [Tensorflow 1.2+](https://www.tensorflow.org/api_docs/)
- [Tensorlayer 1.5.4+](http://tensorlayer.readthedocs.io/en/latest/)
- [GO-Tools](https://github.com/tanghaibao/goatools)
- [SciPy](http://www.scipy.org/install.html)

## Description

GAIA is a genetic algorithm interfaced with a deep neural network as scoring function. It is part of the AiGEM (Artificial Intelligence for Genetic Evolution Mimicking) software suite of the Heidelberg 2017 iGEM Team.

A description, along with data for the software's wetlab validation can be found on [our wiki](http://2017.igem.org/Team:Heidelberg/Software).

## Usage

1. Clone this git repository:
   ```bash
   $ git clone https://github.com/igemsoftware2017/AiGEM_TeamHeidelberg2017 && cd AiGEM_TeamHeidelberg2017/GAIA
   ```

2. To a sequence with the pretrained model:
   Make sure to download the weigths from our [zenodo repository](https://doi.org/10.5281/zenodo.1035806) and to edit the ```config_dict.JSON``` accordingly.
   
   Create a sequence.gaia file in the following format:
   ``` Line:
        1 (optional) starts with '>' contains title of the stats.png plot
        2 '[Float]>GO:[GO-term],[Float]>GO:[GO-term],[...]' contains goal GO-terms with their weight
        3 'Float]>GO:[GO-term],[Float]>GO:[GO-term],[...]' contains avoid GO-terms with their weights
        4 [Sequence to evolve]
        5 'Maxmut: [Integer]' contains maximum number of mutations
        6 'Notgarbage weight: [Float]' contains weight for not_garbage score
        7 'Garbage_weight: [Float]' cotains weight for garbage score''')
   ```
   
   Then run:
   ```bash
   $ python evolve.py --config_json=path/to/config.JSON --sequence=path/to/sequence/file
   ```
3. To plot the GAIA outputs run:
   ```bash
   $ python plot_evolved.py path/to/summaries_dir 
   ```
   
## Documentation

Help on module GAIA:

NAME
    gaia

CLASSES
    builtins.object
        GeneticAlg
        Memory
        Stats
    
    class GeneticAlg(builtins.object)
     |  Methods defined here:
     |  
     |  __init__(self, optionhandler, classifier, TFrecordsgenerator)
     |      Initializes the genetic algorithm with values from a gaia-file, builds the one-hot encoded sequences, the needed
     |      dicts and stores the provided objects. Initializes the classifier for machine inference
     |      
     |      Reads .gaia file:
     |      -----------------
     |      Line:
     |      1 (optional) starts with '>' contains title of the stats.png plot
     |      2 '[Float]>GO:[GO-term],[Float]>GO:[GO-term],[...]' contains goal GO-terms with their weight
     |      3 'Float]>GO:[GO-term],[Float]>GO:[GO-term],[...]' contains avoid GO-terms with their weights
     |      4 [Sequence to evolve]
     |      5 'Maxmut: [Integer]' contains maximum number of mutations
     |      6 'Notgarbage weight: [Float]' contains weight for not_garbage score
     |      7 'Garbage_weight: [Float]' cotains weight for garbage score
     |      
     |      Args:
     |        optionhandler(Optionhandler): Optionhandler from helpers.py that stores all relevant options
     |        classifier(DeeProtein): Classifier that is used to score the sequences. Needs to read a batch of
     |        one-hot encoded sequences as a three dimensional array
     |        TFrecordsgenerator(TFRecordsgenerator): from helpers.py, generates the aa2id and id2aa dict,
     |        the one-hot encoding
     |  
     |  blosumscore(self, seq)
     |      Calculates the sum of the blosum62 matrix entries that correspond to the mutations in seq relative to
     |          self.starseq
     |      Args:
     |          seq(ndarray): contains the sequence that is evaluated
     |      
     |      Returns(float): The sum of the blosum62 entries for the mutations in the given sequence.
     |  
     |  choose(self, seqs, scores, survivalpop=-1)
     |      Takes a set of sequences and chooses the best subset according to the provided scores. The subset contains
     |      Args:
     |          seqs(ndarray): One-hot encoded sequences, the first dimension is the batch dimension
     |          scores(list of float): Scores, has to be in the same order as seqs
     |          survivalpop(list of int): Specifies how many instances of the best, second best, ... are chosen
     |      
     |      Returns:
     |          outseq(ndarray): One-hot encoded chosen sequences.
     |              Each sequence occurs multiple times as specified in survivalpop
     |          buff(ndarray): One-hot encoded input sequences sorted with the scores provided
     |  
     |  count_muts(self, seq)
     |      Counts mutations compared to starting sequence
     |      Args:
     |          seq(ndarray([20, sequence length]): sequence that is compared to starting sequence
     |      
     |      Returns (int): number of mutations in sequence relative to starting sequence
     |  
     |  evolve(self, generations=-1)
     |      Coordinates the evolutionary improvement of a sequence.
     |      Mutates the sequences, scores the sequences, selects sequences. While that happens statistics are updated,
     |      the logfile is written.
     |      Args:
     |          generations (int): overwrites the number of generations specified in the opts object that is generated from
     |          the config.json file (optional).
     |  
     |  len_seq(self, seq)
     |  
     |  mut_residues(self, seq1, seq2=None)
     |      Compares two sequences and returns a string listing up the differences in amino acid composition of the
     |       two sequences
     |      Args:
     |          seq1 (str): Sequence that is compared to either the second sequence or the starting sequence.
     |          seq2 (str): Sequence to which the first sequence is compared to (optional).
     |      
     |      Returns (str): Comma seperated information on how many of each amino acid are in the first and in the second
     |      sequence:
     |      Mutated:
     |      <number of occurences of aa1 in sequence 1>, <number of occurences of aa2 in sequence 1>, ...
     |      to:
     |      <number of occurences of aa1 in sequence 2>, <number of occurences of aa2 in sequence 2>, ...
     |  
     |  mutate_specific(self, muts, seq=None)
     |      Introduced a specific mutation in a given sequence or the starting sequence if no sequence is provided.
     |      Args:
     |          muts (list of str): A list of strings that specifiy the mutations that shall be made in the sequence,
     |              each element starts with an integer that specifies the position to mutatated and ands with a char that
     |              defines to which amino acid the position is mutated.
     |          seq (ndarray([20, sequence length]): sequence, in which the given mutations are introduced (optional).
     |      
     |      Returns (ndarray([20, sequence length]): sequence that has the specified mutations
     |  
     |  mutated_seq(self, seq)
     |      Performs the mutation on one sequence, guarantees that no more than self.maxmut mutations are introduced by
     |          mutating a mutated residue back to its wt amino acid when the number of mutations is too large
     |      Args:
     |          seq(ndarray): one-hot encoded sequence that is mutated
     |      
     |      Returns(ndarray): one-hot encoded sequence that was mutated
     |  
     |  mutates_seq(self, seqs2mutate)
     |      Mutates a set of sequences, preserves the best (which is assumed to be in the first position)
     |      Args:
     |          seqs2mutate(ndarray): first dimension is the batch dimension, second and third are the one-hot encoding.
     |          Sequences to mutate
     |      
     |      Returns(ndarray): Mutated sequences in the same structure as seqs2mutate
     |  
     |  randomize(self, until)
     |      Works only with goal class!
     |      Calculates random mutants with x mutated amino acids and their scores, with x being each number from 0 to until
     |      dumps data as pickle and calls plotting functions.
     |      Args:
     |          until (int): specifies the maximum number of mutations the sequence shall have
     |  
     |  randomize_all(self, until)
     |      Works with all classes!
     |      Calculates random mutants with x mutated amino acids and their scores, with x being each number from 0 to until
     |      dumps data as pickle and calls plotting functions.
     |      Args:
     |          until (int): specifies the maximum number of mutations the sequence shall have
     |  
     |  score(self, seq, classes, classes_variance)
     |      Calculates a score for a sequence using the logists retrieved from the classifier as well as their variance and
     |          other information on the sequence, like the blosum score. Weights for goal and avoid classes are taken from
     |          the gaia file, the blosum weight is normalized with the length of the sequence
     |      Args:
     |          seq(ndarray): Sequence to be scored in one-hot encoding
     |          classes(ndarrray): Mean logits for the sequence from the classifier.
     |          classes_variance(ndarray): Variance between the logits for the sequence from the classifier
     |      
     |      Returns(float): A score for the sequence
     |  
     |  score_combinations(self, ipath, max2combine)
     |      Calculates the scores for all combinations possible for the mutation sets in the input file
     |      that have max2combine elements, dumps data as pickle, writes scores with all available information to logfile.
     |      Args:
     |          ipath (str): path to input file, in which mutations to be comined are specified:
     |          each line is treated as one unit in combination,
     |          if a line contains several mutations these always occur together. The delimiter between two mutations is ','
     |          A mutation starts with an integer that defines the position and ends with a char that defines the amino acid
     |          to which the position is mutated.
     |          max2combine (int): sets the limit on how many lines are combined together.
     |          Useful to decrease the number of possible combinations.
     |  
     |  score_combinations_BINARY(self, ipath, max2combine)
     |      Calculates the scores for all combinations possible for the mutation sets in the input file
     |      that have max2combine elements, dumps data as pickle, writes scores with all available information to logfile.
     |      Args:
     |          ipath (str): path to input file, in which mutations to be comined are specified:
     |          each line is treated as one unit in combination,
     |          if a line contains several mutations these always occur together. The delimiter between two mutations is ','
     |          A mutation starts with an integer that defines the position and ends with a char that defines the amino acid
     |          to which the position is mutated.
     |          max2combine (int): sets the limit on how many lines are combined together.
     |          Useful to decrease the number of possible combinations.
     |  
     |  systematic_mutation(self, seqs2mutate)
     |      Introduces mutations in seqs2mutate and keeps in memory, which mutations were already tried.
     |      Still random positions are chosen, but for each position all amino acids are tried systematically
     |      Preserves the best sequence. Never returns a sequence with more mutations than specified
     |      Args:
     |          seqs2mutate(ndarray): one-hot encoded sequences to mutate
     |      Returns
     |          seqs2mutate(ndarray): one-hot encoded mutated sequences
     |          continue_here(bool): if false, all mutations that can be reached by a single mutation we're tried,
     |              the algorithm will not find a better one
     |  
     |  translate_seq(self, seq, treshold=0.6)
     |      Turns a one-hot encoded sequence into a string. Also supports encodings with a dimension for each amino acid, if
     |          they use the same aa2id dict. Writes the amino acid if the modus is greater than a threshold. If no amino
     |          acid is written for the position, an underscore is written.
     |      Args:
     |          seq(ndarray): one-hot encoded sequence to be translated
     |          treshold(float): threshold over which the amino acid is counted.
     |      
     |      Returns(str): The sequence in capital letters
     |  
     |  walk(self, highl=[100000], highl_names=['Default'], f_width=10, f_height=8, res=80, name='walk_data.png')
     |      Calculates all possible single mutants of a sequence, makes data available to plot, calls plotting functions
     |          and dumps pickle files
     |      Args:
     |          highl (list of int): passed to plotting functions, indicates which classes shall be highlighted
     |          highl_names (list of str): passed to plotting funcitons, speciefies a name for each element in highl
     |          f_width (float): passed to plotting functions, specifies the width of the plots that are written,
     |              except for plots that show position specific information
     |          f_height (float): passed to plotting functions, specifies the height of the plots that are written
     |          res (float): passed to plotting functions, specifies the resolution of the plots that are written
     |          name: passed to plotting functions, specifies the name of plots that are written.
     |              Prefixes or suffixes are added
     |  
     |  write_currfile(self, text, gen=0)
     |      Overwrites a file with current information on the sequence, adds the current time.
     |      Args:
     |          text(str): The information to be written, usually the same that is written to the logfile
     |          gen(int): The generation in which the algorithm is
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Memory(builtins.object)
     |  Stores the information needed to prevent trying the same sequence multiple times. Calculates the next position,
     |  amino acid and backmutating position needed for a mutation
     |  
     |  Methods defined here:
     |  
     |  __init__(self, startseq, seq2mutate, maxmut)
     |      Initialises the memory with a startseq relative to which the number of mutations is calculated and a
     |      sequence2mutate, for which position amino acid and backmutating position are calculated
     |      Args:
     |          startseq(ndarray): one-hot encoded sequence,
     |              the original sequence that is used to count the number of mutations
     |          seq2mutate(ndarrray): one-hot encoded sequence, sequence in which mutations are introduced with the
     |              with the informations provided by the position() function
     |          maxmut(int): Maximum number of mutations that is allowed. If a new mutation would result in more mutations,
     |              a postion to backmutate is provided
     |  
     |  new_tobackmutate(self)
     |      Randomly hooses a new postion to backmutate if needed,
     |          does not take a position that was already used for backmutation with the current mutation position before
     |  
     |  new_tomutate(self)
     |      Randomly hooses a new postion to mutate,
     |          does not take a position that was already used for mutation before.
     |              If all mutation positions were tried before, the algorithm will not find something better as all
     |              combinations of mutations reachable within a single mutation and backmutation were tried.
     |  
     |  position(self)
     |      Calculates all informations needed to introduce mutations in a function
     |      Returns:
     |          aa(int): amino acid, key from id2aa dict. The position is mutated to this residue
     |          pos(int): position in the sequence that is mutated
     |          backpos(int): position in the sequence that is backmutated if back is true
     |          back(bool): true, if backmutation is needed
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Stats(builtins.object)
     |  Holds statistics, saves them as pickle file, plots them as PNGs.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, savedir=None, goallabels=None, avoidslabels=None, goalsdict=None, avoidsdict=None, head=None, pickledir=None)
     |      Initialises the statistics either empty or from a picklefile.
     |      Args:
     |          savedir(str): directory to which the data is saved,
     |              not needed when initialising from a pickle file (optional).
     |          goallabels(list of str): names written to the plots for the goal classes,
     |              not needed when initialising from a pickle file (optional).
     |          avoidslabels(list of str): names written to the plots for the avoid classes,
     |              not needed when initialising from a pickle file (optional).
     |          goalsdict(list of int): keys for the class dict, specify the goal classes for highligthing,
     |              not needed when initialising from a pickle file (optional).
     |          avlidssdict(list of int): keys for the class dict, specify the avoids classes for highligthing,
     |              not needed when initialising from a pickle file (optional).
     |          head(str): Title of the written plots,
     |              not needed when initialising from a pickle file (optional).
     |          pickledir(str): directory from which the Stats object is restored,
     |              only needed when initialising from a pickle file (optional).
     |  
     |  load(self, picklefile)
     |      Loads picklefile and sets the attributes to the values from the picklefile.
     |      Args:
     |          picklefile(str): path to picklefile to restore from
     |  
     |  plot_all(self, f_width=10, f_height=8, res=80, name='stats.png')
     |      Plots the development of each goal score, avoid score and their variances,
     |      indicates wether mutations occur and wether thesystematic mode is used
     |      Args:
     |          f_width (float): specifies the width of the plot that is written
     |          f_height (float): specifies the height of the plot that is written
     |          res (float): specifies the resolution of the plot that is written
     |          name (str): specifies the name of plots that is written
     |  
     |  plotdistoverseq(self, f_width=16, f_height=8, res=200, name='hist.png')
     |      Plots how often a position was mutated in a bar plot, including backmutations
     |      Args:
     |          f_width (float): specifies the width of the plot that is written
     |          f_height (float): specifies the height of the plot that is written
     |          res (float): specifies the resolution of the plot that is written
     |          name (str): specifies the name of plots that is written
     |  
     |  plotdisttooriginal(self, f_width=16, f_height=8, res=200, name='hist_rel.png')
     |      Plots how often a position was not the original residue
     |      Args:
     |          f_width (float): specifies the width of the plot that is written
     |          f_height (float): specifies the height of the plot that is written
     |          res (float): specifies the resolution of the plot that is written
     |          name (str): specifies the name of plots that is written
     |  
     |  update(self, classes, scores, mutated, mutating, seqs, systematic, classes_variance, blosum_score, mutated_aas)
     |      Updates the statistics with a new set of values, dumps the current state of the statistics object as pickle
     |      Args:
     |          classes(ndarray): logits retrieved from the classifier
     |          garbage(garbage scores): depracted
     |          scores(list): Scores calculated by the scoring function
     |          mutated(float): Amount of the sequence that is mutated
     |          mutating(0 or 1): indicates wether a mutation in the best sequenced occured, compared to the last generation
     |          seqs(ndarray): one-hot encoded sequences with additional information in dimension 20 and 21 in the one-hot
     |              dimension (1)
     |          systematic(0 or 1): indicates wether systematic mode was active
     |          classes_variance(ndarray): variance between logits from the classifier
     |          blosum_score(float): sum of blosum62 matrix entries for the mutations in the best sequence
     |          mutated_aas(list): List of positions that are mutated in the current best sequences
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    draw_heatmap(scores, goal_class, wt_scores, startseq_chars, wdir, logfile, f_width=50, f_height=4, res=300)
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
    
    plot_all_data(data, goal, wdir, logfile, highl=[100000], highl_names=['Default'], f_width=10, f_height=8, res=80, name='data.png')
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
    
    plot_data(data, wdir, logfile, f_width=10, f_height=8, res=80, name='data.png')
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
    
    plot_walk(data, goal, wdir, logfile, wt_scores=None, highl=[10000], highl_names=['Default'], aas=[0], f_width=5, f_height=4, res=300, name='walk_data_HD.png')
        Plots graphs indicating how the scores change, when a specific position is mutated.
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

DATA
    __warningregistry__ = {'version': 57, ("unclosed file <_io.TextIOWrapp...
    cdict = {'blue': [(0.0, 0.1154901961, 0.1354901961), (0.5, 1.0, 1.0), ...
    expit = <ufunc 'expit'>
    font = <matplotlib.font_manager.FontProperties object>
    monospaced = <matplotlib.font_manager.FontProperties object>

FILE
    /net/data.isilon/igem/2017/scripts/Heidelberg_2017/GAIA/gaia.py

Help on module plot_randomized:

NAME
    plot_randomized

FUNCTIONS
    main()
        sys.argv[1] is the summariesdir, where gaia wrote the data.

FILE
    /net/data.isilon/igem/2017/scripts/Heidelberg_2017/GAIA/plot_randomized.py


Help on module plot_walk:

NAME
    plot_walk

FUNCTIONS
    main()

FILE
    /net/data.isilon/igem/2017/scripts/Heidelberg_2017/GAIA/plot_walk.py


Help on module score_combinations:

NAME
    score_combinations - Train DeeProtein. A config dict needs to be passed.

FUNCTIONS
    main()

FILE
    /net/data.isilon/igem/2017/scripts/Heidelberg_2017/GAIA/score_combinations.py


Help on module randomize:

NAME
    randomize - Randomizes a sequence, plots scores of DeeProtein. A config dict needs to be passed.

FUNCTIONS
    main()

FILE
    /net/data.isilon/igem/2017/scripts/Heidelberg_2017/GAIA/randomize.py


Help on module evolve:

NAME
    evolve - Train DeeProtein. A config dict needs to be passed.

FUNCTIONS
    main()

FILE
    /net/data.isilon/igem/2017/scripts/Heidelberg_2017/GAIA/evolve.py
