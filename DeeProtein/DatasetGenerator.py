import sys
import re
import os
import pandas as pd
import random
import pickle
import glob
import json
from goatools.obo_parser import GODag
from collections import defaultdict
from collections import Mapping


class DatasetGenerator():
    """The DatasetGenerator takes takes care off the whole data_preprocessing pipeline.
    From the uniprot csv to the final train_dataset.csv the whole pipline is covered by this class. Fist the uniprot.txt
    download file is filtered and import information is extracted into a uniprot.csv.
      This .csv file is then splitted into single .csv files one for each encountered GO_term. These per label .csv's
    contain all sequences annotated with that specific label.
      To generate a dataset a GO_file needs to be passed. This file has to contain the nr of sequences annotatad with
    that GO term, and the GO_term seperated by white_space, one pair per line. This file should contain all classes that
    should be included in the dataset. The dataset generator subsequently parses that file and extracts the sequences
    into the dataset. The set is checked for redundancy and for each label 5 random sequences are extracted and included
    in the test-set.

    Attributes:
      depth: `int32` the maximum depth of GO_labels in the GO-DAG to condsider. DEPRACTED. Use max_depth instead.
      mode: `str` either 'EC' or 'GO', specifies the type of labels to use.
      GODag: `goatools.obo_parser.GODag` object. Holds the information on the GODag.
      max_depth: `int32` the maximum depth of GO_labels in the GO-DAG to condsider.
      max_write: `int32` the maximum nr of samples to write for a GO_label.
      write_count: `int32` counter for the written samples for a term.
      save_dir: `str` the path to the dir where to save the the datasets.
      uniprot_csv: `str` the path to the uniprot.csv
      uniprot_file_path: `str` the path to the raw uniprot.txt download.
      class_to_id_EC: `nested dict` mapping classes to `int` ids and holding the class sizes.
      class_to_id_GO: `dict` mapping classes to `int` ids and holding the class sizes.
      filter_minlength: `bool` whether to filter the uniprot.csv for minlength or not.
      minlength: `int` the minlength to filter for.
      filter_AA: `bool` whether to filter for non-canonical AAs or not.
      train_dataset_csv_path: `str` dir where to store the final datasets.
    """
    def __init__(self, uniprot_file_path, csv_file_path, save_dir, mode='EC'):
        self.depth = 4 #specifies the depth of labels to consider
        self.mode = mode
        self.GODag = GODag('/net/data.isilon/igem/2017/data/gene_ontology/go.obo', optional_attrs=['relationship'])
        self.max_depth = 4 #max_depth in goDAG to consider
        self.max_write = 1000000 #specify the max amount of labels to be written for one class:
        self.write_count = 0 #counter to see how much we already worte
        self.save_dir = save_dir
        self.uniprot_csv = csv_file_path
        self.uniprot_file_path = uniprot_file_path
        self.class_to_id_EC = _recursively_default_dict()
        self.class_to_id_GO = {}
        self.filter_minlength = True
        self.minlength = 175
        self.filter_AA = True
        self.train_dataset_csv_path = '/net/data.isilon/igem/2017/data/uniprot_with_EC/SAfetyNEt/'
        print(save_dir)

        # load the dict if it's there:
        try:
            with open(os.path.join(self.save_dir, 'csv_by_EC', os.path.join(
                    'class2id_{}.p'.format(self.mode))),
                      "rb") as pickle_f:
                self.class_to_id_EC = pickle.load(pickle_f)
                # freeze the default dict
                self.class_to_id_EC.default_factory = None
                print('Loaded EC-class dict.')
        except OSError:
            print('Failed to load EC-class dict. Generating EC-class dict.')
            #self.separate_classes_by_EC()

    def _simple_fileparser(self, in_fobj):
        """A fileparser yielding the lines.

        Args:
          in_fobj: `fileObject` to parse.

        Returns:
          The line.
        """
        for line in in_fobj:
            yield line

    def _uniprot_csv_parser(self, in_fobj):
        """Parser for the uniprot.csv.

        The uniprot csv file is in the following syntax:
        name;rec_name;Pfam;protein_existance;seq;F_GO;P_GO;C_GO;EC;Structure

        Args:
          in_fobj: `fileObject`

        Yields:
          name: `str` the name.
          seq: `str` the sequence.
          GO: `list` holding the GO labels of a sequence.
          EC: `str` holding the EC label.
          structure_str: `str` defninig the secondary structure of the sequence.
        """
        for line in in_fobj:
            fields = line.strip().split(';')
            name = fields[0]
            # str to list
            seq = fields[4]
            go_str = re.sub('[\'\[\]]', '', fields[5])
            GO = go_str.split(',')
            EC_str = re.sub('[\'\[\],]', '', fields[6])
            EC = EC_str.split()
            structure_str = fields[9]
            yield name, seq, GO, EC, structure_str

    def uniprot_to_csv_on_disk(self):
        """Convert the raw uniprot download into a csv file.

        Converts the raw uniprot.txt download (specified in the class_attributes) into a .csv file with the
        following syntax:

        name;rec_name;Pfam;protein_existance;seq;F_GO;P_GO;C_GO;EC;Structure

        After each entry the information is written, to avoid a memory explosion.
        """
        uniprot_dict = {}
        uniprot_csv_path = os.path.join(self.save_dir, 'swissprot_{}.csv'.format(self.mode))
        uniprot_pickle_path = os.path.join(self.save_dir, 'swissprot_{}.p'.format(self.mode))
        out_csv = open(uniprot_csv_path, 'a')
        with open(self.uniprot_file_path, "r") as in_fobj:
            curr_prot_id = ''
            curr_F_GOs = []
            curr_P_GOs = []
            curr_C_GOs = []
            curr_ECs = []
            curr_structure = []
            seq = False
            for line in in_fobj:
                fields = line.strip().split()
                flag = fields[0]
                if flag == 'ID' and len(fields) >= 2:
                    curr_prot_id = fields[1]
                    uniprot_dict[curr_prot_id] = {}
                elif flag == 'DE' and len(fields) >= 2:
                    rec_name = re.search(r'(?<=Full=)(.+?)[;\s]', line)
                    ec_nr = re.search(r'(?<=EC=)([0-9.-]*?)[;\s]', line)
                    if ec_nr:
                        curr_ECs.append(ec_nr.group(1))
                    elif rec_name:
                        uniprot_dict[curr_prot_id]['rec_name'] = rec_name.group(1)
                elif flag == 'DR' and len(fields) >= 2:
                    '''
                    abfrage fuer GOS und PFAM
                    '''
                    # ask for GO first:
                    dr_fields = [ref.rstrip('.;') for ref in fields[1:]]
                    # TODO: should we filter for funcitonalilty here?
                    if dr_fields[0] == 'GO' and dr_fields[2].startswith('F'):
                        curr_F_GOs.append(dr_fields[1])
                        # try:
                        #     uniprot_dict[curr_prot_id]['GO'].append(dr_fields[1])
                        # except KeyError:
                        #     uniprot_dict[curr_prot_id]['GO'] = [dr_fields[1]]
                    elif dr_fields[0] == 'GO' and dr_fields[2].startswith('P'):
                        curr_P_GOs.append(dr_fields[1])
                    elif dr_fields[0] == 'GO' and dr_fields[2].startswith('C'):
                        curr_C_GOs.append(dr_fields[1])

                    elif dr_fields[0] == 'Pfam':
                        uniprot_dict[curr_prot_id]['Pfam'] = dr_fields[2:]
                    else:
                        pass
                elif flag == 'CC' and len(fields) >= 2:
                    '''
                    may content sequence caution warning
                    '''
                    pass

                elif flag == 'PE' and len(fields) >= 2:
                    protein_existance = fields[1]
                    uniprot_dict[curr_prot_id]['protein_existance'] = protein_existance

                elif flag == 'FT' and len(fields) >= 2:
                    """
                    the annotated features. (http://www.uniprot.org/help/sequence_annotation)
                    Those are anotations like catalytic site, binding sites and secondary structure
                    """
                    ft_fields = fields[1:]
                    if ft_fields[0] == 'HELIX':
                        curr_structure.append(('HELIX', fields[2], fields[3]))
                    if ft_fields[0] == 'STRAND':
                        curr_structure.append(('STRAND', fields[2], fields[3]))
                    if ft_fields[0] == 'SHEET':
                        curr_structure.append(('SHEET', fields[2], fields[3]))
                    if ft_fields[0] == 'TURN':
                        curr_structure.append(('TURN', fields[2], fields[3]))
                elif flag == 'SQ' and len(fields) >= 2:
                    seq = True
                    uniprot_dict[curr_prot_id]['seq'] = ''
                elif seq == True:
                    if flag == '//':
                        uniprot_dict[curr_prot_id]['F_GO'] = self._full_annotation(curr_F_GOs)
                        uniprot_dict[curr_prot_id]['P_GO'] = self._full_annotation(curr_P_GOs)
                        uniprot_dict[curr_prot_id]['C_GO'] = self._full_annotation(curr_C_GOs)
                        uniprot_dict[curr_prot_id]['EC'] = curr_ECs
                        uniprot_dict[curr_prot_id]['Structure'] = curr_structure
                        curr_prot_id = ''
                        seq = False
                        # set collectors to []
                        curr_F_GOs = []
                        curr_C_GOs = []
                        curr_P_GOs = []
                        curr_ECs = []
                        curr_structure = []

                        # write the entry to file
                        uniprot_df = pd.DataFrame.from_dict(uniprot_dict, orient='index')
                        uniprot_df.to_csv(out_csv, sep=';', na_rep='', header=False, index=True,
                                          line_terminator='\n')
                        uniprot_dict = {}
                    else:
                        uniprot_dict[curr_prot_id]['seq'] += ''.join(fields)
                else:
                    pass
        out_csv.close()

    def uniprot_to_csv(self):
        """Convert the raw uniprot download into a csv file.

        Converts the raw uniprot.txt download (specified in the class_attributes) into a .csv file with the
        following syntax:

        name;rec_name;Pfam;protein_existance;seq;F_GO;P_GO;C_GO;EC;Structure

        The whole dataframe is held in memory. While this is faster for small files, please use the on_disk method for
        larger files (exceeding a few GB).
        """
        uniprot_dict = {}
        uniprot_csv_path = os.path.join(self.save_dir, 'uniprot_prefiltered_{}.csv'.format(self.mode))
        #uniprot_pickle_path = os.path.join(self.save_dir, '_{}.p'.format(self.mode))
        with open(self.uniprot_file_path, "r") as in_fobj:
            curr_prot_id = ''
            curr_F_GOs = []
            curr_P_GOs = []
            curr_C_GOs = []
            curr_ECs = []
            curr_structure = []
            seq = False
            for line in in_fobj:
                fields = line.strip().split()
                flag = fields[0]
                if flag == 'ID' and len(fields) >= 2:
                    curr_prot_id = fields[1]
                    uniprot_dict[curr_prot_id] = {}
                elif flag == 'DE' and len(fields) >= 2:
                    rec_name = re.search(r'(?<=Full=)(.+?)[;\s]', line)
                    ec_nr = re.search(r'(?<=EC=)([0-9.-]*?)[;\s]', line)
                    if ec_nr:
                        curr_ECs.append(ec_nr.group(1))
                    elif rec_name:
                        uniprot_dict[curr_prot_id]['rec_name'] = rec_name.group(1)
                elif flag == 'DR' and len(fields) >= 2:
                    '''
                    abfrage fuer GOS und PFAM
                    '''
                    # ask for GO first:
                    dr_fields = [ref.rstrip('.;') for ref in fields[1:]]
                    # TODO: should we filter for funcitonalilty here?
                    if dr_fields[0] == 'GO' and dr_fields[2].startswith('F'):
                        curr_F_GOs.append(dr_fields[1])
                        # try:
                        #     uniprot_dict[curr_prot_id]['GO'].append(dr_fields[1])
                        # except KeyError:
                        #     uniprot_dict[curr_prot_id]['GO'] = [dr_fields[1]]
                    elif dr_fields[0] == 'GO' and dr_fields[2].startswith('P'):
                        curr_P_GOs.append(dr_fields[1])
                    elif dr_fields[0] == 'GO' and dr_fields[2].startswith('C'):
                        curr_C_GOs.append(dr_fields[1])

                    elif dr_fields[0] == 'Pfam':
                        uniprot_dict[curr_prot_id]['Pfam'] = dr_fields[2:]
                    else:
                        pass
                elif flag == 'CC' and len(fields) >= 2:
                    '''
                    may content sequence caution warning
                    '''
                    pass

                elif flag == 'PE' and len(fields) >= 2:
                    protein_existance = fields[1]
                    uniprot_dict[curr_prot_id]['protein_existance'] = protein_existance

                elif flag == 'FT' and len(fields) >= 2:
                    """
                    the annotated features. (http://www.uniprot.org/help/sequence_annotation)
                    Those are anotations like catalytic site, binding sites and secondary structure
                    """
                    ft_fields = fields[1:]
                    if ft_fields[0] == 'HELIX':
                        curr_structure.append(('HELIX', fields[2], fields[3]))
                    if ft_fields[0] == 'STRAND':
                        curr_structure.append(('STRAND', fields[2], fields[3]))
                    if ft_fields[0] == 'SHEET':
                        curr_structure.append(('SHEET', fields[2], fields[3]))
                    if ft_fields[0] == 'TURN':
                        curr_structure.append(('TURN', fields[2], fields[3]))
                elif flag == 'SQ' and len(fields) >= 2:
                    seq = True
                    uniprot_dict[curr_prot_id]['seq'] = ''
                elif seq == True:
                    if flag == '//':
                        uniprot_dict[curr_prot_id]['F_GO'] = self._full_annotation(curr_F_GOs)
                        uniprot_dict[curr_prot_id]['P_GO'] = self._full_annotation(curr_P_GOs)
                        uniprot_dict[curr_prot_id]['C_GO'] = self._full_annotation(curr_C_GOs)
                        uniprot_dict[curr_prot_id]['EC'] = curr_ECs
                        uniprot_dict[curr_prot_id]['Structure'] = curr_structure
                        curr_prot_id = ''
                        seq = False
                        # set collectors to []
                        curr_F_GOs = []
                        curr_C_GOs = []
                        curr_P_GOs = []
                        curr_ECs = []
                        curr_structure = []
                    else:
                        uniprot_dict[curr_prot_id]['seq'] += ''.join(fields)
                else:
                    pass
            uniprot_df = pd.DataFrame.from_dict(uniprot_dict, orient='index')
            for key in uniprot_dict.keys():
                uniprot_dict.pop(key)

            uniprot_df.to_csv(uniprot_csv_path, sep=';', na_rep='', header=True, index=True,
                              line_terminator='\n')
            #uniprot_df.to_pickle(uniprot_pickle_path)

    def _full_annotation(self, GO_terms):
        """Takes a list of GO_terms and expands them to full annotation.

        Completes the subdag defined by a list of GO-terms by walking up the GOdag.

        Args:
          GO_terms: `list` the GO_terms to expand.
        Returns:
          A fully annotated list of GO terms including all nodes passed on the way to root.
        """
        full_go = set()

        omitted_GOs = []

        for go in GO_terms:
            # determine if a node has parents and retrieve the set of parent-nodes
            try:
                full_go.update(self.GODag[go].get_all_parents())
                full_go.add(go)
            except KeyError:
                # this means the term might be obsolete as its not in the DAG. store it
                omitted_GOs.append(go)
                pass

        print(omitted_GOs)

        full_go = ','.join(list(full_go))

        return full_go

    def separate_classes_by_GO(self, jobnr=None):
        """Seperates the whole uniprot.csv into GO-specific .csv-files.

        First generates a GO to ID dict, then split the uniprot.csv into GO-term specific files.

        Args:
          jobnr: `str`, a jobnumber if this funciton is used in a jobarray to handle the whole uniprot.csv (optional)
        """
        helper_GO_to_id = {}
        # generate extra folder in savedir to store the single files in
        csv_by_GO_path = os.path.join(self.save_dir, 'csv_by_GO_structure_splitted')
        # get a GO-dict to store the population of all GO-terms we have. Dict is flat and not lvl wise (as wen do not
        # need to reconstruct the DAG.
        self.GO_population_dict = {}

        omitted_GOs = []

        # set up a GOdag for molecular function only: stored in self.GODag
        # got through the uniprot csv once and set up a dict of all GO-terms. Pass an ID
        with open(self.uniprot_csv, "r") as in_fobj:
            in_fobj.readline()
            for name, seq, GO_terms, EC_nrs, structure_str in self._uniprot_csv_parser(in_fobj):
                # iterate through the whole GO annotation and complete it by checking for parents in the DAG
                if self._valid_seq(seq):
                    full_go = set()
                    #if __name__ == '__main__':
                    for go in GO_terms:
                        # determine if a node has parents and retrieve the set of parent-nodes
                        try:
                            full_go.update(self.GODag[go].get_all_parents())
                            full_go.add(go)
                        except KeyError:
                            print(go)
                            # this means the term might be obsolete as its not in the DAG. store it
                            omitted_GOs.append((name, go))
                            pass

                    # sort the full annotation by levels:
                    full_go_by_level = {}
                    levels = set([self.GODag[go].level for go in list(full_go)])
                    for lvl in sorted(list(levels)):
                        full_go_by_level[lvl] = [go for go in list(full_go) if self.GODag[go].level == lvl]

                    for lvl, go_terms in full_go_by_level.items():
                        # open the corresponding csvs and add the line from the uniprot csv.
                        for go_term in go_terms:
                            # update the counters for the population dict
                            try:
                                self.GO_population_dict[go_term] += 1
                            except KeyError:
                                self.GO_population_dict[go_term] = 1

                            with open(os.path.join(csv_by_GO_path, '%d_%s.csv_%s' % (lvl, go_term, jobnr)),"a") as go_csv:
                                line = ";".join([name, seq, ','.join(list(full_go)), ','.join(go_terms),
                                                 go_term, str(lvl), structure_str])
                                line += '\n'
                                go_csv.write(line)
        print(omitted_GOs)

    def extract_names_for_GO_list(self, GO_file, out_file_path):
        """Extract the protein names form the .csv if the annotated GO-terms match a GO-term in the GO-list.

        Args:
          GO_file: `str` path to the GO_file for which to extract the matching protein names.
          out_file_path: `str` the path to the outfile.
        """
        out_file_names_only_path = out_file_path + '.names_only'

        # 1. read the GO-file and list the terms
        GO_list = []
        with open(GO_file, "r") as GO_file:
            for line in GO_file:
                fields = line.strip().split()
                GO = fields[0]
                GO_list.append(GO)

        GO_set = set(GO_list)
        del GO_list

        # 2. iterate over the .csv and extract the relevant terms
        def uniprot_csv_parser(in_fobj):
            for line in in_fobj:
                fields = line.strip().split(';')
                name = fields[0]
                # str to list
                seq = fields[4]
                F_GO = fields[5].split(',') if ',' in fields[5] else fields[5]
                P_GO = fields[6].split(',') if ',' in fields[5] else fields[6]
                C_GO = fields[7].split(',') if ',' in fields[5] else fields[5]
                EC_str = re.sub('[\'\[\],]', '', fields[8])
                EC = EC_str.split()
                yield name, seq, F_GO, P_GO, C_GO, EC

        with open(self.uniprot_csv, "r") as uniprot_csv_obj, \
                open(out_file_path, "w") as out_fobj, \
                open(out_file_names_only_path, "w") as out_fobj_names_only:
            for name, seq, F_GO, P_GO, C_GO, EC in uniprot_csv_parser(uniprot_csv_obj):
                # now iterate over all GO_terms and check if we match a term of the List:
                annotated_GOs = set(F_GO)
                annotated_GOs.update(P_GO)
                annotated_GOs.update(C_GO)
                if annotated_GOs.intersection(GO_set):
                    f = ','.join(F_GO)
                    p = ','.join(P_GO)
                    c = ','.join(C_GO)
                    ec = ','.join(EC)
                    line = [name, seq, f, p, c, ec]
                    line += '\n'
                    out_fobj.write(';'.join(line))
                    out_fobj_names_only.write(name+'\n')

    def filter_FASTA_for_names(self, fasta_file_path, names_file):
        """Extract FASTA entries from a .fasta file by their protein name.

        Args:
          names_file: `str` path to a file containing a protein name per line.
          fasta_file_path: `str` path to the fasta file to extract the entries from.
        """
        names_list = []
        with open(names_file, "r") as in_fobj:
            for line in in_fobj:
                names_list.append(line.strip())

        assert os.path.exists(fasta_file_path)
        out_path = fasta_file_path + '.filtered_%s' % os.path.basename(names_file).split('.')[0]

        with open(fasta_file_path, "r") as in_fasta, open(out_path, "w") as out_fasta:
            curr_entry = []
            for line in in_fasta:
                if line.startswith('>'): #header line
                    if curr_entry:
                        out_fasta.write(''.join(curr_entry))
                        curr_entry = []
                    fields = line.strip().split('|')
                    name = fields[2].split()[0]
                    if name in names_list:
                        curr_entry.append(line)
                else: #sequence
                    if curr_entry: # = if curr_entry != []
                        curr_entry.append(line)
                    else:
                        pass
            # write the last entry:
            if curr_entry: # = if curr_entry != []
                out_fasta.write(''.join(curr_entry))

    def FASTA_to_dict(self, fasta_file_path):
        """Read a fastA file and extract a dict with Altname field as key.

        Args:
          fasta_file_path: `str` fasta file to read. JSON si dumped in the same dir.
        """
        assert os.path.exists(fasta_file_path)
        out_path = fasta_file_path + '.JSON'

        fasta_dict = StratifiedDictFASTA()

        with open(fasta_file_path, "r") as in_fasta, open(out_path, "w") as out_fasta:

            for line in in_fasta:
                if line.startswith('>'): #header line, those are the interesting ones
                    # >sp|C0JAU1|A1H2_LOXSP Phospholipase D LspaSicTox-alphaIA1ii (Fragment) OS=Loxosceles spadicea PE=2 SV=1
                    fields = line.strip().split('|')
                    sp_id = fields[1]
                    description = fields[2].split()
                    altname = description[0]
                    species = re.search(r'(?<=OS=)(.+?[\s].+?)\b', fields[2])
                    species = species.group(1) if species else ''
                    fasta_dict[altname]['sp_id'] = sp_id
                    fasta_dict[altname]['OS'] = species

        with open(out_path, "w") as f:
            json.dump(fasta_dict, f)

    def separate_classes_by_EC(self):
        """Get class to ID dict. Encode each class in a certain depth as an integer

        Generate a .csv file for each EC class in the uniprot.csv.
        """
        helper_EC_to_id = {}

        # generate extra folder in savedir to store the single files in
        csv_by_EC_path = os.path.join(self.save_dir, 'csv_by_EC')

        # make sure the directory is empty as we append to the files:
        if os.path.exists(csv_by_EC_path):
            existing_files = glob.glob(os.path.join(csv_by_EC_path, '*csv'))
            for path in existing_files:
                os.remove(path)
        else:
            print(csv_by_EC_path)
            os.makedirs(csv_by_EC_path)

        #go through the csv once and set up a stratified dict for each class!
        with open(self.uniprot_csv, "r") as in_fobj:
            in_fobj.readline()
            for name, seq, GO, EC_nrs in self._uniprot_csv_parser(in_fobj):
                # TODO: to this point we exclude all protein with multiple EC numbers
                if len(set(EC_nrs)) == 1:
                    EC_nr = EC_nrs[0].split('.')
                    helper_EC_to_id[EC_nrs[0]] = len(helper_EC_to_id)

                    for key, value in [('ID', helper_EC_to_id[EC_nrs[0]]),
                                       ('path', '%s.csv' % EC_nrs[0])]:
                        self.class_to_id_EC[EC_nr[0]][EC_nr[1]][EC_nr[2]][EC_nr[3]][key] = value
                    try:
                        self.class_to_id_EC[EC_nr[0]][EC_nr[1]][EC_nr[2]][EC_nr[3]]['count'] += 1
                    except TypeError:
                        self.class_to_id_EC[EC_nr[0]][EC_nr[1]][EC_nr[2]][EC_nr[3]]['count'] = 1

                    with open(os.path.join(csv_by_EC_path,
                                           self.class_to_id_EC[EC_nr[0]][EC_nr[1]][EC_nr[2]][EC_nr[3]]['path']),
                              "a") as out_fobj:
                        line = ";".join([name, seq, ','.join(GO),
                                         EC_nrs[0], str(helper_EC_to_id[EC_nrs[0]])])
                        line += '\n'
                        out_fobj.write(line)

        with open(os.path.join(csv_by_EC_path, 'class2id_EC.p'), "wb") as out_fobj:
            pickle.dump(self.class_to_id_EC, out_fobj, protocol=pickle.HIGHEST_PROTOCOL)

    def _valid_seq(self, seq):
        """Check a sequence for forbidden AAs and minlength.

        A helper function for filter_and_write().

        Args:
          seq: `str` the sequence to check.

        Returns:
          A `bool` whether the sequence is valid (True) or not.
        """
        if self.filter_AA and self.filter_minlength:
            forbidden_AAs = re.search(r'[BXZOUJ]', seq)
            if len(seq) >= int(self.minlength) and not forbidden_AAs:
                return True
        elif self.filter_AA and not self.filter_minlength:
            forbidden_AAs = re.search(r'[BXZOUJ]', seq)
            if not forbidden_AAs:
                return True
        elif not self.filter_AA and self.filter_minlength:
            if seq >= int(self.minlength):
                return True
        else:
            return False

def _filter_and_write(self, line, out_fobj, EC_to_write=False):
    """
    Takes a line from the uniprot/EC/GO.csv and checks the sequence for validity. Optionally writes a given term to
    all sequences passed (overrides the real terms).

    Args:
      line: `str` a line from the csv that is being filtered.
      out_fobj: `str` the filepath to the outfile.
      EC_to_write: `str` the EC to write (optional).
    """
    fields = line.strip().split(';')
    # line looks like: name, seq, go, ec, ID
    if EC_to_write:
        fields[3] = EC_to_write
        line = ';'.join(fields)
        line += '\n'

    seq = fields[1]

    if self.write_count < self.max_write:
        if self.filter_AA and self.filter_minlength:
            forbidden_AAs = re.search(r'[BXZOUJ]', seq)
            if len(seq) >= int(self.minlength) and not forbidden_AAs:
                out_fobj.write(line)
                out_fobj.flush()
                self.write_count += 1
        elif self.filter_AA and not self.filter_minlength:
            forbidden_AAs = re.search(r'[BXZOUJ]', seq)
            if not forbidden_AAs:
                out_fobj.write(line)
                out_fobj.flush()
                self.write_count += 1
        elif not self.filter_AA and self.filter_minlength:
            if seq >= int(self.minlength):
                out_fobj.write(line)
                out_fobj.flush()
                self.write_count += 1

def filter_count_and_write_all(self):
    """Takes the directory to the EC/GO class files and filters every file.

    Filters for minlength and non-canonical AAs. Every file in the directory is filered and the filtered files
    are saved in a new folder in the self.save_dir.
    """
    csv_by_EC_path_outdir = os.path.join(self.save_dir, 'csv_by_GO_structure_filtered')
    csv_by_EC_path = os.path.join(self.save_dir, 'csv_by_GO_structure')

    if not os.path.exists(csv_by_EC_path_outdir):
        os.mkdir(csv_by_EC_path_outdir)

    os.chdir(csv_by_EC_path)

    csv_files = glob.glob('*.csv')

    self.max_write = 10000000

    for csv in csv_files:
        outfilename = 'filtered%s.' % self.minlength + csv
        outfilepath = os.path.join(csv_by_EC_path_outdir, outfilename)
        print(outfilepath)

        assert outfilepath.endswith(csv)

        with open(outfilepath, "w") as outfobj:
            with open(csv, "r") as infobj:
                for line in infobj:
                    try:
                        self._filter_and_write(line, out_fobj=outfobj)
                    except IndexError:
                        print(line)
                        pass
        self.write_count = 0

def generate_dataset_by_GO_list(self, GO_file):
    """Generate the train and validsets from a given GO-File.
    Generate a dataset from the passed list of GO_terms. Sequences are included only once, even though they might
    be present in multiple GO-term files. The annotated GO-terms are filtered for the passed GO-terms and those
    not included in the passed list are omitted.

    In parallel a valid set is generated by randomly choosing 5 samples per class.

    Args:
      GO_file: A .txt file containing the GOs to be considered. One GO per line.
    """
    dataset_dir = os.path.join(self.save_dir, 'GO_datasets_filtered')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    # check which EC-classes we need to consider:
    GO_list = []
    with open(GO_file, "r") as in_fobj:
        print('Building dataset for: %s' % GO_file)
        for line in in_fobj:
            print(line.strip())
            fields = line.strip().split()
            GO_list.append(fields[1]) # is the EC class, fields[0] is the count

    # construct a dict by lvl:
    GO_dict = {}

    # determine the lvls in the GO_list:
    lvls = set()
    for go_csv in GO_list:
        lvl = go_csv.split('_')[0] # to prevent 1 matching 10 etc
        lvls.update([lvl])
    for lvl in lvls:
        GO_dict[lvl] = [go_csv for go_csv in GO_list if go_csv.startswith(lvl + '_')]

    print(GO_dict.items())

    del lvls

    dataset_dir = os.path.join(dataset_dir, 'dataset_{size}'.format(size=str(len(GO_list))))
    if __name__ == '__main__':
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

    # To construct the dataset and avoid double representation of the sequences, we iterate over the Go terms by
    # level. For each lvl we construct sets of sequences, then pick sequences to be considered in the valid set
    # then fuse the sets to the dataset
    self.train_dataset_csv_path = os.path.join(dataset_dir, 'dataset_{size}_TRAIN.csv'.format(size=str(len(GO_list))))
    self.valid_dataset_csv_path = os.path.join(dataset_dir, 'dataset_{size}_VALID.csv'.format(size=str(len(GO_list))))

    with open(self.train_dataset_csv_path, "w") as train_dataset_fobj, open(self.valid_dataset_csv_path, "w") as valid_dataset_fobj:
        # open a set() for train and one for valid()
        valid_names = set()
        train_names = set()

        #now sort the GO_list by level
        for lvl in sorted(GO_dict.keys(), reverse=True):
            #we have the .csvs listed, thats why we define a list of terms that we'll consider
            considered_terms = [go.split('_')[1].strip('.csv') for go in GO_list]
            for go_csv in GO_dict[lvl]:
                #path = os.path.join(os.path.join(self.save_dir, 'csv_by_GO_structure_filtered'), 'filtered175.%s' % go_csv)
                path = os.path.join(os.path.join(self.save_dir, 'csv_by_GO_structure'), '%s' % go_csv)
                with open(path, "r") as in_csv:
                    lines = []
                    for line in in_csv:
                        # now discard the GO-terms that are irrelevant (e.g. not in the GO_dict)
                        fields = line.strip().split(';')
                        gos = fields[2].split(',')
                        name = fields[0]
                        relevant_gos = []
                        for go in gos:
                            if go in considered_terms:
                                relevant_gos.append(go)
                        # construct a new line and append it to a list:
                        fields_to_write = [fields[0], fields[1], ','.join(relevant_gos), fields[6]]
                        line = ';'.join(fields_to_write) + '\n'
                        lines.append(line)

                    #assert len(lines) > 0, path
                    if len(lines) == 0:
                        print(path)
                        pass
                    else:
                        # select randomly 5 lines from lines
                        random_idxs = []
                        for _ in range(5):
                            index_found = False
                            while not index_found:
                                random_idx = random.randint(0, len(lines)-1)
                                if random_idx not in random_idxs:
                                    random_idxs.append(random_idx)
                                    # now store these in valid_set() and save their names
                                    line = lines[random_idx]
                                    name = line.strip().split(';')[0]
                                    if name not in valid_names:
                                        index_found = True
                                        #valid_set.update([line])
                                        valid_names.update([name])
                                        valid_dataset_fobj.write(line)

                        # write the lines to train where the name dies not match the
                        for _ in range(self.max_write):
                            wrote_line = False
                            while not wrote_line:
                                if lines: # assert that lines not empty
                                    length = len(lines)-1
                                    idx_to_write = random.randint(0, length)
                                    line = lines[idx_to_write]
                                    name = line.strip().split(";")[0]
                                    if name not in valid_names:
                                        if name not in train_names:
                                            train_names.update([name])
                                            train_dataset_fobj.write(line)
                                            wrote_line = True
                                        elif name in train_names:
                                            lines.pop(idx_to_write)
                                    elif name in valid_names:
                                        lines.pop(idx_to_write)
                                else:
                                    break

                        print('Omitted %d lines from %s.\n' % (len(lines), go_csv))
                        del lines #delete the rest

                        train_dataset_fobj.flush()
                        valid_dataset_fobj.flush()


def generate_dataset_by_EC_list(self, EC_file, path_to_EC_saves):
    """Generate the train and validsets from a given GO-File.
    Generate a dataset from the passed list of GO_terms. Sequences are included only once, even though they might
    be present in multiple GO-term files. The annotated GO-terms are filtered for the passed GO-terms and those
    not included in the passed list are omitted.

    In parallel a valid set is generated by randomly choosing 5 samples per class.

    Args:
      GO_file: `str` a .txt filepath containing the GOs to be considered. One GO per line.
      path_to_EC_saves: `str` the directory to the filtered .csvs per EC term.
    """
    dataset_dir = os.path.join(self.save_dir, 'datasets_filtered')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    # check which EC-classes we need to consider:
    EC_list = []
    with open(EC_file, "r") as in_fobj:
        print('Building dataset for: %s' % EC_file)
        for line in in_fobj:
            print(line.strip())
            fields = line.strip().split()
            EC_list.append(fields[1]) # is the EC class, fields[0] is the count

    dataset_dir = os.path.join(dataset_dir, 'dataset_{size}'.format(size=str(len(EC_list))))
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    self.train_dataset_csv_path = os.path.join(dataset_dir, 'dataset_{size}_TRAIN.csv'.format(size=str(len(EC_list))))
    self.valid_dataset_csv_path = os.path.join(dataset_dir, 'dataset_{size}_VALID.csv'.format(size=str(len(EC_list))))

    with open(self.train_dataset_csv_path, "w") as train_dataset_fobj, open(self.valid_dataset_csv_path, "w") as valid_dataset_fobj:
        count = 0
        for ec in EC_list:
            write_count = 0
            EC_nr = ec.split('.')
            # find path and write whole file to the out file
            if '-' not in EC_nr:
                with open(os.path.join(path_to_EC_saves, 'filtered175.' + self.class_to_id_EC[EC_nr[0]][EC_nr[1]][EC_nr[2]][EC_nr[3]]['path']),
                          "r") as ec_fobj:
                    # Write the first 5 lines to Valid
                    line_nr = 0
                    for line in ec_fobj:
                        line_nr += 1
                        if line_nr <=5:
                            #self._filter_and_write(line, train_dataset_fobj, ec_to_write=ec) # set the terms to filter in __init__
                            valid_dataset_fobj.write(line)
                        else:
                            if write_count <= self.max_write:
                                train_dataset_fobj.write(line)

            else:
                leaf_node_paths = self._get_leaf_nodes(EC_nr)
                for path in leaf_node_paths:
                    with open(os.path.join(path_to_EC_saves, path), "r") as ec_fobj:
                        line_nr = 0
                        for line in ec_fobj:
                            line_nr += 1
                            # edit the ec to the parentnode ec
                            fields = line.split().strip()
                            fields[3] = ec
                            line = ';'.join(fields)
                            line += '\n'

                            if line_nr <=5:
                                valid_dataset_fobj.write(line)
                            else:
                                if write_count <= self.max_write:
                                    train_dataset_fobj.write(line)

        train_dataset_fobj.flush()
        valid_dataset_fobj.flush()


def _split_traintestvalid(self, in_path):
    """Split the passed dataest into train and valid.

    Note: This function is depracted. The split is performed automatically for GO-labels.

    Args:
      in_path: `str` the path to the dataset dir.
    """
    wd = os.getcwd()
    os.chdir(os.path.dirname(in_path))
    # search for shuffled version:
    shuffled = glob.glob('*shuffled')[-1]
    if shuffled:
        print(shuffled)
        in_path = shuffled
    train_file_path = in_path + '.train'
    valid_file_path = in_path + '.valid'
    with open(in_path, "r") as in_fobj, \
            open(train_file_path, "w") as train_fobj, \
            open(valid_file_path, "w") as valid_fobj:
        # write every third line to validation-file:
        for n, line in enumerate(in_fobj):
            if n % 3 == 0:
                valid_fobj.write(line)
            else:
                train_fobj.write(line)
    os.chdir(wd)

def _get_leaf_nodes(self, EC_nr):
    """Get the leaf nodes for the passed EC-nr.

    Args:
      EC_nr: the EC_nr to get the leaf nodes for.

    Returns:
      A list of filepaths for all the node ECs.
    """
    EC_nr_toplvls = [eclvl for eclvl in EC_nr if eclvl != '-']
    depth = len(EC_nr) - len(EC_nr_toplvls) # how may iterations we need until we arrive at the leaf nodes

    # get the branch:
    curr_lvl_dict = self.class_to_id_EC
    for eclvl in EC_nr_toplvls:
        curr_lvl_dict = curr_lvl_dict[eclvl]
    file_path_list = []
    # get the filepaths:
    for path in _iter_paths(curr_lvl_dict):
        print(path)
        if path[-1] == 'path':
            item = curr_lvl_dict
            for i in range(len(path)):
                item = item[path[i]]
            file_path_list.append(item)
    return file_path_list

def _iter_paths(tree, parent_path=()):
    """iterate over a three of paths.
    Helper function for get leaf nodes.k

    Args:
      tree: `str` dir to iterate over.
      parent_path: `str` the parent_path for a current_path.

    Yields:
      The leaf path for a current tree.
    """
    tree.default_factory = None
    for path, node in tree.items():
        current_path = parent_path + (path,)
        if isinstance(node, Mapping):
            for inner_path in _iter_paths(node, current_path):
                yield inner_path
        else:
            yield current_path

def _recursively_default_dict():
    """Recursive default dict.

    Returns:
      A dict mapped to the key.
    """
    return defaultdict(_recursively_default_dict)

def _count_lines(file_path):
    """Most simple line count you can imagine.

    Returns:
      A `int` for the line_count.
    """
    count = 0
    with open(file_path, "r") as fobj:
        for line in fobj:
            count += 1
    return count

class StratifiedDict(dict):
    def __missing__(self, key):
        self[key] = {'counts': 0,
                     'id': 0,
                     'path': '',
                     }
        return self[key]

class StratifiedDictFASTA(dict):
    def __missing__(self, key):
        self[key] = {'sp_id': '',
                     'OS': '',
                     'bs_lvl': '',
                     }
        return self[key]

