import sys
import os
import pickle
# import numpy as np



def plot(data, data_2, seq_count, file, original):
    mutated = []
    for el_dict_idx in range(len(data)):
        mut = False
        for entry in data[el_dict_idx]:
            if data[el_dict_idx][entry] != seq_count and data[el_dict_idx][entry] != 0: # leave out seq
                mut = True
        if mut:
            mutated.append(el_dict_idx)

    sum_total_muts = 0
    for el in data_2:
        sum_total_muts += data_2[el]

    for idx in mutated:
        amount_of_total_mutations = data_2['{}{}'.format(original[idx], idx+1)] /sum_total_muts
        file.write('\n\nPosition: {}\nOriginal: {}\nAmount of total mutations: {}\n'.format(idx+1, original[idx], amount_of_total_mutations))
        amount = {}
        for el in data[idx]:
            amount[el] = int(data[idx][el])
        amount[original[idx]] = 0
        for el in data[idx]:
            sum_amounts = 0
            for a in amount:
                sum_amounts += amount[a]
            file.write('{}: {}\n'.format(el, (amount[el])/sum_amounts))



rdir = sys.argv[1]
ldir = os.path.join(sys.argv[1], 'parser_log.txt')
until = sys.argv[2]

logfile = open(ldir, 'w')
logfile.write('Starting with read-directory "{}", {} generations.\n'.format(rdir, until))
logfile.flush()
print('Starting with logfile "{}", read-directory "{}", until generation {}.\n'.format(ldir, rdir, until))
dirs = os.listdir(rdir)

for dir in dirs:
    try:
        if dir.endswith('.txt') or dir.endswith('.p'):
           continue
        logfile.write('Reading {}.\n'.format(dir))
        logfile.flush()
        with open(os.path.join(rdir, dir, 'logfile.txt')) as ifile, open(os.path.join(rdir, dir, 'parsed_{}.txt'.format(until)), 'w') as ofile: # took away, dir,
            for line in ifile:
                if line.startswith('Starting'):
                    break

            startseq = ifile.readline()

            aa_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
            data = [aa_dict.copy() for _ in range(len(startseq))]
            data_2 = {}
            next = False
            mutations = False
            next_mutations = False
            seq_count = 0
            current_muts = []
            for line in ifile:
                if seq_count == int(until)*5:
                    break
                if next:
                    next = False
                    if not ':' in line:
                        seq = line.strip()
                        seq_count += 1
                        for aa_idx in range(len(seq)):
                            data[aa_idx][seq[aa_idx].upper()] += 1

                if next_mutations:
                    next_mutations = False
                    muts =  line.strip().split(',')[:-1]
                    for mut in muts:
                        mut = mut.strip()[:-1]
                        if not mut in current_muts:
                            try:
                                data_2[mut] += 1
                            except:
                                data_2[mut] = 1
                    current_muts = muts
                if line.startswith(('Mutations:')):
                    next_mutations = True
                if line.startswith('Mutated:'):
                    next = True
            ofile.write('{}:\n'
                        'found {} sequences until generation {}.\n\n'.format(os.path.join(rdir, dir), seq_count, until))
            pickle.dump(data, open(os.path.join(rdir,  'parsed_{}.p'.format(until)), 'wb'))
            plot(data, data_2, seq_count, ofile, startseq)

            logfile.write('Dumped pickle.\n\n\n')
            logfile.flush()
    except:
        print('Faild in {}\n'.format(os.path.join(rdir, dir, 'logfile.txt')))

