import RNA
import random
import math
from operator import itemgetter
import re

## Build complement of a single nucleotide
def ntCom(char):
    if char in "aA":
        return "T"
    elif char in "tTuU":
        return "A"
    elif char in "gG":
        return "C"
    elif char in "cC":
        return "G"
    elif char in "nN":
        return "N"
    else:
        raise ValueError("Not a nucleic acid sequence")

## Build the reverse complement of <seq>
def revComp(seq):
    result = ""
    for char in seq:
        result += ntCom(char)
    result = result[::-1]
    return result

## Iterate over closed
def c_iter(ter):
    c_array = []
    for a in ter:
        c_array += [[a.start(),len(a.group())]]
    return c_array

## Iterate over open
def o_iter(ter):
    o_array = []
    for a in ter:
        o_array += [[a.start(),len(a.group())]]
    return o_array

# generate array of closing bonds
def build_clozed(array_nested):
    array = []
    for elem in array_nested:
        array = [elem[0]+elem[1]-i for i in range(1,elem[1]+1)] + array
    return array

# generate array of opening bonds
def build_open(array_nested):
    array = []
    for elem in array_nested:
        array = array + [elem[0]+i for i in range(elem[1])]
    return array

# generate arrays of 5' and 3' pairing bases
def bond(structure):
    open_are = "(\(\(*)"
    clozed_are = "(\)\)*)"
    open = re.compile(open_are)
    clozed = re.compile(clozed_are)
    citer = c_iter(clozed.finditer(structure))
    oiter = o_iter(open.finditer(structure))
    t_prime = build_clozed(citer)
    f_prime = build_open(oiter)
    return f_prime, t_prime

# find bonds consistend amongst any number of secondary structures
def consistendBonds(structures):
    fPrime = []
    tPrime = []
    consistentBonds = []
    allBonds = [[[ttp, tfp] for ttp, tfp in zip(*bonds(struc))] for struc in structures]
    for idx, bonds in enumerate(allBonds):
        for bond in bonds:
            for bonds in allBonds[:idx]+allBonds[idx+1:]:
                if bond in bonds and not bond in consistentBonds:
                    consistentBonds.append(bond)
    for a, b in consistentBonds:
        fPrime.append(a)
        tPrime.append(b)
    return fPrime, tPrime

#RNA.cvar.fold_constrained = 1

# weighted choice over <list> wieghted by <counts>
def WeightedChoice(list,counts):
        population = [val for val, cnt in zip(list,counts) for i in range(cnt)]
        return random.choice(population)

# Mutate a single <letter> according to an <alphabet> weighted by <probabilities>  
def MutateLetter(letter, alphabet, probabilities):
        smallest_p = min(probabilities)
        exponent = math.ceil(-math.log10(smallest_p))
        counts = [int(probability*10**exponent) for probability in probabilities]
        return WeightedChoice(alphabet, counts)

# Mutate a <string>, using the <alphabet> weighted by <probabilities>, leaving all
# nucleotides not marked by "N" in <sequence> invariant, with probability of mutation
# 1 - <normalizedBeta>.
def Mutate(string, alphabet, probabilities, invariantSequence=None, pMutation=1):
        if pMutation == 0:
            return string
        pMut = int((pMutation)*10**math.ceil(-math.log10(pMutation)))
        pNMut = int(10**math.ceil(-math.log10(pMutation)) - pMut)
        if invariantSequence != None:
            result = list(string)
            for idx, ntide in enumerate(invariantSequence):
                if ntide != "N" or WeightedChoice([0, 1], [pNMut, pMut]) == 0:
                    continue
                result[idx] = MutateLetter(string[idx], alphabet, probabilities)
            result = ''.join(result)
        else:
            if pMutation == 1:
                mutate = lambda x : MutateLetter(x, alphabet, probabilities)
                result = ''.join(map(mutate, string))
            else:
                result = list(string)
                for idx in range(len(string)):
                    if WeightedChoice([0, 1], [pNMut, pMut]) == 0:
                        continue
                    else:
                        result[idx] = MutateLetter(result[idx], alphabet, probabilities)
                result = ''.join(result)
        return result

# As above, but using mutations of consistent bonds over <structures>
def ConsistentMutate(string, alphabet, probabilities, structures,
                     invariantSequence=None, pMutation=1):
        if pMutation == 0:
            return string
        result = list(string)
        pMut = int((pMutation)*10**math.ceil(-math.log10(pMutation)))
        pNMut = int(10**math.ceil(-math.log10(pMutation)) - pMut)
        if structures == None:
            return string
        if len(structures) == 1:
            bonds = bond(structures[0])
        else:
            bonds = consistentBonds(structures)
        for idx, ntide in enumerate(string):
            if idx in bonds[0] or WeightedChoice([0, 1], [pNMut, pMut]) == 0:
                continue
            elif idx in bonds[1]:
                result[idx] = MutateLetter(result[idx], alphabet, probabilities)
                result[bonds[0][bonds.index(idx)]] = ntCom(result[idx])
            else:
                result[idx] = MutateLetter(result[idx], alphabet, probabilities)
        result = ''.join(result)
        return result

# Generate a random string of <length> from <alphabet> weighted by <probabilities>
def RandomString(length, alphabet, probabilities):
        smallest_p = min(probabilities)
        exponent = math.ceil(-math.log10(smallest_p))
        counts = [int(probability*10**exponent) for probability in probabilities]
        return ''.join([WeightedChoice(alphabet, counts) for i in range(length)])

# As above, but leaving a <sequence> invariant, and conforming to <structure>
def RandomInvariantString(structure, sequence, alphabet, probabilities):
        smallest_p = min(probabilities)
        exponent = math.ceil(-math.log10(smallest_p))
        counts = [int(probability*10**exponent) for probability in probabilities]
        bonds = bond(structure)
        result = list(sequence)
        for idx, ntide in enumerate(sequence):
            if ntide != "N":
                continue
            if idx in bonds[1]:
                result[idx] = ntCom(result[bonds[0][bonds[1].index(idx)]])
            else:
                result[idx] = WeightedChoice(alphabet, counts)
        result = ''.join(result)
        return result 

# As above, no sequence constraints
def RandomStructuredString(structure, alphabet, probabilities):
        return RandomInvariantString(structure, "N"*len(structure), probabilities)

# Create a list of <Nsamples> random strings
def PopulateRandomStrings(length, alphabet, probabilities, Nsamples):
        return [RandomString(length, alphabet, probabilities) for i in range(Nsamples)]

# as above, invariant strings
def PopulateInvariantStrings(structure, sequence, alphabet, probabilities, Nsamples):
        return [RandomInvariantString(structure, sequence, alphabet, probabilities) for i in range(Nsamples)]

# as above, strings conforming to a structure
def PopulateStructuredStrings(structure, alphabet, probabilities, Nsamples):
        return [RandomInvariantString(structure, alphabet, probabilities) for i in range(Nsamples)]

# Evolve a population according to a <fitness_function> (Quasi Clonal Selection)
def EvolveStrings(population, fitness_function, alphabet, probabilities,
                  Nsamples, fuzz=0, sequence=None, structures=None, pMutation=[0, 0]):
        fitness_map = [fitness_function(element) for element in population]
        ranking = sorted(enumerate(fitness_map), key=itemgetter(1))
        indices = [element[0] for element in ranking]
        fitness = [element[1] for element in ranking]
        strings = [population[i] for i in indices[0:fuzz+1]]
        counts = [int(math.floor(fitness[i]/(sum(fitness[0:fuzz+1])+1e-6)*Nsamples)) for i in range(fuzz+1)]
        counts[0] += Nsamples-sum(counts)
        mutate = lambda x : Mutate(x, alphabet,
                                   probabilities,
                                   invariantSequence=sequence,
                                   pMutation=pMutation[0])
        consistentMutate = lambda x : ConsistentMutate(x, alphabet,
                                             probabilities,
                                             structures,
                                             invariantSequence=sequence,
                                             pMutation=pMutation[1])
        result = [string for string, cnt in zip(strings,counts) for i in range(cnt-1)]
        result = list(map(consistentMutate, result))
        result = list(map(mutate, result))
        result.append(population[indices[0]])
        return result

# Lots of Fitness Functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def ConstrainBonded(sequence, constraint):
    structure, energy = RNA.pf_fold(sequence)
    matrix = [[RNA.get_pr(i+1,j+1) for i in range(len(sequence))] for j in range(len(sequence))]
    score = 0
    for alem, blem in zip(*bond(constraint)):
        score -= matrix[alem][blem]
    return score

def ConstrainNumberOfBonded(sequence, constraint, threshold=1e-6):
    structure, energy = RNA.pf_fold(sequence)
    matrix = [[RNA.get_pr(i+1,j+1) for i in range(len(sequence))] for j in range(len(sequence))]
    score = 0
    for alem, blem in zip(*bond(constraint)):
        score -= 1 if matrix[alem][blem] >= threshold else -1
    return score


def ConstrainEnergy(sequence):
    structure, energy = RNA.pf_fold(sequence)
    return energy

def ConstrainEntropy(sequence, position):
    structure, energy = RNA.pf_fold(sequence)
    matrix = [[RNA.get_pr(i+1,j+1) for i in range(len(sequence))] for j in range(len(sequence))]
    score = 0
    for j in range(len(sequence)):
        score -= matrix[position][j] * math.log((matrix[position][j]+1e-12) * len(seq))
    return score

def ConstrainRelativeEntropy(sequence, probabilityMatrix):
    structure, energy = RNA.pf_fold(sequence)
    matrix = [[RNA.get_pr(i+1,j+1) for i in range(len(sequence))] for j in range(len(sequence))]
    score = 0
    for j in range(len(sequence)):
        for i in range(len(sequence)):
            score -= matrix[i][j] * math.log((matrix[i][j]+1e-12)
                                           / (probabilityMatrix[i][j]+1e-12))
    return score

def ConstrainNonbonded(sequence, constraint):
    structure, energy = RNA.pf_fold(sequence)
    matrix = [[RNA.get_pr(i+1,j+1) for i in range(len(sequence))] for j in range(len(sequence))]
    score = 0
    nonbonded = [idx for idx, marker in enumerate(constraint) if marker in ".x"]
    for idx in nonbonded:
        score += sum(matrix[idx])
    return score

def ConstrainStructure(sequence, constraint):
    return ConstrainBonded(sequence, constraint) + ConstrainNonbonded(sequence, constraint)

def ConstrainGC(sequence, GC):
    return abs(GC - sum([1./len(sequence) for ntide in sequence if ntide in "GCgc"])*100)

def ConstrainAT(sequence, AT):
    return abs(AT - sum([1./len(sequence) for ntide in sequence if ntide in "ATat"])*100)

def ConstrainBase(sequence, position, variants, bonus=0, malus=1):
    if sequence[position] in variants:
        return bonus
    else:
        return malus

def ConstrainSingleRepeat(sequence, maxRep, base=None):
    longestRepeat = 0
    currentRepeat = 0
    badness = 0
    nRepGTMaxRep = 0
    countedQ = False
    lastNtide = ""
    if base == None:
        doCount = lambda x: True
    else:
        doCount = lambda x: x == base
    for ntide in sequence:
        if ntide == lastNtide and doCount(ntide):
            currentRepeat += 1
            if currentRepeat > longestRepeat:
                longestRepeat = currentRepeat
        else:
            currentRepeat = 0
            lastNtide = ntide
            countedQ = False
        if currentRepeat > maxRep and not countedQ:
            nRepGTMaxRep += 1
            badness += 1
            countedQ = True
        else:
            badness += 1
    return longestRepeat, badness, nRepGTMaxRep

    
def choose(population, fitness_function, fuzz=0):
    fitness_map = [fitness_function(element) for element in population]
    ranking = sorted(enumerate(fitness_map), key=itemgetter(1))
    indices = [element[0] for element in ranking]
    fitness = [element[1] for element in ranking]
    strings = [population[i] for i in indices[0:fuzz+1]]
    return strings

    
def StructureFitness(string, constraint):
        result = ConstrainStructure(string, constraint)
        return result+1e-6

def DistanceFitness(string, constraint1, constraint2):
        return abs(RNA.pf_fold(string, constraint2)[1]-RNA.pf_fold(string, constraint1)[1])

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Lots of Fitness Functions

