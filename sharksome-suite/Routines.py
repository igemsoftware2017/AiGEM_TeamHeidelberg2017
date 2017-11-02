from mpmath import mp as math
import numpy as np

math.dps = 60

def kullbackLeiblerDivergence(sample,reference_sample):
	return -math.fsum(np.asarray(sample)*map(math.log,np.asarray(sample)/np.asarray(reference_sample)))

def Entropy(sample):
    return -math.fsum(map(math.fmul, np.asarray(sample), map(math.log, map(math.fmul, np.asarray(sample), [len(sample)]*len(sample)))))

def GoodEnergy(energy, threshold):
    if energy > threshold:
        return -1.0
    else:
        return 1.0

# Score for optimization of both Energy and Entropy
def Score(sample, beta):
	return -S(sample, beta)*min(sample)

# Score penalizing bad energies
def ScoreThreshold(sample, beta, threshold):
    result = S(sample, beta)*GoodEnergy(min(sample), threshold)
    #print("ENTROPY: %s"%result)
    return result

def ZPS(sample, beta=0.01):
	Z = math.fsum(map(math.exp, map(math.fmul, [-beta]*len(sample), np.asarray(sample))))
	P = map(math.fdiv, map(math.exp, map(math.fmul, [-beta]*len(sample), np.asarray(sample))), np.asarray([Z]*len(sample)))
	S = Entropy(P)
	return Z, P, S

def S(sample, beta=0.01):
	#print("ENTROPY: %s"%ZPS(sample, beta)[2])
	return ZPS(sample, beta)[2]

#choose the best set of positions by their free energies
def best_position(self, positions_s, free_energies):
	best_index = np.argsort(np.asarray(free_energies))
	return positions_s[best_index[0]]

#choose complex candidates by their entropies
def choose_candidates(self, entropies, sequences, threshold=0.0):
	best_sequences = []
	best_entropy_id = np.argsort(np.asarray(entropies))
	best_entropy = entropies[best_entropy_id]
	for index, entropy in enumerate(entropies):
		if entropy <= best_entropy + threshold:
			best_sequences.append(sequences[index])
	return sequences
