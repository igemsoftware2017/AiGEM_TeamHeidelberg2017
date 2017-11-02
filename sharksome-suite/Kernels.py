#Kompilierte Versionen der rechenintensiven Operationen

from numba import autojit, jit
import numpy as np
from simtk.openmm import app
from simtk import unit

def catchZero(numeric):
	return numeric + 1e-50

@jit
def rotateKernel(positions, element, axis, angle):
	x, y, z = np.asarray(axis)/(np.linalg.norm(np.asarray(axis)))
	phi_2 = angle/2.
	pos = np.array(positions[:])
	shift_forward = -pos[element[1]]
	s = np.math.sin(phi_2)
	c = np.math.cos(phi_2)
	rot = np.array([[2*(np.power(x,2)-1)*np.power(s,2)+1, 2*x*y*np.power(s,2)-2*z*c*s, 2*x*z*np.power(s,2)+2*y*c*s],
					[2*x*y*np.power(s,2)+2*z*c*s, 2*(np.power(y,2)-1)*np.power(s,2)+1, 2*z*y*np.power(s,2)-2*x*c*s],
	                [2*x*z*np.power(s,2)-2*y*c*s, 2*z*y*np.power(s,2)+2*x*c*s, 2*(np.power(z,2)-1)*np.power(s,2)+1]])
            
	for j in range(element[1],element[2]):
		pos[j] += shift_forward
            
	for j in range(element[1],element[2]):
		roted = np.dot(pos[j],rot)
		pos[j] = roted - shift_forward

	return unit.Quantity(pos, unit.angstrom)

@jit
def translateKernel(positions, element, shift):
	pos = np.array(positions)
	for j in range(element[0],element[2]):
		pos[j] += shift
	return pos

@jit
def centerOfMass(positions):
	return positions.sum(axis=0)/len(positions)

@jit
def radius(center, positions):
	return max(map(np.linalg.norm, np.asarray(positions) - np.asarray(center)))

@jit
def kullbackLeiblerDivergenceKernel(sample, reference_sample):
	return -(np.array(sample)*np.log(np.array(sample)/np.array(reference_sample))).sum()

@jit
def EntropyKernel(sample):
	#print("START_DIAGNOSIS:")
	#print("LEN_SAMPLE: %s"%len(sample))
	#print("SAMPLE: %s"%sample[0:10])
	return -(np.array(sample)*np.log(catchZero(np.array(sample)*len(sample)))).sum()

@jit
def ZPS(sample, beta=0.001):
	Z = np.exp(-beta*np.array(sample)).sum()
	#print("ZED: %s"%Z)
	P = np.exp(-beta*np.array(sample))/catchZero(Z)
	S = EntropyKernel(P)
	#print("ENTROPY: %s"%S)
	return Z, P, S

@jit
def S(sample, beta=0.001):
	return ZPS(sample, beta)[2]
