from numba import autojit, jit
import numpy as np

@jit
def nList(x, N):
	result = []
	for i in range(N):
		result.append(i**2 % x)
	return result

def List(x, N):
	result = []
	for i in range(N):
		result.append(i**2 % x)
	return result

@autojit
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
		pos[j] = roted
		pos[j] -= shift_forward
                
	return pos

def rotate(positions, element, axis, angle):
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
		pos[j] = roted
		pos[j] -= shift_forward
                
	return pos