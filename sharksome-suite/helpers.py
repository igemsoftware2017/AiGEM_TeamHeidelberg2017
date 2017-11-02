import numpy as np
from simtk import unit

def angstrom(array):
    return array*unit.angstrom

def nostrom(quantity):
    return quantity.value_in_unit(unit.angstrom)

def kJ(array):
    return array*unit.kilojoules_per_mole

def noJ(quantity):
    return quantity.value_in_unit(unit.kilojoules_per_mole)

def angle(array1, array2):
    return np.arccos(np.clip(
            np.dot(np.array(array1),np.array(array2))/(np.linalg.norm(np.array(array1))*np.linalg.norm(np.array(array2))),
            -1,1))

def directed_angle(array1, array2, axis):
    n_array1 = np.asarray(array1)
    n_array1 /= np.linalg.norm(n_array1)
    n_array2 = np.asarray(array2)
    n_array2 /= np.linalg.norm(n_array2)
    return np.arctan2(np.dot(np.asarray(axis),
                             np.cross(np.asarray(array1), 
                                      np.asarray(array2))),
                      np.dot(np.asarray(array1),
                             np.asarray(array2)))

