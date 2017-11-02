import copy
import numpy as np
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from mpmath import mp as math
from collections import defaultdict
import subprocess
from Prepare import makeLib
from helpers import angle as ang
from helpers import directed_angle as d_ang
from helpers import (angstrom,nostrom,kJ,noJ)
from LoadFrom import PDBStructure
from Structure import *

## Represents a molecule chain, comprising multiple residues
class Chain(object):
    def __init__(self, Complex, Structure, sequence=None, start=0, ID=0):
        self.id = ID
        self.start = start
        self.start_history = start
        self.complex = Complex
        self.residues_start = []
        self.length = 0
        self.length_history = self.length
        self.element = [self.start, self.start+1, self.start+self.length]
        self.structure = Structure
        self.alias_sequence = ''
        self.sequence = ''
        self.sequence_array = []
        self.alias_sequence_array = []
        self.append_history = []
        self.prepend_history = []
        if sequence:
            self.alias_sequence = sequence
            self.sequence = self.structure.translate(self.alias_sequence)
            self.sequence_array = self.sequence.split(' ')
            self.alias_sequence_array = self.alias_sequence.split(' ')
            self.length = sum(map(self.structure.residue_length.__getitem__, self.sequence_array))
            self.length_history = self.length
            tally = 0
            for residue in self.sequence_array:
                self.residues_start.append(tally)
                tally += self.structure.residue_length[residue]
            self.element = [self.start, self.start+1, self.start+self.length]
        else:
            pass
    
    ## Updates adjacent chains in parent complex. Users should not touch this!
    def update_chains(self):
        length = self.length
        self.length = sum(map(self.structure.residue_length.__getitem__, self.sequence_array))
        ##print(self.length)
        self.residues_start = []
        tally = 0
        for residue in self.sequence_array:
            self.residues_start.append(tally)
            tally += self.structure.residue_length[residue]
        self.element = [self.start, self.start + 1, self.start+self.length]
        start = copy.deepcopy(self.start)
        ##print(self.element)
        for chain in self.complex.chains:
            chain.start_history = chain.start
            if chain.start >= start:
                chain.start += self.length - length
                chain.start_history += 0#self.length - length
                chain.element = [chain.start, chain.start + 1, chain.start+chain.length]
            else:
                pass
        self.start -= self.length - length
	self.start_history -= 0 #self.length - length
	self.element = [self.start, self.start + 1, self.start + self.length]
    
    ## Loads a sequence into the chain, overriding all previous sequences.
    #  Afterwards, call Complex.build() on your complex!
    def create_sequence(self, sequence):
        alias_sequence_array = sequence.split(' ')
        sequence_array = self.structure.translate(sequence).split(' ')
        for letter in sequence_array:
            if letter in self.structure.residue_names:
                pass
            else:
                raise ValueError('Residue not defined! CANNOT create sequence!')
        self.alias_sequence = sequence
        self.sequence = self.structure.translate(self.alias_sequence)
        self.alias_sequence_array = alias_sequence_array
        self.sequence_array = sequence_array
        self.update_chains()
        #print(self.length)
        self.start_history = self.start
        self.length_history = self.length
        self.sequence_array_history = self.sequence_array
        
    def append_sequence(self, sequence):
        start_history = self.start
        length = self.length
        seq_ar_length = len(self.sequence_array)
        self.create_sequence(" ".join(self.alias_sequence_array[:]+[sequence]))
        self.length_history = length
        self.start_history = start_history
        self.prepend_history = []
        self.append_history = self.sequence_array[seq_ar_length:]
        
    def prepend_sequence(self, sequence):
        length = self.length
        seq_ar_length = len(self.sequence_array)
        self.create_sequence(" ".join([sequence]+self.alias_sequence_array[:]))
        self.length_history = length
        self.start_history = self.start+self.length-length
        self.prepend_history = self.sequence_array[:len(self.sequence_array)-seq_ar_length]
        #print(self.prepend_history)
    
    ## rotates the positions corresponding to the specified element.
    #  @param element
    #  [start, bond, end]
    #  @param angle
    #  Angle in radians (0,2*PI)
    #  @param reverse
    #  specifies, whether positions IN, or OUTSIDE the element should be rotated    
    def rotate_element(self, element, angle, reverse=False):
        revised_element = element[:]
	rev = reverse
        if rev:
            if revised_element[2] == None:
                revised_element[2] = 0
            # Blocks reversed rotation, if element[2] not None
            else:
                revised_element[2] = revised_element[1]
		rev = False
            
        if len(revised_element) == 3 and revised_element[2] != None:
            revised_element = [index + self.start for index in revised_element]
            self.complex.rotate_element(revised_element, angle, reverse=rev)
        elif (len(revised_element) == 3) and (revised_element[2] == None):
            revised_element = [revised_element[0] + self.start, revised_element[1] + self.start, self.length + self.start]
            self.complex.rotate_element(revised_element, angle, reverse=rev)
        else:
            raise ValueError('Rotable element contains too many or too few components!')
    
    ## rotates a residue in the chain specified by
    #  @param residue_index
    #  integer index specifying residue to be rotated
    #  @param residue_element_index
    #  integer index specifying rotating element
    #  @param angle
    #  Angle in radians (0, 2*PI)
    def rotate_in_residue(self, residue_index, residue_element_index, angle, reverse=False):
        rev = reverse
        revised_residue_index = residue_index
        if residue_index < 0:
            revised_residue_index += len(self.sequence_array)
        element = self.structure.rotating_elements[self.sequence_array[revised_residue_index]][residue_element_index]
        #LOOK HERE!!!!!!!!!!!!! 
        for i in range(len(element)):
	    if element[i] and element[i] < 0:
                element[i] += self.structure.residue_length[self.sequence_array[revised_residue_index]]
        if element[2] == None:
            revised_element = [element[0]+self.residues_start[revised_residue_index], element[1]+self.residues_start[revised_residue_index], None]
        elif element[2] == 0:
            revised_element = [element[0]+self.residues_start[revised_residue_index],
                               element[1]+self.residues_start[revised_residue_index],
                               element[2]]
        else:
            revised_element = [element[0]+self.residues_start[revised_residue_index],
                               element[1]+self.residues_start[revised_residue_index],
                               element[2]+self.residues_start[revised_residue_index]]
            rev = False
        ##print("Revised Element: %s"%revised_element)
        self.rotate_element(revised_element, angle, reverse=rev)
    
    ## rotate_historic_element
    #  deprecated!
    def rotate_historic_element(self, historic_element, angle):
        if historic_element[2]:
            self.rotate_element([historic_element[0]+self.start_history-self.start,
                                 historic_element[1]+self.start_history-self.start,
                                 historic_element[2]+self.start_history-self.start],angle)
        else:
            self.rotate_element([historic_element[0]+self.start_history-self.start,
                                 historic_element[0]+self.start_history-self.start,
                                 None], angle)

    ## rotate_in_historic_residue
    #  deprecated!
    def rotate_in_historic_residue(self, historic_index, element_index, angle):
        offset = len(self.prepend_history)
        self.rotate_in_residue(historic_index+offset, element_index, angle)

    ## rotates the whole chain about the axis specified by axis
    def rotate_global(self, axis, angle):
        self.complex.rotate_global(self.element, axis, angle)
    
    ## translate the whole chain by shift
    def translate_global(self, shift):
        self.complex.translate_global(self.element, shift)

##Represents a complex containing multiple molecule chains.
class Complex(object):
    
    ## Specifies a complex by giving a force field governing it, defaulting to "leaprc.ff12SB"
    # @param force_field Specifies the force field governing the complex. 
    def __init__(self, force_field="leaprc.ff12SB"):
        self.build_string = """
                            source %s
                            source leaprc.gaff
                            """%(force_field)
        self.prmtop = None
        self.inpcrd = None
        self.positions = None
        self.topology = None
        self.chains = None
        self.system = None
        self.integrator = None
        self.simulation = None
            
    def add_chain(self, sequence, structure):
        if self.chains:
            start = sum([chain.length for chain in self.chains])
            chainID = len(self.chains)
        else:
            self.chains = []
            start = 0
            chainID = 0
        self.chains.append(Chain(self, structure, sequence=sequence, start=start, ID=chainID))

    def add_chain_from_PDB(self, pdb, structure=None, pdb_name='PDB', parameterized=False):
        length = makeLib(pdb, pdb_name, parameterized=parameterized)
        path = '/'.join(pdb.split('/')[:-1])
        structure = Structure([pdb_name], residue_length=[length], residue_path=path)
        self.add_chain(pdb_name, structure)
            
    def build(self, target_path="", file_name="out"):
        build_string = self.build_string 
        if self.chains:
            for chain in self.chains:
                self.build_string += chain.structure.init_string
            for index, chain in enumerate(self.chains):
                if chain.sequence:
                    self.build_string +="""
                                        CHAIN%s = sequence {%s}
                                        """%(index, chain.sequence)
                #NOCHMAL ANSEHEN
                #else:
                #    raise ValueError("Empty Chain Index: %s"%index)
            chain_string = " ".join([("CHAIN%s"%index if chain.sequence else "") for index, chain in enumerate(self.chains)])
            self.build_string +="""
                                UNION = combine {%s}
                                saveamberparm UNION %s.prmtop %s.inpcrd
                                quit
                                """%(chain_string, target_path+file_name, target_path+file_name)
            infile = open("%s%s.in"%(target_path, file_name),"w")
            infile.write(self.build_string)
            infile.close()
            self.build_string = build_string
            #os.remove("%s%s.in"%(target_path, file_name))
            result = subprocess.call("tleap -f %s%s.in"%(target_path,file_name),shell=True)
            self.prmtop = app.AmberPrmtopFile(target_path+file_name+".prmtop")
            self.inpcrd = app.AmberInpcrdFile(target_path+file_name+".inpcrd")
            self.topology = self.prmtop.topology
            self.positions = self.inpcrd.positions
            self.integrator = mm.LangevinIntegrator(300.*unit.kelvin, 1./unit.picosecond, 0.002*unit.picoseconds)
            self.system = self.prmtop.createSystem(nonbondedCutoff=5*unit.angstrom, nonbondedMethod=app.NoCutoff,
                                                   constraints=None, implicitSolvent=app.OBC1)
            self.simulation = app.Simulation(self.topology, self.system, self.integrator)
        else:
            raise ValueError('Empty Complex! CANNOT build!')
            
    def rebuild(self, target_path="", file_name="out", exclusion=[]):
        old_positions = self.positions[:]
        self.build()
        #print("EXPECTED LENGTH OF POSITIONS: %s"%len(self.positions))
        for index, chain in enumerate(self.chains):
            if not (chain in exclusion):
                pre_positions = self.positions[chain.start:chain.start_history]
                chain_positions = old_positions[chain.start:chain.start + chain.length_history]
                post_positions = self.positions[chain.start_history + chain.length_history:chain.start + chain.length]
                
                if len(pre_positions) != 0 and chain.prepend_history:
                    # Fixing positions of prepended atoms from here on:
                    
                    pre_positions = self.positions[chain.start:chain.start_history + 1]
                    pre_vector = self.positions[chain.start_history + chain.structure.connect[chain.prepend_history[-1]][1][0]] - self.positions[chain.start_history + 1]
                    old_pre_vector = old_positions[chain.start] - old_positions[chain.start + 1]
                    angle = -ang(nostrom(pre_vector), nostrom(old_pre_vector))
                    axis = np.cross(np.asarray(nostrom(pre_vector)), np.asarray(nostrom(old_pre_vector)))
                    if all(axis == np.zeros(3)):
                        axis = np.array([1.,0.,0.])
                        angle = 0
                    else:
                        axis /= np.linalg.norm(axis)
                    x, y, z = axis
                    phi_2 = angle/2.
                    pos = pre_positions[:]
                    shift_forward = mm.Vec3(0,0,0)*unit.angstroms-pos[-1+chain.structure.connect[chain.prepend_history[-1]][1][0]]
                    s = np.math.sin(phi_2)
                    c = np.math.cos(phi_2)
                    rot = np.array([[2*(np.power(x,2)-1)*np.power(s,2)+1, 2*x*y*np.power(s,2)-2*z*c*s, 2*x*z*np.power(s,2)+2*y*c*s],
                                    [2*x*y*np.power(s,2)+2*z*c*s, 2*(np.power(y,2)-1)*np.power(s,2)+1, 2*z*y*np.power(s,2)-2*x*c*s],
                                    [2*x*z*np.power(s,2)-2*y*c*s, 2*z*y*np.power(s,2)+2*x*c*s, 2*(np.power(z,2)-1)*np.power(s,2)+1]])

                    for j in range(0,len(pos)):
                        pos[j] += shift_forward

                    ## LOOK HERE!
                    # A correction for the bond length of the prepended residue / chain:
                    # The Vector connecting the old first atom of the chain to be prepended to is multiplied by
                    # the specified length of the connecting bond.

                    shift_back = chain_positions[chain.structure.connect[chain.sequence_array[len(chain.prepend_history)]][1][1]]
                    pre_bond_shift = (chain.structure.connect[chain.prepend_history[-1]][2])*old_pre_vector/np.linalg.norm(np.asarray(nostrom(old_pre_vector))) - old_pre_vector

                    for j in range(0,len(pos)):
                        roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)),rot)
                        pos[j] = mm.Vec3(roted[0],roted[1],roted[2])*unit.angstrom
                        pos[j] += shift_back + pre_bond_shift

                    pre_positions = pos[:]
                    chain_positions[0] += pre_bond_shift
                    
                    self.positions = self.positions[:chain.start] + pre_positions[:] + chain_positions[1:] + self.positions[chain.start+chain.length:]

                    # Stop fixing positions of prepended atoms.
               
                if len(post_positions) != 0 and chain.append_history:
                    # Fixing positions of appended atoms from here on:

                    post_positions = self.positions[chain.start_history + chain.length_history - 1:chain.start_history+chain.length]
                    post_vector = self.positions[chain.start_history + chain.length_history - 1] - self.positions[chain.start_history + chain.length_history - 2]
                    old_post_vector = old_positions[chain.start_history + chain.length_history - 1] - old_positions[chain.start_history + chain.length_history - 2]
                    angle = -ang(nostrom(post_vector), nostrom(old_post_vector))
                    axis = np.cross(np.asarray(nostrom(post_vector)), np.asarray(nostrom(old_post_vector)))
                    if all(axis == np.zeros(3)):
                        axis = np.array([1.,0.,0.])
                        angle = 0.
                    else:
                        axis /= np.linalg.norm(axis)
                    x, y, z = axis
                    phi_2 = angle/2.
                    pos = post_positions[:]
                    shift_forward = mm.Vec3(0,0,0)*unit.angstroms - pos[chain.structure.connect[chain.append_history[0]][0][0]]
                    s = np.math.sin(phi_2)
                    c = np.math.cos(phi_2)
                    rot = np.array([[2*(np.power(x,2)-1)*np.power(s,2)+1, 2*x*y*np.power(s,2)-2*z*c*s, 2*x*z*np.power(s,2)+2*y*c*s],
                                    [2*x*y*np.power(s,2)+2*z*c*s, 2*(np.power(y,2)-1)*np.power(s,2)+1, 2*z*y*np.power(s,2)-2*x*c*s],
                                    [2*x*z*np.power(s,2)-2*y*c*s, 2*z*y*np.power(s,2)+2*x*c*s, 2*(np.power(z,2)-1)*np.power(s,2)+1]])

                    for j in range(0,len(pos)):
                        pos[j] += shift_forward

                    ## LOOK HERE!
                    # A correction for the bond length of the prepended residue / chain:
                    # The Vector connecting the old first atom of the chain to be prepended to is multiplied by
                    # the specified length of the connecting bond.

                    post_bond_shift = (chain.structure.connect[chain.append_history[0]][2])*old_post_vector/np.linalg.norm(np.asarray(nostrom(old_post_vector))) - old_post_vector
                    shift_back = chain_positions[chain.structure.connect[chain.sequence_array[-len(chain.append_history)]][0][1]]

                    for pos_idx, pos_elem in enumerate(pos):
                        roted = np.dot(np.array(pos_elem.value_in_unit(unit.angstrom)),rot)
                        pos[pos_idx] = mm.Vec3(roted[0],roted[1],roted[2])*unit.angstrom
                        pos[pos_idx] += shift_back + post_bond_shift

                    post_positions = pos[:]
                    chain_positions[-1] += post_bond_shift
                    #print("ACTUAL_LENGTH: %s"%(len(chain_positions[:-1] + post_positions[:])))
                    #print(len(self.positions[chain.start:chain.start+chain.length]))
                    self.positions = self.positions[:chain.start] + chain_positions[:-1] + post_positions[:] + self.positions[chain.start+chain.length:]

                    # Stop fixing positions of propended atoms.

                if not (chain.append_history or chain.prepend_history):
                    self.positions = self.positions[:chain.start] + old_positions[chain.start_history:chain.start_history + chain.length_history] + self.positions[chain.start + chain.length:]
                    #print("LENGHT OF POSITIONS: %s"%len(self.positions))
                    #pass
    

            else:
                pass
            
    def rotate_element(self, element, angle, reverse=False):
        revised_element = element[:]
        if self.positions:
            pos = self.positions[:]
            vec_a = (pos[revised_element[1]]-pos[revised_element[0]])
            if revised_element[2] <= revised_element[0]:
                revised_element_1 = revised_element[1]
                revised_element[1] = revised_element[2]
                revised_element[2] = revised_element_1
            self.rotate_global(revised_element, vec_a, angle, reverse=reverse, glob=False)
        else:
            raise ValueError('This Complex contains no positions! You CANNOT rotate!')
            
    def rotate_global(self, element, axis, angle, reverse=False, glob=True):
        if self.positions:
            x, y, z = np.asarray(nostrom(axis))/(np.linalg.norm(np.asarray(nostrom(axis))))
            phi_2 = angle/2.
            pos = self.positions[:]
            starting_index = 1
            if glob:
                 starting_index = 0
            ##print(element)
            if reverse:
                 shift_forward = mm.Vec3(0,0,0)*unit.angstroms-pos[element[2]]
            else:    
                 shift_forward = mm.Vec3(0,0,0)*unit.angstroms-pos[element[starting_index]]
            s = np.math.sin(phi_2)
            c = np.math.cos(phi_2)
            rot = np.array([[2*(np.power(x,2)-1)*np.power(s,2)+1, 2*x*y*np.power(s,2)-2*z*c*s, 2*x*z*np.power(s,2)+2*y*c*s],
                            [2*x*y*np.power(s,2)+2*z*c*s, 2*(np.power(y,2)-1)*np.power(s,2)+1, 2*z*y*np.power(s,2)-2*x*c*s],
                            [2*x*z*np.power(s,2)-2*y*c*s, 2*z*y*np.power(s,2)+2*x*c*s, 2*(np.power(z,2)-1)*np.power(s,2)+1]])
            
            for j in range(element[starting_index],element[2]):
                pos[j] += shift_forward
            
            for j in range(element[starting_index],element[2]):
                roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)),rot)
                pos[j] = mm.Vec3(roted[0],roted[1],roted[2])*unit.angstrom
                pos[j] -= shift_forward
                
            positions_new = pos
            self.positions = positions_new[:]
        else:
            raise ValueError('This Complex contains no positions! You CANNOT rotate!')
        
    def translate_global(self, element, shift):
        if self.positions:
#           vec_shift = mm.Vec3(*shift)*unit.angstroms
            vec_shift = shift
            pos = self.positions[:]
            ##print(element)
            for j in range(element[0],element[2]):
                pos[j] += vec_shift
            
            positions_new = pos
            self.positions = positions_new[:]
        else:
            raise ValueError('This Complex contains no positions! You CANNOT translate!')
            
    def fit_sequence_to_chain(self, chain_id, sequence):
        #Get the chain to be fit
        chain = self.chains[chain_id]
        #Get the current array of residues
        current_sequence_array = chain.sequence_array[:]
        #Build the new array of residues
        new_alias_sequence_array = sequence.split(" ")
        new_sequence_array = chain.structure.translate(sequence).split(' ')
        #Get the current positions
        current_positions = self.positions[:]
	#Get current residue starts
	current_residues_start = chain.residues_start[:]
	#Get current chain starts
	current_chain_start = chain.start
	#Get current element
	current_chain_element = chain.element[:]
        #Build new sequence
        chain.create_sequence(sequence)
        #Rebuild the complex, excluding the chain to be fit
        self.rebuild(exclusion=[chain])
        #Translate chain to the position of the first atom to be fit
        self.translate_global(chain.element, current_positions[current_chain_start]-self.positions[chain.start])
        vec0 = self.positions[chain.element[1]] - self.positions[chain.element[0]]
        vec1 = current_positions[current_chain_element[1]] - current_positions[current_chain_element[0]]
        angle = ang(nostrom(vec0),nostrom(vec1))
        global_axis = mm.Vec3(*np.cross(np.asarray(nostrom(vec0)), np.asarray(nostrom(vec1)))) * unit.angstrom
        if (np.asarray(nostrom(global_axis)) == np.array([0.,0.,0.])).all():
		global_axis = mm.Vec3(1.,0.,0.)*unit.angstrom
		angle = 0.
	#Rotate chain to fit first rotable bond
        ##print("chain element: %s"%chain.element)
        self.rotate_global(chain.element, global_axis, -angle)
        #Iterate over residues and their rotable elements
        for residue_id, [residue, residue_current] in enumerate(zip(chain.sequence_array, current_sequence_array)):
            for element_id, (element, element_current) in enumerate(zip(chain.structure.rotating_elements[residue],
									chain.structure.rotating_elements[residue_current])):
                ##print(element)
		revised0 = element[0]+chain.start+chain.residues_start[residue_id]
                revised1 = element[1]+chain.start+chain.residues_start[residue_id]
                revised0_current = element_current[0]+current_chain_start+current_residues_start[residue_id]
                revised1_current = element_current[1]+current_chain_start+current_residues_start[residue_id]
                axis = self.positions[revised1] - self.positions[revised0]
		if not np.linalg.norm(np.asarray(nostrom(axis))) == 0:
	                axis /= np.linalg.norm(np.asarray(nostrom(axis)))
		else:
			axis = mm.Vec3(1.,0.,0.)
                #Bond to next-to-nearest neighbour along increasing indices in the element
                vec0 = self.positions[revised1+1] - self.positions[revised1]
                vec1 = current_positions[revised1_current+1] - current_positions[revised1_current]
                #Projection onto plane perpendicular to axis
                proj_vec0 = np.asarray(nostrom(vec0)) - np.dot(axis,nostrom(vec0))*nostrom(axis)
                proj_vec1 = np.asarray(nostrom(vec1)) - np.dot(axis,nostrom(vec1))*nostrom(axis)
                #Get angle between projections
                angle = d_ang(proj_vec0, proj_vec1, axis)
                #Rotate element by that angle, to align to backbone
                chain.rotate_in_residue(residue_id, element_id, -angle)

    def fit_sequence_to_chain_split_join(self, chain_id, sequence):
        #Get the chain to be fit
        chain = self.chains[chain_id]
        #Get the current array of residues
        current_sequence_array = chain.sequence_array[:]
        #Build the new array of residues
        new_alias_sequence_array = sequence.split(" ")
        new_sequence_array = chain.structure.translate(sequence).split(' ')
        #Get the current positions
        current_positions = self.positions[:]
	#Get current residue starts
	current_residues_start = chain.residues_start[:]
	#Get current chain starts
	current_chain_start = chain.start
        #Build new sequence
        chain.create_sequence(sequence)
        #Rebuild the complex, excluding the chain to be fit
        self.rebuild(exclusion=[chain])
        #Split Positions
        current_positions_split = [current_positions[chain.structure.backbone_elements[residue][idx][0] + current_residues_start[idy]:
                                                     chain.structure.backbone_elements[residue][idx][1] + current_residues_start[idy] + 2 - idx]
                                   for idy, residue in enumerate(current_sequence_array) for idx in [0,1]]

        positions_split = [self.positions[chain.structure.backbone_elements[residue][0][1] + chain.residues_start[idy]:
                                          chain.structure.backbone_elements[residue][1][0] + chain.residues_start[idy]]
                           for idy, residue in enumerate(new_sequence_array)]

        positions = []
        for index, residue in enumerate(chain.sequence_array):
            #substituent -= substituent[0]
            subst = positions_split[index]
            vec0 = subst[chain.structure.backbone_elements[residue][0][2]-chain.structure.backbone_elements[residue][0][1]]-subst[0]
            vec1 = current_positions[current_residues_start[index] + chain.structure.backbone_elements[current_sequence_array[index]][0][2]] - current_positions[current_residues_start[index] + chain.structure.backbone_elements[current_sequence_array[index]][0][1]] # - current_positions_split[2*index][-2]
            #vec1 = current_positions[current_residues_start[index] + chain.structure.backbone[residue][0][1]][-1] - current_positions_split[2*index][-2]
            axis = np.cross(np.asarray(nostrom(vec0)), np.asarray(nostrom(vec1)))
            #print("axis: %s"%axis)
            if (axis == np.zeros(3)).all():
                axis = np.array([1.,0.,0.])
                angle = 0.
            else:
                axis /= np.linalg.norm(axis)
                angle = -d_ang(np.asarray(nostrom(vec0)), np.asarray(nostrom(vec1)), axis)
            #print("angle: %s"%angle)
            x, y, z = axis
            s = np.math.sin(angle/2.)
            c = np.math.cos(angle/2.)
            rot = np.array([[2*(np.power(x,2)-1)*np.power(s,2)+1, 2*x*y*np.power(s,2)-2*z*c*s, 2*x*z*np.power(s,2)+2*y*c*s],
                            [2*x*y*np.power(s,2)+2*z*c*s, 2*(np.power(y,2)-1)*np.power(s,2)+1, 2*z*y*np.power(s,2)-2*x*c*s],
                            [2*x*z*np.power(s,2)-2*y*c*s, 2*z*y*np.power(s,2)+2*x*c*s, 2*(np.power(z,2)-1)*np.power(s,2)+1]])
            

            shift_forward = subst[chain.structure.backbone_elements[residue][0][2] - chain.structure.backbone_elements[residue][0][1]]
            for j in range(chain.structure.backbone_elements[residue][0][2] - chain.structure.backbone_elements[residue][0][2], len(subst)):
                subst[j] -= shift_forward
                roted = np.dot(np.array(subst[j].value_in_unit(unit.angstrom)),rot)
                subst[j] = mm.Vec3(roted[0],roted[1],roted[2])*unit.angstrom
                subst[j] += current_positions[current_residues_start[index] + chain.structure.backbone_elements[current_sequence_array[index]][0][2]]
            
            #print(vec1-subst[1]+subst[0])

            #substituent += current_positions_split[2*index][-1]
            for position in current_positions_split[2*index][:]:
                positions.append(position)
            for position in subst[2:]:
                positions.append(position)
            for position in current_positions_split[2*index + 1]:
                positions.append(position)

        #print("%s = %s"%(vec0, vec1))
        #print("%s = %s"%(len(positions), len(self.positions)))
        #assert(len(positions) == len(self.positions))
        for idx, position in enumerate(positions):
            self.positions[idx] = position

    #get current complex energy
    def get_energy(self):
        self.simulation.context.setPositions(self.positions)
        state = self.simulation.context.getState(getPositions=True,getEnergy=True,groups=1)
        free_E = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        return free_E, self.positions
    
    #Wrapper for OpenMM local energy minimization
    def minimize(self, max_iterations=100):
        self.simulation.context.setPositions(self.positions)
        self.simulation.minimizeEnergy(maxIterations=max_iterations)
        state = self.simulation.context.getState(getPositions=True,getEnergy=True,groups=1)
        self.positions = state.getPositions()
        free_E = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        return free_E

    #Wrapper for OpenMM MD steps
    def step(self, number_of_steps):
        self.simulation.step(number_of_steps)
        self.positions = self.simulation.context.getPositions()
        return self.get_energy()

    def rigid_minimize(self, max_iterations=100, max_step_iterations=100):
        energy = None
        for i in range(max_iterations):
            for chain in self.chains:
                for idx, residue in enumerate(chain.sequence_array):    
                    for j in range(max_step_iterations):
                        positions = self.positions[:]
                        chain.rotate_in_residue(idx, np.random.choice([elem for elem in range(len(chain.structure.rotating_elements[residue]))]),
                                                     np.random.uniform(-np.math.pi, np.math.pi))
                        free_E = self.get_energy()[0]
                        if free_E < energy or energy == None:
                            energy = free_E
                            #print(energy)
                            self.positions = positions[:]
    
    #Introducing 'chain-wriggling' into the framework
    def pert_min(self, size=1e-1, iterations=50):
        for repeat in range(iterations):
            for i in range(len(self.positions)):
                self.positions[i] += np.random.uniform(-size,size,3)*unit.angstrom
            self.minimize()
