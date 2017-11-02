from collections import defaultdict
import numpy as np
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit

## @class Structure
# Defines the chemical properties for molecules based on residues. 
class Structure(object):
    
    def __init__(self, residue_names, residue_length=None,
                    rotating_elements=None, backbone_elements=None, connect=None, 
                    residue_path=None, alias=None):
        self.residue_names = residue_names
        self.residue_path = residue_path
        self.init_string = """
        """
        if self.residue_path != None:
            for name in self.residue_names:
                self.init_string += """
                loadoff %s.lib
		loadamberparams %s.frcmod 
                """%(residue_path + "/" + name, residue_path + "/" + name)
        else:
            pass    
        
        self.residue_length = defaultdict(lambda : 0)
        
        if residue_length:
            for index, residue in enumerate(self.residue_names):
                self.residue_length[residue] = residue_length[index]
        
        self.connect = defaultdict(lambda : [[0,-1],[-2,0],1.6,1.6])
        
        if connect:
            for index, residue in enumerate(self.residue_names):
                self.connect[residue] = connect[index]

        self.alias = defaultdict(lambda : None)
        if self.residue_names:
            for residue_name in self.residue_names:
                self.alias[residue_name] = [residue_name]*4
        if alias:
            for element in alias:
                self.alias[element[0]] = element[1:]

        """
        Array of tuples, containing the first atom of a rotating unit,
        and the last atom rotating with it as a whole.
        If a last atom does not exist, the second component should be None.
        """
        self.rotating_elements = defaultdict(lambda : None)
        for name in residue_names:
            self.rotating_elements[name] = [None]
        
	#print("rotating_elements? "+str(rotating_elements==None))

        if rotating_elements:
            for residue, start, bond, end in rotating_elements:
                revised_start = start
                revised_bond = bond
                revised_end = end
                if start < 0:
                    revised_start += self.residue_length[residue]
                if bond < 0:
                    revised_bond += self.residue_length[residue]
                if end != None and end < 0:
                    revised_end += self.residue_length[residue]
                if self.rotating_elements[residue] == [None]:
                    self.rotating_elements[residue] = [[start, bond, end]]
                elif self.rotating_elements[residue] == None:
                    raise ValueError('Residue does not exist! CANNOT assign rotability!')
                else:
                    self.rotating_elements[residue].append([start, bond, end])

        self.backbone_elements = defaultdict(lambda : None)
        if backbone_elements:
            for residue, start, middle_pre, bond, middle_post, end in backbone_elements:
                revised_start = start
                revised_middle_pre = middle_pre
                revised_bond = bond
                revised_middle_post = middle_post
                revised_end = end
                if start < 0:
                    revised_start += self.residue_length[residue]
                if middle_pre < 0:
                    revised_middle_pre += self.residue_length[residue]
                if bond < 0:
                    revised_bond += self.residue_length[residue]
                if middle_post < 0:
                    revised_middle_post += self.residue_length[residue]
                if end < 0:
                    revised_end += self.residue_length[residue]
                self.backbone_elements[residue] = [[revised_start, revised_middle_pre, bond], [revised_middle_post, revised_end]]

    def add_rotation(self, residue_name, rotations):
        """
            add_rotation:
            -------------
            A function for specifying new rotable joints for residue types in this Chemistry.
        """
        if self.rotating_elements[residue_name] == [None]:
            self.rotating_elements[residue_name] = []
        if isinstance(rotations[0],basestring):
            for rotation in rotations:
                self.rotating_elements[residue_name].append(rotation)
        elif isinstance(rotations, basestring):
            self.rotating_elements[residue_name].append(rotation)
        else:
            raise ValueError("The input supplied is not a three component list specifying a rotation!")
        return self.rotating_elements

    def translate(self, sequence):
        sequence_array = sequence.split(' ')
        if len(sequence_array) == 1:
            return self.alias[sequence_array[0]][0]
        else:
            return " ".join([self.alias[sequence_array[0]][1]]
                + [aliasElement[2] for aliasElement in map(self.alias.__getitem__, sequence_array)][1:-1]
                + [self.alias[sequence_array[-1]][3]])
