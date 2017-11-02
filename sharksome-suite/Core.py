import numpy as np
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from mpmath import mp as math
from collections import defaultdict
import subprocess
from helpers import angle as ang
from helpers import (angstrom,nostrom,kJ,noJ)

#Required for: Structure
import xml.etree.ElementTree as XMLTree

#numpy compatible fast cartesian product
from sklearn.utils.extmath import cartesian

## @class Represents a space, from which samples can be taken
class Space(object):
    
    ## Creates a new Space 
    # @param membership_boolean_function
    # Specifies a function reference, specifying which elements are contained in the Space.
    # @param lower_bound
    # Specifies the lower bound of elements to check for, if they are in the Space.
    # @param upper_bound
    # Specifies the upper bound of elements to check for, if they are in the Space.
    # @param units
    # Specifies the used length unit. Typically angstroms. 
    def __init__(self, membership_boolean_function, units=unit.angstroms, lower_bound=-1e6, upper_bound=1e6):
        self.is_in = membership_boolean_function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.units = units
        self.volume = 0
    
    ##Returns an element of the space
    # @returns 3D coordinates of the element
    def generator(self):
        result = None
        while result == None:
            candidate = np.random.uniform(self.lower_bound,self.upper_bound,3)
            if self.is_in(candidate):
                result = candidate
        return result
        
## @class Box
# Represents a 3D box sampling space oriented along the coordinate system
class Box(Space):
    ## Creates a new Space 
    # @param x_width
    # Specifies the width in x-direction
    # @param y_width
    # Specifies the width in y-direction
    # @param z_width
    # Specifies the width in z-direction
    # @param centre
    # Specifies the position of the box via its centre
    # @param units
    # Specifies the used length unit. Typically angstroms. 
    def __init__(self, x_width, y_width, z_width, centre, units=unit.angstroms):
        def boolean(position):
            x_in = (position[0] <= centre[0]+x_width/2 and position[0] >= centre[0]-x_width/2)
            y_in = (position[1] <= centre[1]+y_width/2 and position[1] >= centre[1]-y_width/2)
            z_in = (position[2] <= centre[2]+z_width/2 and position[2] >= centre[2]-z_width/2)
            return (x_in and y_in and z_in)
        super(Box, self).__init__(boolean)
        self.x_width = x_width
        self.y_width = y_width
        self.z_width = z_width
        self.centre = centre
        self.volume = x_width*y_width*z_width*(2*np.math.pi)**3
    
    ##Returns an element of the space (Box)
    # @returns 3D coordinates of the element    
    def generator(self):
        axis = np.random.uniform(-1,1,3)
        x,y,z = axis/np.linalg.norm(axis)
        protation_angle = np.random.uniform(0, np.math.pi)
        return np.array([np.random.uniform(self.centre[0]-self.x_width/2, self.centre[0]+self.x_width/2),
                       np.random.uniform(self.centre[1]-self.y_width/2, self.centre[1]+self.y_width/2),
                       np.random.uniform(self.centre[2]-self.z_width/2, self.centre[2]+self.z_width/2),
                       x, y, z, rotation_angle])
## @class Cube
# Represents a 3D cubical sampling space    
class Cube(Box):
    ## Creates a new Space 
    # @param width
    # Specifies the width in all directions
    # @param centre
    # Specifies the position of the box via its centre
    # @param units
    # Specifies the used length unit. Typically angstroms. 
    def __init__(self, width, centre, units=unit.angstroms):
        super(Cube, self).__init__(width, width, width, centre, units=unit.angstroms)

## @class Sphere
# Represents a 3D spherical sampling space        
class Sphere(Space):
    ## Creates a new Space 
    # @param radius
    # Specifies the radius of the sphere
    # @param centre
    # Specifies the position of the box via its centre
    # @param units
    # Specifies the used length unit. Typically angstroms. 
    def __init__(self, radius, centre, units=unit.angstroms):
        def boolean(position):
            position.abs() <= radius 
        super(Sphere, self).__init__(boolean)
        self.radius = radius
        self.centre = centre
        self.volume = 4./3.*np.math.pi*radius**3*(2*np.math.pi)**3

    ##Returns an element of the space (Sphere)
    # @returns 3D coordinates of the element 
    def generator(self):
        axis = np.random.uniform(-1,1,3)
        x,y,z = axis/np.linalg.norm(axis)
        r = np.random.uniform(0,radius)
        phi = np.random.uniform(0, 2*np.math.pi)
        psi = np.random.uniform(0, np.math.pi)
        result = r*np.array([np.math.cos(phi)*np.math.sin(psi),
                             np.math.sin(phi)*np.math.sin(psi),
                             np.math.cos(psi), x, y, z,
                             rotation_angle])
        return result
    
## @class Sampler
# Abstract class providing the basic functionality for stochastic sampling.
# A sample() call increments the "achieved" counter for easy statistical analysis.
### Implement parallelization capabilities
class Sampler(object):
    #FUTURE PLEASE: make me serializable/implement distribution helpers

    ## Creates a sampler instance
    # @param space
    # Specifies the space (.space) which shall be sampled. This may be a 3D space for molecular dynamic simulation or a self defined mathematical space for different sampling purposes.
    # @param size
    # Specifies the sampling size over which shall be sampled in a Sampler.run() call
    # @param sample_function
    # References a function returning an object, which is appended to the .samples List.
    def __init__(self, space,  size=0, sample_function=lambda : None):
        self.space = space
        self.sample_function = sample_function
        self.samples = []
        self.size = size
        self.achieved = 0
        
    ## Performs a single sampling operation
    # Inherited functions MUST implement a self.achieved += 1 statement!
    def sample(self):
        self.achieved += 1
        self.samples.append(self.sample_function())
        return 0
    
    ## Performs a sampling run over the space.
    # The sampling size can either be defined by the sampling_size parameter or is obtained from the Sampler.size property.
    # Remember to reset the Sampler.achieved variable after a run when reusing a Sampler instance.
    # @param sampling_size
    # Specifies an explicit sampling size for this run. 
    def run(self, sampling_size=0):
        sampling_size = sampling_size if (sampling_size !=0) else self.size
        while self.achieved < sampling_size:
            self.sample()
        return 0

## @class EnergySampler
# Class providing basic functionality for MD energy sampling based on Sampler
# Technology based on the openMM Infrastructure    
class EnergySampler(Sampler):
    ## Creates a new EnergySampler
    # @param Complex
    # Hands over the complex providing the information necessary for calculations
    # @param space
    # Spezifies the 3D Space to be sampled
    # @param size
    # Defines the sampling size to be taken.
    def __init__(self, Complex, space, size):
        super(Sampler, self).__init__(space, size=size)
        self.complex = Complex
        self.prmtop = Complex.prmtop
        self.inpcrd = Complex.inpcrd
        self.positions = self.inpcrd.positions[:]
        self.topology = self.prmtop.topology
        self.ligand_range = Complex.ligand_range
        self.aptamer_positions = self.positions[ligand_range[1]-1:]
        self.ligand_positions = self.positions[:ligand_range[1]]
        self.system = self.prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, implicitSolvent=app.OBC1)
        self.integrator = mm.LangevinIntegrator(300.*unit.kelvin, 1./unit.picosecond, 0.002*unit.picoseconds)
        self.simulation = app.Simulation(self.topology, self.system, self.integrator)
        
    # REVISE 

    def transform_complex(transformation_array):
        shift_vector = transformation_array[:3]
        x,y,z = transformation_array[3:-1]
        phi_by_two = transformation_array[-1]/2.
        s = np.math.sin(phi_by_two)
        c = np.math.cos(phi_by_two)
        rotation_matrix = np.array([[2*(np.power(x,2)-1)*np.power(s,2)+1, 2*x*y*np.power(s,2)-2*z*c*s, 2*x*z*np.power(s,2)+2*y*c*s],
                                    [2*x*y*np.power(s,2)+2*z*c*s, 2*(np.power(y,2)-1)*np.power(s,2)+1, 2*z*y*np.power(s,2)-2*x*c*s],
                                    [2*x*z*np.power(s,2)-2*y*c*s, 2*z*y*np.power(s,2)+2*x*c*s, 2*(np.power(z,2)-1)*np.power(s,2)+1]])
        aptamer_position_dummy = self.aptamer_positions[:]
        ligand_position_dummy = self.ligand_positions[:]
        drift = self.aptamer_positions.mean(axis=0)
        ligand_drift = self.ligand_positions.mean(axis=0)
        aptamer_position_dummy -= drift
        ligand_position_dummy -= drift
        aptamer_position_dummy = np.tensordot(rotation_matrix, aptamer_position_dummy, axis=0)
        aptamer_position_dummy += shift_vector
        self.positions = ligand_position_dummy + aptamer_position_dummy
    
    ##"private" function calculating energy of the complex 
    def energy_function(self):
        self.simulation.context.setPositions(self.positions)
        state = simulation.context.getState(getPositions=True,getEnergy=True,groups=1)
        free_E = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        return free_E, self.positions
    
    # ENDREVISE

    def sample(self):
        self.samples.append(self.energy_function())
        self.achieved += 1
        return 0
        
    
class EnergyMCSampler(EnergySampler):

    def __init__(self, Complex, space, size):
        super(EnergyMCSampler, self).__init__(Complex, space, size)
        self.best_energy = 1e100
        self.best_positions = self.positions
    
    def sample(self):
        self.transform_complex(self.space.generator())
        energy_position = self.energy_function()
        if energy_position[0] < self.best_energy:
            self.best_energy = energy_position[0]
            self.best_positions = energy_position[1][:]
        self.achieved += 1
        return 0
    
class EnergyGridSampler(EnergySampler):
    
    def __init__(self, Complex, space, size, grid_spacing=1, grid_bounds=[-1e2,1e2]):
        super(EnergyGridSampler, self).__init__(Complex, space, size)
        self.best_energy = 1e100
        self.best_positions = self.positions
        self.grid_spacing = grid_spacing
        self.grid_bounds = grid_bounds
        self.grid = cartesian((np.linspace(self.grid_bounds[0],self.grid_bounds[1],grid_spacing),
                              np.linspace(self.grid_bounds[0],self.grid_bounds[1],grid_spacing),
                              np.linspace(self.grid_bounds[0],self.grid_bounds[1],grid_spacing)))
        self.mask = np.apply_along_axis(self.space.is_in, self.grid, axis=0)
        grid = []
        for position, truth in zip(self.grid, self.mask):
            if truth:
                grid.append(position)
        self.grid = np.array(grid)
        
    def sample(self):
        transformation_array = self.grid[0]
        np.delete(grid[0])
        self.transform_complex(self.space.generator())
        energy_position = self.energy_function()
        if energy_position[0] < self.best_energy:
            self.best_energy = energy_position[0]
            self.best_positions = energy_position[1][:]
        self.achieved += 1
        return 0

## @class Structure
# Defines the chemical properties for molecules based on residues. 
class Structure(object):
    
    def __init__(self, residue_names, residue_length=None,
                    rotating_elements=None, connect=None, 
                    residue_path=None, alias=None):
        self.residue_names = residue_names
        self.residue_path = residue_path
        self.init_string = """
        """
        if self.residue_path != None:
            for name in self.residue_names:
                self.init_string += """
                %s loadoff %s.lib 
                """%(name, name)
        else:
            pass    
        
        self.residue_length = defaultdict(lambda : 0)
        
        if residue_length:
            for index, residue in enumerate(self.residue_names):
                self.residue_length[residue] = residue_length[index]
        
        self.connect = defaultdict(lambda : [[0,-1],[-2,0],1.,1.])
        self.residue_bond_lengths = defaultdict(lambda : [1.0*unit.angstrom, 1.0*unit.angstrom])
        
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
                if self.rotating_elements[residue] == [None]:
                    self.rotating_elements[residue] = [[start, bond, end]]
                elif self.rotating_elements[residue] == None:
                    raise ValueError('Residue does not exist! CANNOT assign rotability!')
                else:
                    self.rotating_elements[residue].append([start, bond, end])

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
    
    def delete_rotation(self, residue_name, rotations):
        if isinstance(rotations[0],basestring):
            for rotation in rotations:
                if rotation in self.rotating_elements[residue_name]:
                    self.rotating_elements[residue_name].remove(rotation)
                else:
                    pass
        elif isinstance(rotations, basestring):
            if rotation in self.rotating_elements[residue_name]:
                self.rotating_elements[residue_name].remove(rotation)
            else:
                pass
        else:
            raise ValueError("The input supplied is not a two component list specifying a rotation!")
        return self.rotating_elements
    
    def set_length(self, residue_name, length):
        self.residue_length[residue_name] = length
        
    def set_bonds(self, residue_name, bond_lengths):
        self.residue_bond_lengths = bond_lengths

    def translate(self, sequence):
        sequence_array = sequence.split(' ')
        if len(sequence_array) == 1:
            return self.alias[sequence_array[0]][0]
        else:
            return " ".join([self.alias[sequence_array[0]][1]]
                + [aliasElement[2] for aliasElement in map(self.alias.__getitem__, sequence_array)]
                + [self.alias[sequence_array[-1]][3]])

## Represents a complex containing multiple molecule chains.
class Complex(object):
    
    ## Specifies a complex by giving a force field governing it, defaulting to "leaprc.ff12SB"
    # @param force_field Specifies the force field governing the complex. 
    def __init__(self, force_field="oldff/leaprc.ff99"):
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
    
    ## Represents a molecule chain, comprising multiple residues
    class Chain(object):
        def __init__(self, Complex, Structure, sequence=None, start=0):
            self.start = start
            self.start_history = start
            self.complex = Complex
            self.residues_start = []
            self.length = 0
            self.length_history = self.length
            self.element = [self.start, self.start+1, self.start+self.length-1]
            self.structure = Structure
            self.alias_sequence = sequence
            self.sequence = self.structure.translate(self.alias_sequence)
            self.sequence_array = []
            self.alias_sequence_array = []
            self.append_history = []
            self.prepend_history = []
            if self.sequence:
                self.sequence_array = self.sequence.split(' ')
                self.alias_sequence_array = self.alias_sequence.split(' ')
                self.length = sum(map(self.structure.residue_length.__getitem__, self.sequence_array))
                tally = 0
                for residue in self.sequence_array:
                    self.residues_start.append(tally)
                    tally += self.structure.residue_length[residue]
            else:
                pass
            
        def update_chains(self):
            length = self.length
            self.length = sum(map(self.structure.residue_length.__getitem__, self.sequence_array))
            self.residues_start = []
            tally = 0
            for residue in self.sequence_array:
                self.residues_start.append(tally)
                tally += self.structure.residue_length[residue]
            self.element = [self.start, self.start + 1, self.start+self.length-1]
            for chain in self.complex.chains:
                if chain.start > self.start:
                    chain.start += self.length - length
                    chain.start_history += self.length - length
                    chain.element = [chain.start, chain.start + 1, chain.start+chain.length-1]
                else:
                    pass
            
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
            
        def rotate_element(self, element, angle):
            if len(element) == 3 and element[2]:
                revised_element = [index + self.start for index in element]
                self.complex.rotate_element(revised_element, angle)
            elif (len(element) == 3) and (element[2] == None):
                revised_element = [element[0] + self.start, element[1] + self.start, self.length + self.start]
                self.complex.rotate_element(revised_element, angle)
            else:
                raise ValueError('Rotable element contains too many or too few components!')
                
        def rotate_in_residue(self, residue_index, residue_element_index, angle):
            element = self.structure.rotating_elements[self.sequence_array[residue_index]][residue_element_index]
            #LOOK HERE!!!!!!!!!!!!!
            if element[1] < 0:
                element[1] += self.structure.residue_length[self.sequence_array[residue_index]]
            if element[2] == None:
                revised_element = [element[0]+self.start+self.residues_start[residue_index], element[1]+self.start+self.residues_start[residue_index], None]
            else:
                revised_element = [element[0]+self.start+self.residues_start[residue_index],
                                   element[1]+self.start+self.residues_start[residue_index],
                                   element[2]+self.start+self.residues_start[residue_index]]
            self.rotate_element(revised_element, angle)
                
        def rotate_historic_element(self, historic_element, angle):
            if historic_element[2]:
                self.rotate_element([historic_element[0]+self.start_history-self.start,
                                     historic_element[1]+self.start_history-self.start,
                                     historic_element[2]+self.start_history-self.start],angle)
            else:
                self.rotate_element([historic_element[0]+self.start_history-self.start,
                                     historic_element[0]+self.start_history-self.start,
                                     None], angle)
                
        def rotate_in_historic_residue(self, historic_index, element_index, angle):
            offset = len(self.prepend_history)
            self.rotate_in_residue(historic_index+offset, element_index, angle)
        
        def rotate_global(self, axis, angle):
            self.complex.rotate_global(self.element, axis, angle)
            
        def translate_global(self, shift):
            self.complex.translate_global(self.element, shift)
            
    def add_chain(self, sequence, Structure):
        if self.chains:
            start = sum([chain.length for chain in self.chains])
        else:
            self.chains = []
            start = 0
        self.chains.append(self.Chain(self, Structure, sequence=sequence, start=start))
            
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
                else:
                    raise ValueError("Empty Chain Index: %s"%index)
            chain_string = " ".join(["CHAIN%s"%index for index in range(len(self.chains))])
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
            self.system = self.prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, implicitSolvent=app.OBC1)
            self.integrator = mm.LangevinIntegrator(300.*unit.kelvin, 1./unit.picosecond, 0.002*unit.picoseconds)
            self.simulation = app.Simulation(self.topology, self.system, self.integrator)
        else:
            raise ValueError('Empty Complex! CANNOT build!')
            
    def rebuild(self, target_path="", file_name="out", exclusion=[]):
        old_positions = self.positions[:]
        self.build()
        for chain in self.chains:
            if not (chain in exclusion):
                pre_positions = self.positions[chain.start:chain.start_history]
                chain_positions = old_positions[chain.start:chain.length_history]
                post_positions = self.positions[chain.start_history + chain.length_history:chain.length]
                
                if pre_positions:
                    # Fixing positions of prepended atoms from here on:
                    
                    pre_positions = self.positions[chain.start:chain.start_history + 1]
                    pre_vector = self.positions[chain.start_history + chain.structure.connect[chain.prepend_history[-1]][1][0]] - self.positions[chain.start_history + 1]
                    old_pre_vector = old_positions[chain.start] - old_positions[chain.start + 1]
                    angle = -ang(nostrom(pre_vector), nostrom(old_pre_vector))
                    axis = np.cross(np.asarray(nostrom(pre_vector)), np.asarray(nostrom(old_pre_vector)))
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
                    pre_bond_shift = (chain.structure.connect[chain.prepend_history[-1]][2]-1.)*old_pre_vector

                    for j in range(0,len(pos)):
                        roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)),rot)
                        pos[j] = mm.Vec3(roted[0],roted[1],roted[2])*unit.angstrom
                        pos[j] += shift_back + pre_bond_shift

                    pre_positions = pos[:]
                    chain_positions[0] += pre_bond_shift

                    self.positions = pre_positions[:] + chain_positions[1:] + post_positions[1:]

                    # Stop fixing positions of prepended atoms.
               
                if post_positions:
                    # Fixing positions of appended atoms from here on:

                    post_positions = self.positions[chain.start_history + chain.length_history - 1:chain.length]
                    post_vector = self.positions[chain.start_history + chain.length_history - 1] - self.positions[chain.start_history + chain.length_history - 2]
                    old_post_vector = old_positions[chain.start_history + chain.length_history - 1] - old_positions[chain.start_history + chain.length_history - 2]
                    angle = -ang(nostrom(post_vector), nostrom(old_post_vector))
                    axis = np.cross(np.asarray(nostrom(post_vector)), np.asarray(nostrom(old_post_vector)))
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

                    post_bond_shift = (chain.structure.connect[chain.append_history[0]][2]-1.)*old_post_vector
                    shift_back = chain_positions[chain.structure.connect[chain.sequence_array[-len(chain.append_history)]][0][1]]

                    for pos_idx, pos_elem in enumerate(pos):
                        roted = np.dot(np.array(pos_elem.value_in_unit(unit.angstrom)),rot)
                        pos[pos_idx] = mm.Vec3(roted[0],roted[1],roted[2])*unit.angstrom
                        pos[pos_idx] += shift_back + post_bond_shift

                    post_positions = pos[:]
                    chain_positions[-1] += post_bond_shift

                    self.positions = chain_positions[:-1] + post_positions[:]

                    # Stop fixing positions of propended atoms.

                else:
                    pass
    

            else:
                pass
            
    def rotate_element(self, element, angle):
        if self.positions:
            pos = self.positions[:]
            vec_a = (pos[element[1]]-pos[element[0]])
            self.rotate_global(element, vec_a, angle)
        else:
            raise ValueError('This Complex contains no positions! You CANNOT rotate!')
            
    def rotate_global(self, element, axis, angle):
        if self.positions:
            x, y, z = np.asarray(nostrom(axis))/(np.linalg.norm(np.asarray(nostrom(axis))))
            phi_2 = angle/2.
            pos = self.positions[:]
            shift_forward = mm.Vec3(0,0,0)*unit.angstroms-pos[element[1]]
            s = np.math.sin(phi_2)
            c = np.math.cos(phi_2)
            rot = np.array([[2*(np.power(x,2)-1)*np.power(s,2)+1, 2*x*y*np.power(s,2)-2*z*c*s, 2*x*z*np.power(s,2)+2*y*c*s],
                            [2*x*y*np.power(s,2)+2*z*c*s, 2*(np.power(y,2)-1)*np.power(s,2)+1, 2*z*y*np.power(s,2)-2*x*c*s],
                            [2*x*z*np.power(s,2)-2*y*c*s, 2*z*y*np.power(s,2)+2*x*c*s, 2*(np.power(z,2)-1)*np.power(s,2)+1]])
            
            for j in range(element[1],element[2]):
                pos[j] += shift_forward
            
            for j in range(element[1],element[2]):
                roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)),rot)
                pos[j] = mm.Vec3(roted[0],roted[1],roted[2])*unit.angstrom
                pos[j] -= shift_forward
                
            positions_new = pos
            self.positions = positions_new[:]
        else:
            raise ValueError('This Complex contains no positions! You CANNOT rotate!')
        
    def translate_global(self, element, shift):
        if self.positions:
            vec_shift = mm.Vec3(*shift)*unit.angstroms
            pos = self.positions[:]
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
        current_sequence_array = chain.sequence_array
        #Build the new array of residues
        new_alias_sequence_array = sequence.split(" ")
        new_sequence_array = chain.structure.translate(sequence).split(' ')
        #Get the current positions
        current_positions = self.positions[:]
        #Build new sequence
        chain.create_sequence(sequence)
        #Rebuild the complex, excluding the chain to be fit
        self.rebuild(exclusion=[chain])
        #Translate chain to the position of the first atom to be fit
        self.translate_global(chain.element, current_positions[chain.start]-self.positions[chain.start])
        vec0 = self.positions[chain.element[1]] - self.positions[chain.element[0]]
        vec1 = self.current_positions[chain.element[1]] - self.current_positions[chain.element[0]]
        angle = ang(vec0,vec1)
        global_axis = np.cross(np.asarray(vec0), np.asarray(vec1))
        #Rotate chain to fit first rotable bond
        self.rotate_global(global_axis, angle)
        #Iterate over residues and their rotable elements
        for residue_id, residue in enumerate(chain.sequence_array):
            for element_id, element in enumerate(chain.structure.rotating_elements[residue]):
                revised0 = element[0]+chain.start+chain.residues_start[residue_id]
                revised1 = element[1]+chain.start+chain.residues_start[residue_id]
                axis = current_positions[revised1] - current_positions[revised0]
                axis /= np.linalg.norm(np.asarray(axis))
                #Bond to next-to-nearest neighbour along increasing indices in the element
                vec0 = self.positions[revised1+1] - self.positions[revised1]
                vec1 = self.current_positions[revised1+1] - self.current_positions[revised1]
                #Projection onto plane perpemdicular to axis
                proj_vec0 -= np.dot(axis,vec0)*axis
                proj_vec1 -= np.dot(axis,vec1)*axis
                #Get angle between projections
                angle = ang(proj_vec0, proj_vec1)
                #Rotate element by that angle, to align to backbone
                chain.rotate_in_residue(residue_id, element_id, angle)

    #get current complex energy
    def get_energy(self):
        self.simulation.context.setPositions(self.positions)
        state = self.simulation.context.getState(getPositions=True,getEnergy=True,groups=1)
        free_E = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        return free_E, self.positions
    
    #Wrapper for OpenMM local energy minimization
    def minimize(self, max_iterations=500):
        self.simulation.context.setPositions(self.positions)
        self.simulation.minimizeEnergy(maxIterations=max_iterations)
        self.positions = self.simulation.context.getState(getPositions=True,getEnergy=True,groups=1).getPositions()
        free_E = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        return free_E
    
    #Introducing 'chain-wriggling' into the framework
    def pert_min(self, size=1e-4, iterations=10):
        for repeat in range(iterations):
            for position in self.positions:
                position += np.random.uniform(-size,size,3)
            self.minimize()

#Turn computation / sampler into functions? They are not very class-like, and functions would be easier to MPIify
class Computation(object):
    
    def __init__(Sampler, Complex, beta):
        self.sampler = Sampler
        self.complex = Complex
        self.positions = None
        self.energies = None
        self.partitions = None
        self.probabilities = None
        self.entropies = None
        self.sequences = None
        self.beta = beta
    
    #naively compute the partition function Z, probabilities P and entropy S of a given array of energies
    def naiveZPS(self):
        P = np.exp(-beta*np.asarray(self.energies))
        Z = P.sum()
        P /= Z
        S = -(np.log(P/len(P))*P).sum()
        return Z, P, S
    
    #compute Z P and S by monte carlo integration
    def mcZPS(self, volume):
        P = np.exp(-beta*np.asarray(self.energies))
        Z = P.sum()/len(P)*volume
        P /= Z
        S = -(np.log(P/len(P))*P).sum()/len(P)*volume
        return Z, P, S
    
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
