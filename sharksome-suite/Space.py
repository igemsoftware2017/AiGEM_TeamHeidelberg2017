import numpy as np
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit


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
        rotation_angle = np.random.uniform(0, np.math.pi)
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
        rotation_angle = np.random.uniform(0, 2*np.math.pi)
        result = np.array([r*np.math.cos(phi)*np.math.sin(psi),
                           r*np.math.sin(phi)*np.math.sin(psi),
                           r*np.math.cos(psi), x, y, z,
                           rotation_angle])
        return result

## @class SphericalShell
# Represents a 3D spherical shell sampling space        
class SphericalShell(Space):
    ## Creates a new Space 
    # @param outerRadius
    # Specifies the radius of the outer sphere
    # @param innerRadius
    # Specifies the radius of the inner sphere
    # @param centre
    # Specifies the position of the box via its centre
    # @param units
    # Specifies the used length unit. Typically angstroms. 
    def __init__(self, innerRadius, outerRadius, centre, units=unit.angstroms):
        def boolean(position):
            position.abs() <= outerRadius and innerRadius <= position.abs()
        super(SphericalShell, self).__init__(boolean)
        self.innerRadius = innerRadius
        self.outerRadius = outerRadius
        self.centre = centre
        self.volume = (4./3.*np.math.pi*outerRadius**3*(2*np.math.pi)**3
                       - 4./3.*np.math.pi*innerRadius**3*(2*np.math.pi)**3)

    ##Returns an element of the space (Sphere)
    # @returns 3D coordinates of the element 
    def generator(self):
        axis = np.random.uniform(-1,1,3)
        x,y,z = axis/np.linalg.norm(axis)
        r = np.random.uniform(self.innerRadius, self.outerRadius)
        phi = np.random.uniform(0, 2*np.math.pi)
        psi = np.random.uniform(0, np.math.pi)
        rotation_angle = np.random.uniform(0, 2*np.math.pi)
        result = np.array([r*np.math.cos(phi)*np.math.sin(psi),
                           r*np.math.sin(phi)*np.math.sin(psi),
                           r*np.math.cos(psi), x, y, z,
                           rotation_angle])
        return result


## @class NAngles
# Represents U(1)^N
class NAngles(Space):
    ## __init__
    # @param number
    # the number N of angles to randomly generate
    def __init__(self, number):
        def boolean(position):
            return all(0 <= element and element <= 2*np.math.pi for element in position)
        super(NAngles, self).__init__(boolean)
        self.number = number

    ## returns self.number random angles in radians
    def generator(self):
        return np.array([np.random.uniform(0, 2*np.math.pi) for i in range(self.number)])
