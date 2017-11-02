from ..Complex import Complex
from ..Structure import Structure
import numpy as np

#Build Structure-object for DNA residues
DNA = XMLStructure("DNA.xml")

#Build maximally hindered dinucleotide complex:
dinucleotide = Complex("leaprc.ff12SB")
dinucleotide.add_chain("G G", DNA)
dinucleotide.build()
dinChain = dinucleotide.chains[0]

#Define dict of good angles:
good_angles = {}
energy_map = []

#Sample all combinations of angles:
for angle1 in range(0, 360, 1):
	for angle2 in range(0, 360, 1):
		for angle3 in range(0, 360, 1):
			for angle4 in range(0, 360, 1):
				dinChain.rotate_in_residue(1, 0, angle1/360.*3.14)
				dinChain.rotate_in_residue(1, 1, angle2/360.*3.14)
				dinChain.rotate_in_residue(1, 2, angle3/360.*3.14)
				dinChain.rotate_in_residue(1, 3, angle4/360.*3.14)
				energy = dinucleotide.get_energy()[0]
				energy_map.append([angle1,
								   angle2,
								   angle3,
								   angle4,
								   energy])
				if energy < 0:
					good_angles[energy] = [angle1,
										   angle2,
										   angle3,
										   angle4]


#Write results to file
good_angle_file = open("good_angles.out", "w")
energy_map_file = open("energy_map.out", "w")

for key in good_angles:
	good_angle_file.write(key + " : " + good_angles[key])

for element in emergy_map:
	energy_map_file.write("%s%s%s%s: %s" % element)

print("Ratio = " + len(good_angles)/(360.)**(4))
