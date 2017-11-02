from Complex import Complex
from Structure import Structure
from LoadFrom import XMLStructure
import numpy as np
from simtk.openmm import app

PROT = XMLStructure("PROT.xml")

#Build maximally hindered dinucleotide complex:
dipeptide = Complex("leaprc.ff12SB")
dipeptide.add_chain("NALA CALA", PROT)
dipeptide.build()
pep = dipeptide.chains[0]
#dipeptide.minimize()
print("Starting Energy = %s" % dipeptide.get_energy()[0])

file = open("dipeptide.pdb", "w")
#Sample:
for angle1 in range(0, 360, 90):
	for angle2 in range(0, 360, 90):
		for angle3 in range(0, 360, 90):
			pep.rotate_in_residue(0, 0, 90/360.*3.14)
			app.PDBFile.writeModel(dipeptide.topology, dipeptide.positions, file=file, modelIndex=1)
			pep.rotate_in_residue(0, 1, 90/360.*3.14)
			app.PDBFile.writeModel(dipeptide.topology, dipeptide.positions, file=file, modelIndex=1)
			pep.rotate_in_residue(0, 2, 90/360.*3.14)
			app.PDBFile.writeModel(dipeptide.topology, dipeptide.positions, file=file, modelIndex=1)
				

pep.append_sequence("ALA")
dipeptide.rebuild()
dipeptide.minimize()
app.PDBFile.writeModel(dipeptide.topology, dipeptide.positions, file=file, modelIndex=1)
pep.prepend_sequence("ALA")
dipeptide.rebuild()
dipeptide.minimize()
app.PDBFile.writeModel(dipeptide.topology, dipeptide.positions, file=file, modelIndex=1)

file.close()

print("Done.")
