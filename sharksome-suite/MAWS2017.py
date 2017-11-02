#MAWS is part of the sharksome software suite
#This software is published under MIT license
#COPYRIGHT 2017 Michael Jendrusch
#Authors: Michael Jendrusch, Stefan Holderbach

#VERSION Information
#Note: This information should be updated with every technical change to ensure that
#      every calculation can be linked to its software version.
VERSION = "2.0"
RELEASE_DATE = "2017"
METHOD = "Kullback-Leibler"

import copy
import numpy as np
import argparse
from datetime import datetime
from LoadFrom import XMLStructure
from helpers import nostrom
from Complex import Complex
from Structure import Structure
from Routines import ZPS, S
from Kernels import centerOfMass
from collections import defaultdict
from operator import itemgetter
from simtk.openmm import unit
from simtk.openmm import app
import Space

#Parser
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Job name.")
parser.add_argument("-b", "--beta", type=float,
					help="Inverse temperature.")
parser.add_argument("-c1", "--firstchunksize", type=int,
					help="Number of samples in the first MAWS step.")
parser.add_argument("-c2", "--secondchunksize", type=int,
					help="Number of samples in all subsequent MAWS steps.")
parser.add_argument("-t", "--ntides", type=int,
					help="Number of nucleotides in the aptamer.")
parser.add_argument("-p", "--path",
					help="Path to your PDB file.")
args = parser.parse_args()


#PARAMS
#Number of samples taken per step
JOB_NAME = "GFP"
if args.name:
	JOB_NAME = args.name
BETA = 0.01
if args.beta:
	BETA = args.beta
FIRST_CHUNK_SIZE = 5000
if args.firstchunksize:
	FIRST_CHUNK_SIZE = args.firstchunksize
CHUNK_SIZE = 5000
if args.secondchunksize:
	CHUNK_SIZE = args.secondchunksize
N_NTIDES = 15
if args.ntides:
	N_NTIDES = args.ntides
PDB_PATH = "/net/data.isilon/ag-reils/igem2015/testenv/silane.pdb"
if args.path:
	PDB_PATH = args.path
#N_CHUNKS = 1
#FUZZY = 1
##Number of rotable junctions in DNA, to distinguish forward and backward rotation
N_ELEMENTS = 4


#Open a pdb file, to monitor progress
output = open("{0}_output.log".format(JOB_NAME),"w")
entropyLog = open("{0}_entropy.log".format(JOB_NAME), "w")
step = open("{0}_step_cache.pdb".format(JOB_NAME), "w")

#Starting logfile
output.write("MAWS - Making Aptamers With Software\n")
output.write("Active version: {0} (released:_{1})\n".format(VERSION, RELEASE_DATE))
output.write("Computational method: {0}\n".format(METHOD))
output.write("Job: {0}\n".format(JOB_NAME))
output.write("Input file: {0}\n".format(PDB_PATH))
output.write("Sample number in initial step: {0}\n".format(FIRST_CHUNK_SIZE))
output.write("Sample number per further steps: {0}\n".format(CHUNK_SIZE))
output.write("Number of further steps: {0} (sequence length = )\n".format(N_NTIDES, N_NTIDES + 1))
output.write("Value of beta: {0}\n".format(BETA))
output.write("Start time: {0}\n".format(str(datetime.now())))

#Build Structure-object for DNA residues
RNA = XMLStructure("RNA.xml")

#Instantiate the Complex for further computation
cpx = Complex("leaprc.ff12SB")

#Add an empty Chain to the Complex, of structure DNA
cpx.add_chain('', RNA)

#Add a chain to the complex using a pdb file (e.g. "xylanase.pdb")
cpx.add_chain_from_PDB(PDB_PATH,parameterized=False)

#Build a complex with the pdb only, to get center of mass of the pdb --#
c = Complex("leaprc.ff12SB")

c.add_chain_from_PDB(PDB_PATH,parameterized=False)

c.build()
#----------------------------------------------------------------------#

#Create a sampling Cube of Diameter 50. Angstroms around the pdb center of mass
cube = Space.Cube(20., centerOfMass(np.asarray(nostrom(c.positions))))

#Create a sampling Space of the direct sum of N_ELEMENTS angles
rotations = Space.NAngles(N_ELEMENTS)

#Initialize variables for picking out the best aptamer
best_entropy = None
best_sequence = None
best_positions = None

output.write("Initialized succesfully!\n")

#for each nucleotide in GATC
for ntide in 'GAUC':
	output.write("{0}: starting initial step for '{1}'\n".format(str(datetime.now()),ntide))
	energies = []
	free_E = None
	position = None
	#Get a full copy of our Complex
	complex = copy.deepcopy(cpx)
	#Pick a chain to be our aptamer
	aptamer = complex.chains[0]
	#Initialize the chain with a nucleotide
	aptamer.create_sequence(ntide)
	#Build the Complex
        print("INTO LEAP ---------------------------------------------------------------------")
	complex.build()
        print("OUT OF LEAP -------------------------------------------------------------------")
	#Remember its initial positions
	positions0 = complex.positions[:]
	#For the number of samples
	for i in range(FIRST_CHUNK_SIZE):
		#Get a new sample from the cube
		orientation = cube.generator()
		#Get a new sample from the angles
		rotation = rotations.generator()
		#Translate the aptamer by the displacement part of the sample from the cube generator
		aptamer.translate_global(orientation[0:3]*unit.angstrom)
		#Rotate the aptamer by the rotation part of the sample from the cube generator
		aptamer.rotate_global(orientation[3:-1]*unit.angstrom, orientation[-1])
		#For all thing rotating
		for j in range(N_ELEMENTS):
			#Rotate around the bond by generated angle
			aptamer.rotate_in_residue(0, j, rotation[j])
		#Get energy of the complex
		energy = complex.get_energy()[0]
		#Compare to lowest energy, if lowest...
		if free_E == None or energy < free_E:
			#Tell
			print(energy)
			#Set free energy to energy
			free_E = energy
			#Remember positions
			position = complex.positions[:]
		#Remember energy
		energies.append(energy)
		#Reset positions
		complex.positions = positions0[:]
	#Calculate entropy
	entropy = S(energies, beta=BETA)

	#Performing outputs
	pdblog = open("{0}_1_{1}.pdb".format(JOB_NAME,ntide),"w")
	app.PDBFile.writeModel(copy.deepcopy(complex.topology), position[:], file=pdblog, modelIndex=1)
	pdblog.close()

	entropyLog.write("SEQUENCE: {0} ENTROPY: {1} ENERGY: {2}\n".format(aptamer.alias_sequence, entropy, free_E))
	#Check if best ...
	if best_entropy == None or entropy < best_entropy:
		best_entropy = entropy
		best_sequence = ntide
		best_positions = position[:]
		best_topology = copy.deepcopy(complex.topology)

app.PDBFile.writeModel(best_topology, best_positions, file=step, modelIndex=1)
#Output best as well
pdblog = open("{0}_best_1_{1}.pdb".format(JOB_NAME,ntide),"w")
app.PDBFile.writeModel(best_topology, best_positions, file=pdblog, modelIndex=1)
pdblog.close()

output.write("{0}: Completed first step. Selected nucleotide: {1}\n".format(str(datetime.now()), best_sequence))
output.write("{0}: Starting further steps to append {1} nucleotides\n".format(str(datetime.now()), N_NTIDES))

#For how many nucleotides we want (5)
for i in range(N_NTIDES):
	#Same as above, more or less
	best_old_sequence = best_sequence
	best_old_positions = best_positions[:]
	best_entropy = None
	for ntide in 'GAUC':
		#For append nucleotide or prepend nucleotide
		for append in [True, False]:
			energies = []
			free_E = None
			position = None
			#Get our complex
			complex = copy.deepcopy(cpx)
			#Get our aptamer
			aptamer = complex.chains[0]
			aptamer.create_sequence(best_old_sequence)
                        print("INTO LEAP ------------------------------------------------------------------------------")
			complex.build()
                        print("OUT OF LEAP ----------------------------------------------------------------------------")
			#Readjust positions
			complex.positions = best_old_positions[:]
			if append:
				#Append new nucleotide
				aptamer.append_sequence(ntide)
			else:
				#Prepend new nucleotide
				aptamer.prepend_sequence(ntide)
                        print("INTO LEAP ------------------------------------------------------------------------------")
			complex.rebuild()
                        print("OUT OF LEAP ----------------------------------------------------------------------------")
			## Optionally minimize or "shake" complex, to find lower energy local minimum
			#not recommended! causes issues with proteins
			#complex.minimize()
			complex.pert_min(size=0.5)
			#Remember positions
			positions0 = complex.positions[:]

			#For number of samples
			for k in range(CHUNK_SIZE):
				#Get random angles
				rotation = rotations.generator()
				#For everything forward
				for j in range(N_ELEMENTS-1):
					#Rotate the new nucleotide's bonds
					if append:
						aptamer.rotate_in_residue(-1, j, rotation[j])
					else:
						aptamer.rotate_in_residue(0, j, rotation[j], reverse=True)
				#For everything backward (C3'-O3')
				#Rotate the old nucleotides' bond
				if append:
					aptamer.rotate_in_residue(-2, 3, rotation[3])
				else:
					aptamer.rotate_in_residue(0, 3, rotation[3], reverse=True)
				#Get energy
				energy = complex.get_energy()[0]
				#Check if best
				if free_E == None or energy < free_E:
					print(energy)
					free_E = energy
					position = complex.positions[:]
				#Remember energies
				energies.append(energy)
				#Reset positions
				complex.positions = positions0[:]

			entropy = S(energies, beta=BETA)

			#outputs
			pdblog = open("{0}_{1}_{2}.pdb".format(JOB_NAME, i+2, ntide), "w")
			app.PDBFile.writeModel(copy.deepcopy(complex.topology), position[:], file=pdblog, modelIndex=1)
			pdblog.close()

			entropyLog.write("SEQUENCE: {0} ENTROPY: {1} ENERGY: {2}\n".format(aptamer.alias_sequence, entropy, free_E))
			#Choose best
			if best_entropy == None or entropy < best_entropy:
				best_entropy = entropy
				best_positions = position[:]
				best_sequence = aptamer.alias_sequence
				best_topology = copy.deepcopy(complex.topology)
	app.PDBFile.writeModel(best_topology, best_positions, file=step, modelIndex=1)
	#Output best as well
	output.write("{0}: Completed step {1}. Selected sequence: {2}\n".format(str(datetime.now()), i+2, best_sequence))
	pdblog = open("{0}_best_{1}_{2}.pdb".format(JOB_NAME, i+2, ntide),"w")
	app.PDBFile.writeModel(best_topology, best_positions, file=pdblog, modelIndex=1)
	pdblog.close()


#Render resulting aptamer to pdb
result_complex = copy.deepcopy(cpx)
aptamer = result_complex.chains[0]
aptamer.create_sequence(best_sequence)
result_complex.build()
result_complex.positions = best_positions[:]
pdb_result = open("{0}_RESULT.pdb".format(JOB_NAME),"w")
app.PDBFile.writeModel(result_complex.topology, result_complex.positions, file=pdb_result)
pdb_result.close()

output.write("{0}: Run completed. Thank you for using MAWS!\n\n".format(str(datetime.now())))
output.write("Final sequence: {0}\n".format(best_sequence))


#Garbage collection
step.close()
entropyLog.close()
output.close()
