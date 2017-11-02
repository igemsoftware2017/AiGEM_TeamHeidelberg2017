#Insert Parsers preferably here, plus load from PDB, load from mol2, etc.

from simtk.openmm import app
import Structure
import simtk.openmm as mm
from simtk import unit
import xml.etree.ElementTree as et

def XMLParseStructure(path):
	tree = et.parse(path)
	root = tree.getroot()
	aliases = []
	residue_names = []
	residue_length = []
	rotators = []
	backbone = []
	connect = []
	residue_path = root.find("residuePath").text if root.find("residuePath") else None

	#ALIAS
	for alias in root.findall("alias"):
		for element in alias.findall("element"):
			if len(element) == 4:
				elem = ['']*5
				elem[0] = element.get('name')
				elem[1] = element.find('alone').text
				elem[2] = element.find('start').text
				elem[3] = element.find('middle').text
				elem[4] = element.find('end').text
				aliases.append(elem)
			else:
				pass

	for residue in root.findall("residue"):
		residue_rotations = []
		#NAMES
		residue_names.append(residue.get("name"))
		#LENGTHS
		residue_length.append(int(residue.get("length")))
		#ROTATIONS
		for rotation in residue.findall('rotation'):
			residue_rotations.append([residue.get("name"),
									  int(rotation.find("start").text), 
									  int(rotation.find("bond").text), 
									  None if rotation.find("end").text == 'end' else int(rotation.find("end").text)])
		rotators += residue_rotations
		#BACKBONE
		backbone.append([residue.get("name"),
					 int(residue.find("backbone").find("start").text),
					 int(residue.find("backbone").find("middle_pre").text),
					 int(residue.find("backbone").find("bond").text),
					 int(residue.find("backbone").find("middle_post").text),
					 int(residue.find("backbone").find("end").text)])
		#CONNECT
		append = [int(residue.find('append').find('newFirstAtom').text),
				  int(residue.find('append').find('oldLastAtom').text),
				  float(residue.find('append').find('bondLength').text)]
		prepend = [int(residue.find('prepend').find('newLastAtom').text),
				   int(residue.find('prepend').find('oldFirstAtom').text),
				   float(residue.find('prepend').find('bondLength').text)]
		residue_connect = [append[:2], prepend[:2], append[2], prepend[2]]
		connect.append(residue_connect)

	return residue_names, residue_length, rotators, backbone, connect, residue_path, aliases

def XMLStructure(path):
	parsed = XMLParseStructure(path)
	return Structure.Structure(*parsed[0:2], rotating_elements=parsed[2], backbone_elements=parsed[3], connect=parsed[4], residue_path=parsed[5], alias=parsed[6])

def PDBParseStructure(path):
	pdb = app.PDBFile(path)
	topology = pdb.topology
	residue_names = []
	residue_length = []
	for residue in topology.residues():
		if not (residue.name in residue_names):
			residue_names.append(residue.name)
			residue_length.append(sum(1 for elem in residue.atoms()))
	return residue_names, residue_length

def PDBStructure(path):
	return Structure.Structure(*PDBParseStructure(path))
