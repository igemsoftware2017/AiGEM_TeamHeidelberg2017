from Complex import *
from Structure import *

DNAresidues = ["DGN", "DAN", "DTN", "DCN",
               "DG", "DA", "DT", "DC", 
               "DG5", "DA5", "DT5", "DC5",
               "DG3", "DA3", "DT3", "DC3"]

DNAphosphate = [["DGN", 0, 1, None],
                ["DAN", 0, 1, None], 
                ["DTN", 0, 1, None], 
                ["DCN", 0, 1, None],
                ["DG5", 0, 1, None],
                ["DA5", 0, 1, None], 
                ["DT5", 0, 1, None], 
                ["DC5", 0, 1, None],
                ["DG", 0, 3, None],
                ["DA", 0, 3, None], 
                ["DT", 0, 3, None], 
                ["DC", 0, 3, None],
                ["DG3", 0, 3, None],
                ["DA3", 0, 3, None], 
                ["DT3", 0, 3, None], 
                ["DC3", 0, 3, None]]

DNAoc5 = [["DGN", 2, 3, None],
          ["DAN", 2, 3, None], 
          ["DTN", 2, 3, None], 
          ["DCN", 2, 3, None],
          ["DG5", 2, 3, None],
          ["DA5", 2, 3, None], 
          ["DT5", 2, 3, None], 
          ["DC5", 2, 3, None],
          ["DG", 3, 4, None],
          ["DA", 3, 4, None], 
          ["DT", 3, 4, None], 
          ["DC", 3, 4, None],
          ["DG3", 3, 4, None],
          ["DA3", 3, 4, None], 
          ["DT3", 3, 4, None], 
          ["DC3", 3, 4, None]]

DNAbase = [["DGN", 8, 10, -4],
           ["DAN", 8, 10, -4], 
           ["DTN", 8, 10, -4], 
           ["DCN", 8, 10, -4],
           ["DG5", 8, 10, -4],
           ["DA5", 8, 10, -4], 
           ["DT5", 8, 10, -4], 
           ["DC5", 8, 10, -4],
           ["DG", 10, 12, -4],
           ["DA", 10, 12, -4], 
           ["DT", 10, 12, -4], 
           ["DC", 10, 12, -4],
           ["DG3", 10, 12, -4],
           ["DA3", 10, 12, -4], 
           ["DT3", 10, 12, -4], 
           ["DC3", 10, 12, -4]]

DNAlengths = [32, 31, 31, 29, 
              33, 32, 32, 30,
              31, 30, 30, 28,
              34, 33, 33, 31]

DNAalias = [["DGN","DGN","DG5","DG","DG3"],["DAN","DAN","DA5","DA","DA3"],["DTN","DTN","DT5","DT","DT3"],["DCN","DCN","DC5","DC","DC3"],
            ["DG3","DG3","DG","DG","DG3"],["DA3","DA3","DA","DA","DA3"],["DT3","DT3","DT","DT","DT3"],["DC3","DC3","DC","DC","DC3"],
            ["DG5","DG5","DG5","DG","DG"],["DA5","DA5","DA5","DA","DA"],["DT5","DT5","DT5","DT","DT"],["DC5","DC5","DC5","DC","DC"],
            ["G","DGN","DG5","DG","DG3"],["A","DAN","DA5","DA","DA3"],["T","DTN","DT5","DT","DT3"],["C","DCN","DC5","DC","DC3"]]



DNA = Structure(DNAresidues,
                residue_length=DNAlengths,
                rotating_elements=DNAphosphate+DNAoc5+DNAbase,
                alias=DNAalias)

cpx = Complex()
cpx.add_chain("G C A T", DNA)
cpx.build()
chain = cpx.chains[0]
fil = open("TestInputGeometry.pdb","w")
fol = open("TestOutputGeometry.pdb","w")
app.PDBFile.writeModel(cpx.topology, cpx.positions,
                       file=fil, modelIndex=1)
chain.rotate_in_residue(3,1,30.)
chain.prepend_sequence("A C")
cpx.rebuild()
chain.append_sequence("A C")
cpx.rebuild()
chain.append_sequence("A C")
cpx.rebuild()
app.PDBFile.writeModel(cpx.topology, cpx.positions,
                       file=fol, modelIndex=1)
