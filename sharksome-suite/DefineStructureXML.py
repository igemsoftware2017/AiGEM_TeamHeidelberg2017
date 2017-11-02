from Structure import Structure
#from xml.etree.ElementTree import ElementTree, Element, SubElement, Comment, tostring, dump
from lxml import etree

def WriteStructureXML(structure, path):
    tree = etree.Element('structure')
    if(structure.residue_path!=None):
        Xresidue_path = etree.Element('residuePath')
        Xresidue_path.text = structure.residue_path
        tree.append(Xresidue_path)
    Xaliases = etree.Element('alias')
    for key, values in structure.alias.iteritems():
        Xalias = etree.Element('element')
        Xalias.set('name',key)
        Xalone = etree.Element('alone')
        Xalone.text = values[0]
        Xalias.append(Xalone)
        Xstart = etree.Element('start')
        Xstart.text = values[1]
        Xalias.append(Xstart)
        Xmiddle = etree.Element('middle')
        Xmiddle.text = values[2]
        Xalias.append(Xmiddle)
        Xend = etree.Element('end')
        Xend.text = values[3]
        Xalias.append(Xend)
        #append alias element to the list of aliases.
        Xaliases.append(Xalias)
    tree.append(Xaliases)

    
    for name in structure.residue_names:
        Xresidue = etree.Element('residue')
        Xresidue.set('name', name)
        Xresidue.set('length', str(structure.residue_length[name]))
        #Adding append/prepend connects
        connect = structure.connect[name]

        Xappend = etree.Element('append')
        Xprepend = etree.Element('prepend')

        XappNewFatom = etree.Element('newFirstAtom')
        XappNewFatom.text = str(connect[0][0])
        XappOldLatom = etree.Element('oldLastAtom')
        XappOldLatom.text = str(connect[0][1])
        XappBondL = etree.Element('bondLength')
        XappBondL.text = str(connect[2])
        Xappend.append(XappNewFatom)
        Xappend.append(XappOldLatom)
        Xappend.append(XappBondL)

        XprpNewLatom = etree.Element('newLastAtom')
        XprpNewLatom.text = str(connect[1][0])
        XprpOldFatom = etree.Element('oldFirstAtom')
        XprpOldFatom.text = str(connect[1][1])
        XprpBondL = etree.Element('bondLength')
        XprpBondL.text = str(connect[3])
        Xprepend.append(XprpNewLatom)
        Xprepend.append(XprpOldFatom)
        Xprepend.append(XprpBondL)
        
        Xresidue.append(Xappend)
        Xresidue.append(Xprepend)

        #Adding backbone information
        
        backbone = structure.backbone_elements[name]
        if backbone != None:
        #INFO: [[start, middle_pre, bond], [middle_post, end]]
            Xbackbone = etree.Element('backbone')

            Xbbstart = etree.Element('start')
            Xbbstart.text = str(backbone[0][0])
            Xbackbone.append(Xbbstart)
            Xbbmiddle_pre = etree.Element('middle_pre')
            Xbbmiddle_pre.text = str(backbone[0][1])
            Xbackbone.append(Xbbmiddle_pre)
            Xbbbond = etree.Element('bond')
            Xbbbond.text = str(backbone[0][2])
            Xbackbone.append(Xbbbond)
            Xbbmiddle_post = etree.Element('middle_post')
            Xbbmiddle_post.text = str(backbone[1][0])
            Xbackbone.append(Xbbmiddle_post)
            Xbbend = etree.Element('end')
            Xbbend.text = str(backbone[1][1])
            Xbackbone.append(Xbbend)

            Xresidue.append(Xbackbone)
            
        #Adding rotating elements
        i = 0
        for rotation in structure.rotating_elements[name]:
            if rotation != [None]:
                Xrotation = etree.Element('rotation')
                #Vorschlag Rotationen indizieren
                Xrotation.set('index', str(i))
                i += 1
                
                Xstart = etree.Element('start')
                Xstart.text = str(rotation[0])
                Xrotation.append(Xstart)
                Xbond = etree.Element('bond')
                Xbond.text = str(rotation[1])
                Xrotation.append(Xbond)
                #Handling the special 'end' property of the end rotation tag
                Xend = etree.Element('end')
                if rotation[2] == None:
                    Xend.text = 'end'                    
                else:
                    Xend.text = str(rotation[2])
                Xrotation.append(Xend)
                Xresidue.append(Xrotation)
        
        #append complete residue definition to tree
        tree.append(Xresidue)
    #nice output using lxml
    result = etree.tostring(tree, pretty_print=True)
    #Writes the output tree to the given path. Implement error handling?
    file = open(path, "w")
    file.write(result)
    file.close()



#Specification of DNA-residues begins here ---------------------------------------------------------------------------------------#
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

DNAoc5 = [["DGN", 1, 2, None],
          ["DAN", 1, 2, None],
          ["DTN", 1, 2, None],
          ["DCN", 1, 2, None],
          ["DG5", 1, 2, None],
          ["DA5", 1, 2, None],
          ["DT5", 1, 2, None],
          ["DC5", 1, 2, None],
          ["DG", 3, 4, None],
          ["DA", 3, 4, None],
          ["DT", 3, 4, None],
          ["DC", 3, 4, None],
          ["DG3", 3, 4, None],
          ["DA3", 3, 4, None],
          ["DT3", 3, 4, None],
          ["DC3", 3, 4, None]]

DNAbase = [["DGN", 8, 10, -7],
           ["DAN", 8, 10, -7],
           ["DTN", 8, 10, -7],
           ["DCN", 8, 10, -7],
           ["DG5", 8, 10, -6],
           ["DA5", 8, 10, -6],
           ["DT5", 8, 10, -6],
           ["DC5", 8, 10, -6],
           ["DG", 10, 12, -6],
           ["DA", 10, 12, -6],
           ["DT", 10, 12, -6],
           ["DC", 10, 12, -6],
           ["DG3", 10, 12, -7],
           ["DA3", 10, 12, -7],
           ["DT3", 10, 12, -7],
           ["DC3", 10, 12, -7]]

DNAoc3 = [["DGN", -4, -2, None],
          ["DAN", -4, -2, None],
          ["DTN", -4, -2, None],
          ["DCN", -4, -2, None],
          ["DG5", -3, -1, None],
          ["DA5", -3, -1, None],
          ["DT5", -3, -1, None],
          ["DC5", -3, -1, None],
          ["DG", -3, -1, None],
          ["DA", -3, -1, None],
          ["DT", -3, -1, None],
          ["DC", -3, -1, None],
          ["DG3", -4, -2, None],
          ["DA3", -4, -2, None],
          ["DT3", -4, -2, None],
          ["DC3", -4, -2, None]]


DNAbackbone = [["DGN", 0, 8, 10, -7, -1],
               ["DAN", 0, 8, 10, -7, -1],
               ["DTN", 0, 8, 10, -7, -1],
               ["DCN", 0, 8, 10, -7, -1],
               ["DG5", 0, 8, 10, -6, -1],
               ["DA5", 0, 8, 10, -6, -1],
               ["DT5", 0, 8, 10, -6, -1],
               ["DC5", 0, 8, 10, -6, -1],
               ["DG", 0, 10, 12, -6, -1],
               ["DA", 0, 10, 12, -6, -1],
               ["DT", 0, 10, 12, -6, -1],
               ["DC", 0, 10, 12, -6, -1],
               ["DG3", 0, 10, 12, -7, -1],
               ["DA3", 0, 10, 12, -7, -1],
               ["DT3", 0, 10, 12, -7, -1],
               ["DC3", 0, 10, 12, -7, -1]]

DNAlengths = [32, 31, 31, 29,
              33, 32, 32, 30,
              31, 30, 30, 28,
              34, 33, 33, 31]


DNAalias = [["DGN","DGN","DG5","DG","DG3"],["DAN","DAN","DA5","DA","DA3"],["DTN","DTN","DT5","DT","DT3"],["DCN","DCN","DC5","DC","DC3"],
            ["DG3","DG3","DG","DG","DG3"],["DA3","DA3","DA","DA","DA3"],["DT3","DT3","DT","DT","DT3"],["DC3","DC3","DC","DC","DC3"],
            ["DG5","DG5","DG5","DG","DG"],["DA5","DA5","DA5","DA","DA"],["DT5","DT5","DT5","DT","DT"],["DC5","DC5","DC5","DC","DC"],
            ["G","DGN","DG5","DG","DG3"],["A","DAN","DA5","DA","DA3"],["T","DTN","DT5","DT","DT3"],["C","DCN","DC5","DC","DC3"]]

#Specification of DNA-residues ends here -----------------------------------------------------------------------------------------#

#Build Structure-object for DNA residues
DNA = Structure(DNAresidues,
                residue_length=DNAlengths,
                rotating_elements=DNAphosphate+DNAoc5+DNAbase+DNAoc3,
                backbone_elements=DNAbackbone,
                alias=DNAalias)

WriteStructureXML(DNA,"DNA.xml")
