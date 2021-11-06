import numpy as np
import os
from .. import tools


def relion_create_symmetry_ops_file(symm_ops, filename):
    """ Create a symmetry operator file
    by running relion_refine --print_symmetry_ops."""

    cmd = 'relion_refine --sym %s --print_symmetry_ops --o ./ > %s' % (symm_ops, filename)
    tools.run_commands(cmd)
    return


def matrix_from_symmetry(symString):
    """ Create the symmetry matrices from a given string using Relion convention.
    We use use 'relion_refine --sym' to generate the symmetry file and
    then parse it to load the matrices. """

    tmpSymFile = "relion_symops.tmp"
    relion_create_symmetry_ops_file(symString, tmpSymFile)
    matrices = load_matrix_symmetry_file(tmpSymFile)
    os.remove(tmpSymFile)
    return matrices


def load_matrix_symmetry_file(filename):
    """ Load matrix symmetry file."""

    with open(filename, 'r') as f:
        lines = f.readlines()
    
    rot_mats = np.zeros(shape=(0,3,3))
    
    for line_num in np.arange(len(lines)):
        
        line = lines[line_num]
        
        if " R(" in line:
            row0 = lines[line_num+1].split()
            row1 = lines[line_num+2].split()
            row2 = lines[line_num+3].split()
            
            new_mat = [
                list(map(float, row0)),
                list(map(float, row1)),
                list(map(float, row2))]
            
            rot_mats = np.vstack([rot_mats, [new_mat]])
            
    return rot_mats
