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


def symmetry_expand_density(
        mask_filename, symm_mat, output_name='expanded.mrc', **kwargs):
    """ Rotates a density map around specified symmetries

    Inputs
    ----------
    mask_filename : str,
        The filename of the density to rotate.
    symm_mat : array-like or str,
        The symmetry matrices for rotating the density.
    output_name : str,
        Filename to write rotated density.
    """
    # if symm_mat is a string, load the matricies
    if type(symm_mat) is str:
        symm_mat = load_matrix_symmetry_file(symm_mat)
    
    # load mrc file and extract density as 1D array with locations
    with mrc.open(mask_filename) as mask:
        mask_shape = mask.data.shape
        voxel_size = float(mask.voxel_size['x'])
        x,y,z = mask.data.nonzero()
        center_point = np.array([i/2 for i in mask_shape], dtype=int)
        coords_centered = np.array([x,y,z]).T - center_point
        mask_data = np.copy(mask.data[x,y,z])
        del x, y, z
    
    # apply rotation to coordinates (xyz location is reversed to true x,y,z, hence [::-1])
    rotated_mask_coords = np.array(
        np.around(
            [
                R.from_matrix(mat).apply(coords_centered[:,::-1])
                for mat in symm_mat]), dtype=int)[:,:,::-1]
    
    with mrc.new(output_name, **kwargs) as mrc_output:
        mrc_output.set_data(np.zeros(mask_shape, dtype='<f4'))
        mrc_output.voxel_size = voxel_size
        for coords in rotated_mask_coords:
            x_tmp,y_tmp,z_tmp = (coords+center_point).T
            mrc_output.data[x_tmp,y_tmp,z_tmp] = mask_data
        mrc_output.update_header_from_data()
        mrc_output.update_header_stats()
    return
