import mrcfile as mrc
import numpy as np


def mask_distance_vec(mask_filename):
    """ Calculates the distance and angle of the center of the mask
    from the mask's origin.

    Inputs
    ----------
    mask_filename : str,
        The name of the .mrc file containing a mask of a single asymmetric unit.
        The distance and vector are used to create subparticles.

    Returns
    ----------
    distance : float,
        The distance from the center of the box to the asymmetric unit (in pixels)
    vec : nd.array, shape=(3, )
        The vector from the origin of the box to the asymmetric unit.
    """
    mask = mrc.open(mask_filename)
    z,y,x = mask.data.nonzero()
    x_center = (x.max() + x.min()) / 2.
    y_center = (y.max() + y.min()) / 2.
    z_center = (z.max() + z.min()) / 2.
    particle_center = np.array(
        [
            x_center, y_center, z_center])
    center_point = np.array([i/2 for i in mask.data.shape])

    distance = np.sqrt(
        np.sum(
            [
                (x0-x1)**2
                for x0,x1 in zip(particle_center, center_point)]))
    vec = (particle_center - center_point)
    if np.all(particle_center == center_point):
        raise
    vec = vec / np.sqrt(np.sum(vec**2))
    return distance, vec


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


def current_box_dims(map_density):
    if type(map_density) is mrc.mrcfile.MrcFile:
        x_dim = map_density.header['mx']
        y_dim = map_density.header['my']
        z_dim = map_density.header['mz']
    elif type(map_density) is str:
        with mrc.open(map_density) as map_density:
            x_dim = map_density.header['mx']
            y_dim = map_density.header['my']
            z_dim = map_density.header['mz']
    box_dims = np.array([x_dim, y_dim, z_dim])
    return box_dims


def find_box_dims(map_density):
    """Returns the box dimensions of density within a specified map."""
    if type(map_density) is mrc.mrcfile.MrcFile:
        z,y,x = map_density.data.nonzero()
        box_dims = np.array(
            [
                np.ceil(x.max() - x.min()), 
                np.ceil(y.max() - y.min()),
                np.ceil(z.max() - z.min())], dtype=int)
    elif type(map_density) is str:
        with mrc.open(map_density) as map_density:
            z,y,x = map_density.data.nonzero()
            box_dims = np.array(
                [
                    np.ceil(x.max() - x.min()), 
                    np.ceil(y.max() - y.min()),
                    np.ceil(z.max() - z.min())], dtype=int)
    return box_dims


def trim_map_by_value(
        map_filename, value, output_name='map_trimmed.mrc', **kwargs):
    with mrc.open(map_filename) as map_density:
        mask_iis = map_density.data >= value

        with mrc.new(output_name, **kwargs) as mrc_output:
            mrc_output_data = np.zeros(map_density.data.shape, dtype='<f4')
            mrc_output_data[mask_iis] = map_density.data[mask_iis]
            mrc_output.set_data(mrc_output_data)
            mrc_output.voxel_size = float(map_density.voxel_size['x'])
            mrc_output.update_header_from_data()
            mrc_output.update_header_stats()
    return


def extract_from_map(
        map_filename, distance_from_origin=None, vec=None, extraction_center=None,
        box_dims=None, mask_filename=None, recenter=True, output_name='extracted_map.mrc',
        **kwargs):
    """ Extracts a portion of density from an .mrc file
    
    Inputs
    ----------
    map_filename : str,
        The filename containing the original map.
    distance_from_origin : float, default=None,
        Provide the distance from the origin to extract (in pixels).
    vec : array-like, default=None,
        The vector pointing from the origin to the location of extraction
        center.
    extraction_center : array-like, default=None,
        The center voxel coordinate to re-extract from.
    box_dims : int or array-like, default=None, shape=(1,) or shape=(3,),
        Provide the dimensions of the box to extract, or optionally provide
        dimenions of a rectangular prism to extract. Dimenions are in pixels.
        If none provided, determines minimum box of current density.
    mask_filename : str, default=None,
        Optionally use input mask to determine box size and extract specific density.
    output_name : str, default='extracted_map.mrc',
        The output filename for the extracted mrc file.
    """
    if mask_filename is not None:
        with mrc.open(mask_filename) as mask:
            mask_shape = mask.data.shape
            mask_iis = mask.data.nonzero()

    with mrc.open(map_filename) as map_density:

        if mask_filename is not None:
            assert map_density.data.shape == mask_shape

        # get voxel size
        voxel_size = float(map_density.voxel_size['x'])
        
        # determine extraction center
        if extraction_center is None:
            z_dim, y_dim, x_dim = map_density.data.shape
            map_dim = np.array([x_dim, y_dim, z_dim])
            map_center = map_dim / 2
            if recenter:
                extraction_center = map_center + vec*distance_from_origin
            else:
                extraction_center = map_center

        # determine box dimensions
        if box_dims is None:
            if mask_filename is None:
                box_dims = find_box_dims(map_density)
            else:
                box_dims = find_box_dims(mask_filename)
        elif type(box_dims) is int:
            box_dims = np.array([box_dims, box_dims, box_dims])
        elif type(box_dims) is list:
            box_dims = np.array(box_dims)
        box_edges = box_dims / 2
        extraction_corner = extraction_center - box_edges

        # create initial indices for extraction and placement of voxels        
        old_map_start_stop = np.array(
            np.around(
                np.vstack([extraction_corner, (extraction_corner + box_dims)]).T, 0), dtype=int)
        new_map_start_stop = np.array([np.zeros(3), box_dims], dtype=int).T
        
        # clip bottom of new map
        new_map_start_stop[old_map_start_stop[:,0] < 0, 0] -= old_map_start_stop[old_map_start_stop[:,0] < 0, 0]

        # clip top of new map
        new_map_start_stop[old_map_start_stop[:,1] > map_dim, 1] -= \
            old_map_start_stop[old_map_start_stop[:,1] > map_dim, 1] - map_dim[old_map_start_stop[:,1] > map_dim]

        # clip old map
        old_map_start_stop[old_map_start_stop[:,0] < 0, 0] = 0
        old_map_start_stop[old_map_start_stop[:,1] > map_dim, 1] = map_dim[old_map_start_stop[:,1] > map_dim]

        # optionally take only masked region        
        if mask_filename is not None:
            map_data = np.zeros(shape=map_density.data.shape, dtype='<f4')
            map_data[mask_iis] = map_density.data[mask_iis]
        else:
            map_data = map_density.data

        print(new_map_start_stop, old_map_start_stop)
        print(box_dims)

        # write new map        
        with mrc.new(output_name, **kwargs) as mrc_output:
            mrc_output.set_data(np.zeros(shape=(box_dims[2], box_dims[1], box_dims[0]), dtype='<f4'))
            mrc_output.voxel_size = voxel_size
            mrc_output.data[
                    new_map_start_stop[2,0]:new_map_start_stop[2,1],
                    new_map_start_stop[1,0]:new_map_start_stop[1,1],
                    new_map_start_stop[0,0]:new_map_start_stop[0,1]] = \
                map_data[
                    old_map_start_stop[2,0]:old_map_start_stop[2,1],
                    old_map_start_stop[1,0]:old_map_start_stop[1,1],
                    old_map_start_stop[0,0]:old_map_start_stop[0,1]]
            mrc_output.update_header_from_data()
            mrc_output.update_header_stats()
    return
