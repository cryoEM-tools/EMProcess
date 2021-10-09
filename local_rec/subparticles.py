import math
import mrcfile as mrc
import numpy as np
import tqdm
from ..formats.star import Particles 
from scipy.spatial.transform import Rotation as R


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
    x,y,z = mask.data.nonzero()
    particle_center = np.array(
        [   
            x.mean(), y.mean(), z.mean()])
    center_point = np.array([i/2 for i in mask.data.shape])

    distance = np.sqrt(
        np.sum( 
            [   
                (x0-x1)**2
                for x0,x1 in zip(particle_center, center_point)]))
    vec = (particle_center - center_point)
    vec = vec / np.sqrt(np.sum(vec**2))
    return distance, vec[::-1]


def filter_subparticles_distance(subparticles, filter_distance=1.0, ang_pix=1.0):
    """ Filters particles whose X,Y coordinates are too close to eachother

    Inputs
    ----------
    subparticles : object, EMProcess.Particles, shape=(n_particles, ),
        Particles object containing particles to filter.
    filter_distance : float, default=1.0,
        The minimum allowable distance between subparticles (in Å).
    ang_pix : float, default=1.0,
        The pixel size of micrographs (in Å)

    Returns
    ----------
    subparticles_filtered : object, EMProcess.Particles, shape=(n_filtered_particles, ),
        Particles object containing particles that are have a nearest neighbor
        distance greater than the filter distance.
    """
    # shuffle indicies to not introduce orientation bias
    iis_random = np.arange(int(subparticles._n_particles))
    np.random.shuffle(iis_random)
    
    #obtain coordinates
    x_coords = subparticles.data_particles.CoordinateX.data * ang_pix
    y_coords = subparticles.data_particles.CoordinateY.data * ang_pix
    coords = np.array([x_coords, y_coords]).T[iis_random]

    subparticles_filtered = subparticles.copy()
    iis_to_filter = np.zeros(shape=(0,))
    
    for n in np.arange(subparticles_filtered._n_particles, dtype=int):
        if not n in iis_to_filter:
            dists_to_n = np.sqrt(np.sum((coords - coords[n])**2, axis=1))
            iis_to_filter_tmp = np.where(dists_to_n <= filter_distance)[0]
            iis_to_filter = np.concatenate([iis_to_filter, iis_to_filter_tmp[iis_to_filter_tmp > n]])
    iis_to_filter = np.unique(iis_to_filter)
    iis = np.array(
        np.setdiff1d(
            np.arange(subparticles_filtered._n_particles),
            iis_to_filter), dtype=int)
    subparticles_filtered = subparticles_filtered[iis_random[iis]]
    return subparticles_filtered


def create_subparticles(
        particles, symm_mat_set, asymmetric_distance=1.0, ang_pix=None,
        v0=[0,0,1], filter_front=False):
    """ Creates subparticles based on applied symmetry.

    Inputs
    ----------
    particles : object, EMProcess.Particles, shape=(n_particles,)
        Object of particles STAR file to create subparticles from.
    symm_mat_set : array-like, shape=(n_rotations, 3, 3),
        A set of symmetry matricies that define the symmetry of the particle.
    asymmetric_distance : float, default = 1.0,
        The distance from the center of the particle to the center of the
        asymmetric unit. Can be calculated from a .mrc file.
    ang_pix : float, default=None
        Pixel size of input image in Å. If none is supplied, uses a value of 1..
    v0 : array-like, shape=(3, ),
        Initial orientation of subparticle based on model density. Symmetry
        operations are performed on this to define all symmetry positions.
        Defaults to being applied to a vector pointing in z-axis only.

    Returns
    ----------
    subparticles : object, EMProcess.Particles, shape=(n_particles * n_rotations,)
        The Particles object containing all subparticles to be extracted.
    front_facing : nd.array, shape=(n_particles * n_rotations, )
        A boolean array detailing if a particular subparticle is outward facing
        with respect to the micrograph image.
    """

    # generate all rotations
    asymmetric_points = np.array([R.from_matrix(mat).apply(v0) for mat in symm_mat_set])

    # generate lists
    subparticles = []
    front_facing = []

    for particle_num in tqdm.tqdm(range(int(particles._n_particles))):
        
        particle = particles[particle_num]

        # get euler angles to rotate points that define asymmetric unit
        rot = math.radians(particle.data_particles.AngleRot.data[0])
        tilt = math.radians(particle.data_particles.AngleTilt.data[0])
        psi = math.radians(particle.data_particles.AnglePsi.data[0])

        # define rotation matrix and rotate the points that define asymmetric units
        r_particle = R.from_euler('zyz', [rot, tilt, psi], degrees=False)
        rotated_points = r_particle.apply(
            asymmetric_points * asymmetric_distance)
        
        # create new Particles instance to represent subparticles
        new_particles = Particles.concatenate(
            [particle for i in np.arange(symm_mat_set.shape[0])])
        
        # determine the new euler angles as referenced from the v0 vector
        new_rots, new_tilts, new_psis = np.array(
            [
                R.align_vectors([p], [v0])[0].as_euler('zyz', degrees=True)
                for p in rotated_points]).T
        
        # set new angles
        new_particles.data_particles.AngleRot[:] = new_rots
        new_particles.data_particles.AngleTilt[:] = new_tilts
        new_particles.data_particles.AnglePsi[:] = new_psis

        CoordinateX_orig = particle.data_particles.CoordinateX.data[0]
        CoordinateY_orig = particle.data_particles.CoordinateY.data[0]

        # front-facing if z-coordinate is positive
        front_facing_tmp = np.array((rotated_points[:,2] > 0), dtype=bool)

        # pull x, y, z coordinates of rotated asymmetric centers
        x = rotated_points[:, 0]
        y = rotated_points[:, 1]
        z = rotated_points[:, 2]

        # recenter around original particle origin (decimal origin)
        if hasattr(particle.data_particles, 'OriginX'):
            x += new_particles.data_particles.OriginX.data
            y += new_particles.data_particles.OriginY.data
        elif hasattr(particle.data_particles, 'OriginXAngst'):
            x += new_particles.data_particles.OriginXAngst.data / ang_pix
            y += new_particles.data_particles.OriginYAngst.data / ang_pix
        else:
            raise

        # adjust defocus value
        if hasattr(particle.data_particles, 'DefocusU'):
            z_ang = ang_pix * z
            new_particles.data_particles.DefocusU[:] = particle.data_particles.DefocusU.data[0] + z_ang
            new_particles.data_particles.DefocusU[:] = particle.data_particles.DefocusV.data[0] + z_ang

        # recenter in whole image (large pixel value)
        x_coord = CoordinateX_orig - x
        y_coord = CoordinateY_orig - y
        new_particles.data_particles.CoordinateX[:] = np.floor(x_coord)
        new_particles.data_particles.CoordinateY[:] = np.floor(y_coord)
        if hasattr(particle.data_particles, 'OriginX'):
            new_particles.data_particles.OriginX[:] = x % 1
            new_particles.data_particles.OriginY[:] = y % 1
        elif hasattr(particle.data_particles, 'OriginXAngst'):
            new_particles.data_particles.OriginXAngst[:] = (x * ang_pix) % 1
            new_particles.data_particles.OriginYAngst[:] = (y * ang_pix) % 1

        # append to running lists
        subparticles.append(new_particles)
        front_facing.append(front_facing_tmp)

    # concatenate running lists
    front_facing = np.concatenate(front_facing)
    if filter_front:
        subparticles = Particles.concatenate(subparticles)[front_facing]
    else:
        subparticles = Particles.concatenate(subparticles)
    return subparticles
