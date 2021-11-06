import math
import mrcfile as mrc
import numpy as np
import tqdm
import sys
from ..formats.star import Particles 
from scipy.spatial.transform import Rotation as R


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

    # determine number of particles
    n_particles = int(subparticles._n_particles)
    
    #obtain micrograph names
    mic_names = np.copy(subparticles.data_particles.MicrographName.data)
    unique_mic_names = np.unique(mic_names)
    
    # initialize list
    iis_to_filter_total = []
    
    #obtain coordinates
    x_coords = subparticles.data_particles.CoordinateX.data * ang_pix
    y_coords = subparticles.data_particles.CoordinateY.data * ang_pix
    coords = np.array([x_coords, y_coords]).T
    
    # loop over micrographs names
    for mic_name in unique_mic_names:
        
        # find indices of micrograph
        iis_mic = np.where(mic_names == unique_mic_names)[0]

        # shuffle indices to not introduce orientation bias
        iis_random = np.arange(iis_mic.shape[0])
        np.random.shuffle(iis_random)
        
        #shuffle coordinates
        coords_tmp = coords[iis_mic[iis_random]]
        
        # initialize filter
        iis_to_filter = np.zeros(shape=(0,), dtype=int)

        for n in np.arange(coords_tmp.shape[0]):
            if not n in iis_to_filter:
                dists_to_n = np.sqrt(np.sum((coords_tmp - coords_tmp[n])**2, axis=1))
                iis_to_filter_tmp = np.where(dists_to_n <= filter_distance)[0]
                iis_to_filter = np.concatenate([iis_to_filter, iis_to_filter_tmp[iis_to_filter_tmp > n]])
        
        iis_to_filter_total.append(iis_mic[iis_random[np.unique(iis_to_filter)]])
    
    iis_to_filter_total = np.concatenate(iis_to_filter_total)
    iis = np.array(
        np.setdiff1d(
            np.arange(coords.shape[0]), iis_to_filter_total))
    subparticles_filtered = subparticles.copy()[iis]

    return subparticles_filtered


def create_subparticles(
        particles, symm_mat_set=None, asymmetric_points=None, asymmetric_distance=1.0,
        ang_pix=None, v0=[0,0,1], filter_front=False):
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
    if asymmetric_points is None:
        asymmetric_points = np.array(
            [R.from_matrix(mat).apply(v0) for mat in symm_mat_set])
    else:
        asymmetric_points /= np.sqrt(np.sum(asymmetric_points**2, axis=1))[:,None]

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
            [particle for i in np.arange(asymmetric_points.shape[0])])
        
        # determine the new euler angles as referenced from the v0 vector
        try:
            new_rots, new_tilts, new_psis = np.array(
                [
                    R.align_vectors([p], [v0])[0].as_euler('zyz', degrees=True)
                    for p in rotated_points]).T
        except:
            print("euler angle estimation failed!")
            print("v0: %s \n rotated_points: %s" % (v0, rotated_points))
            sys.exit()
        
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
