import numpy as np
from cryosparc_compute import dataset


def pull_coordinates_locations(particles):
    cs_particles = dataset.Dataset()
    cs_particles.from_file(particles)

    micrograph_pixel_size = None
    if micrograph_pixel_size is None:
        try:
            micrograph_pixel_size = cs_particles['blob/psize_A']
        except:
            micrograph_pixel_size = cs_particles['mscope_params/psize_A']
    particle_pixel_size = cs_particles['alignments2D/psize_A']
    shifts_x, shifts_y = cs_particles['alignments2D/shift'].T * particle_pixel_size

    micrograph_nx, micrograph_ny = cs_particles['location/micrograph_shape'].T
    locations_xs = cs_particles['location/center_x_frac'] * \
        micrograph_pixel_size * micrograph_nx - shifts_x
    locations_ys = cs_particles['location/center_y_frac'] * \
        micrograph_pixel_size * micrograph_ny - shifts_y

    coordinateX = np.around(locations_xs / micrograph_pixel_size, 0)
    coordinateY = np.around(locations_ys / micrograph_pixel_size, 0)

    return coordinateX, coordinateY, cs_particles['location/micrograph_path']
