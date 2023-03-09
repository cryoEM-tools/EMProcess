import numpy as np
from cryosparc_compute import dataset


def pull_coordinates_locations(particles):
    cs_particles = dataset.Dataset.load(particles)

    # determine X,Y coordinates
    xy_shapes = cs_particles['location/micrograph_shape']
    coordinateX,coordinateY = np.around(
        np.array(
            [
                cs_particles['location/center_x_frac'],
                cs_particles['location/center_y_frac']])*xy_shapes.T, 0)

    return coordinateX, coordinateY, cs_particles['location/micrograph_path']
