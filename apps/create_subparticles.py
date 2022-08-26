import argparse
import numpy as np
from EMProcess.formats.star import Particles
from EMProcess import local_rec
from EMProcess import maps
import sys

def process_command_line(argv):

    parser = argparse.ArgumentParser(
        description='Creates subparticles for local reconstruction')
    parser.add_argument(
        '--particles', type=str, help='Particles filename to create subparticles')
    parser.add_argument(
        '--symm', type=str,
        help='Symmetry of particles, defined by RELION. i.e. C1, I1, I2, ...')
    parser.add_argument(
        '--symm_file', type=str, default=None, help="RELION symmetry file.")
    parser.add_argument(
        '--asymmetric_points', default=None, type=str,
        help="optionally supply asymmetric points that define location of subparticles")
    parser.add_argument(
        '--mask', type=str,
        help='Filename of mask containing asymmetric unit.')
    parser.add_argument(
        '--ang_pix', type=float, default=1.0, help='Pixel size in Å')
    parser.add_argument(
        '--output', type=str, default='subparticles.star',
        help='Output filename for subparticles')
    parser.add_argument(
        '--filter_front', action='store_true',
        help='Optionally filter particles to only include front-facing orientations.')
    parser.add_argument(
        '--filter_distance', type=float, default=None,
        help='Min dist between particles in image in pixels')

    args = parser.parse_args(argv)

    return


def main(argv):

    args = process_command_line(argv)

    # obtain symmetry matrix set
    if args.symm_file:
        symm_mat_set = local_rec.load_matrix_symmetry_file(args.symm_file)
    elif args.symm:
        symm_mat_set = local_rec.matrix_from_symmetry(args.symm)

    # obtain distance and vector from origin to center of asymmetric unit
    dist, v0 = maps.mask_distance_vec(args.mask)

    # load particles
    particles = Particles(args.particles)

    if args.asymmetric_points is None:
        # generate subparticles
        subparticles = local_rec.create_subparticles(
            particles, symm_mat_set=symm_mat_set, asymmetric_distance=dist,
            ang_pix=args.ang_pix, v0=v0, filter_front=args.filter_front)
    else:
        asymmetric_points = np.loadtxt(args.asymmetric_points)
        subparticles = local_rec.create_subparticles(
            particles, asymmetric_points=asymmetric_points,
            asymmetric_distance=dist, ang_pix=args.ang_pix,
            filter_front=args.filter_front)

    # filter distance
    if args.filter_distance is not None:
        subparticles = local_rec.filter_subparticles_distance(
            subparticles, filter_distance=args.filter_distance, ang_pix=args.ang_pix)

    # save subparticles
    subparticles.write(args.output)

if __name__=='__main__':
    sys.exit(main(sys.argv))
