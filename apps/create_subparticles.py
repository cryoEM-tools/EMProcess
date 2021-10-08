import argparse
from EMProcess.formats.star import Particles
from EMProcess import local_rec

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
    '--filter_distance', default=None,
    help='Min dist between particles in image in pixels')


def entry_point():

    args = parser.parse_args()

    # obtain symmetry matrix set
    if args.symm_file:
        symm_mat_set = local_rec.load_matrix_symmetry_file(args.symm_file)
    else:
        symm_mat_set = local_rec.matrix_from_symmetry(args.symm)

    # obtain distance and vector from origin to center of asymmetric unit
    dist, v0 = local_rec.mask_distance_vec(args.mask)

    # load particles
    particles = Particles(args.particles)

    # generate subparticles
    subparticles = local_rec.create_subparticles(
        particles, symm_mat_set, asymmetric_distance=dist, ang_pix=args.ang_pix,
        v0=v0, filter_front=args.filter_front)

    # filter distance
    if args.filter_distance:
        subparticles = local_rec.filter_subparticles_distance(
            subparticles, filter_distance=args.filter_distance, ang_pix=args.ang_pix)

    # save subparticles
    subparticles.write(args.output)

if __name__=='__main__':
    entry_point()
