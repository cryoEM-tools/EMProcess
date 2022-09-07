import argparse
import numpy as np
import os
import sys
from EMProcess.formats import star

def process_command_line(argv):

    parser = argparse.ArgumentParser(
        description='Joins or appends data from multiple star files.'
                    'Joining star files concatenates entries with the same'
                    'number of labels. Appending adds unique labels to a set'
                    'number of particles.')
    parser.add_argument(
        '-i', '--inputs', nargs='+', type=str, help='Input star filenames.')
    parser.add_argument(
        '-a', '--action', choices=['append','join', 'reorder'], default='join',
        help='Determine if appending, joining, or reordering labels star files.'
             'Appending concatenates the unique labels between 2 Particle'
             'starfiles. Joining concatenates particle entried between multiple'
             'starfiles with the same labels. Reordering orders the labels to'
             'fit relion preferences.')
    parser.add_argument(
        '--add-filename-path', action='store_true',
        help='Optionally add filename as a label. Only valid for joining'
             'star files.')
    parser.add_argument(
        '-o', '--output_name', type=str, default='new_star.star',
        help='The output star filename.')

    args = parser.parse_args(argv)

    return args


def main(argv):

    args = process_command_line(argv)

    # parse inputs
    input_filenames = np.array(
        [os.path.abspath(i) for i in args.inputs], dtype=str)

    # determine if joining or appending
    if args.action == 'join':
        assert input_filenames.shape[0] > 1
        new_star = star.join_stars(input_filenames, args.add_filename_path)
    elif args.action == 'append':
        assert input_filenames.shape[0] == 2
        particles0 = star.Particles(input_filenames[0])
        particles1 = star.Particles(input_filenames[1])
        new_star = star.append_particles(particles0, particles1)
    elif args.action == 'reorder':
        assert input_filenames.shape[0] == 1
        new_star = star.Particles(input_filenames[0])
        new_star.relion_label_ordering()

    # write new starfile output
    new_star.write(args.output_name)


if __name__=='__main__':
    sys.exit(main(sys.argv))
