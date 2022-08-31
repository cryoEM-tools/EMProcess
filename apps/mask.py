import argparse
import numpy as np
import sys
from EMProcess import maps

def process_command_line(argv):

    parser = argparse.ArgumentParser(
        description='Extracts boxed density from maps')
    parser.add_argument(
        '-i', '--input_name', type=str, help='Input map filename')
    parser.add_argument(
        '-t', '--trim_value', type=float,
        help="Values less than the trim_value are set to zero..")
    parser.add_argument(
        '-o', '--output_name', type=str, default='mask.mrc',
        help='The output map filename.')

    print(argv)
    args = parser.parse_args(argv)

    return args


def main(argv):

    args = process_command_line(argv)

    maps.trim_map_by_value(args.input_name, args.trim_value, output_name=args.output_name)

if __name__=='__main__':
    sys.exit(main(sys.argv))
