import argparse
import numpy as np
from EMProcess import maps

parser = argparse.ArgumentParser(
    description='Extracts boxed density from maps')
parser.add_argument(
    '--input_name', type=str, help='Input map filename')
parser.add_argument(
    '--trim_value', type=float,
    help="Values less than the trim_value are set to zero..")
parser.add_argument(
    '--output_name', type=str, default='mask.mrc',
    help='The output map filename.')

def entry_point():

    # parse arguments
    args = parser.parse_args()

    maps.trim_map_by_value(args.input_name, args.trim_value, output_name=args.output_name)

if __name__=='__main__':
    entry_point()
