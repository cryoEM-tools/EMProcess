import argparse
import numpy as np
from EMProcess import maps

parser = argparse.ArgumentParser(
    description='Extracts boxed density from maps')
parser.add_argument(
    '--input_name', type=str, help='Input map filename')
parser.add_argument(
    '--output_name', type=str,
    help='The output map filename.')
parser.add_argument(
    '--mask', type=str, default=None,
    help="Mask filename to use for determining box location and size.")
parser.add_argument(
    '--distance_from_origin', default=None, type=float,
    help="Distance from origin of extraction center (in pixels). " + \
        "Overwritten if mask is provided")
parser.add_argument(
    '--vec', nargs='+', default=None,
    help='Direction from map origin of box center')
parser.add_argument(
    '--extraction_center', nargs='+', default=None,
    help='Locatoin of center pixel')
parser.add_argument(
    '--box_dims', nargs='+', default=None,
    help="Dimensions of box to extract. Overwritten if mask is provided. If " + \
        "None, uses dimensions of nonzero elements.")
parser.add_argument(
    '--max_dim', action='store_true', default=False,
    help='Optionally use masks maximum dimension for cubic box size.')
parser.add_argument(
    '--only_masked', action='store_true', default=False,
    help='extract only masked region values')
parser.add_argument(
    '--keep_dims', action='store_true', default=False,
    help='keep the original mask box dimensions.')
parser.add_argument(
    '--overwrite', action='store_true', default=True,
    help='Will overwrite filenames')

def entry_point():

    # parse arguments
    args = parser.parse_args()

    # determine if information is to be grabbed from mask
    if args.mask is None:
        dist = args.distance_from_origin
        v0 = args.vec
        box_dims = args.box_dims
        extraction_center = args.extraction_center
    else:
        # obtain distance and vector from origin to center of asymmetric unit
        dist, v0 = maps.mask_distance_vec(args.mask)
        box_dims = maps.find_box_dims(args.mask)
        extraction_center = None

        # optionally make box cubic with largest box dimension
        if args.max_dim:
            box_dims = box_dims.max()

    recenter = True
    if args.keep_dims:
        box_dims = maps.current_box_dims(args.input_name)
        recenter = False

    # optionally extract only masked region
    if args.only_masked:
        mask_filename = args.mask
    else:
        mask_filename = None

    # extract map
    maps.extract_from_map(
        map_filename=args.input_name, distance_from_origin=dist,
        vec=v0, box_dims=box_dims, extraction_center=extraction_center,
        output_name=args.output_name, mask_filename=mask_filename,
        recenter=recenter, overwrite=args.overwrite)


if __name__=='__main__':
    entry_point()
