import argparse
import numpy as np
from EMProcess import maps

parser = argparse.ArgumentParser(
    description='Extracts boxed density from maps')
parser.add_argument(
    '--mask', type=str, default=None,
    help="Mask filename to use for determining box location and size.")
parser.add_argument(
    '--output_name', type=str, default=None,
    help="output filename to optionally store extracted mask.")
parser.add_argument(
    '--max_dim', type=bool, default=True,
    help="optionally use the maximum dimension for box size")
parser.add_argument(
    '--padding', type=int, default=0, help="number of pixels to extend box")
parser.add_argument(
    '--query', action='store_true', default=False,
    help="don't save box, only query box dimensions, vector, and center loc")
parser.add_argument(
    '--overwrite', action='store_true', default=False,
    help="optionally overwrite extraction mask")



def entry_point():

    args = parser.parse_args()
    
    mask_filename = args.mask
    
    dist,vec = maps.mask_distance_vec(mask_filename)
    box_dims = maps.find_box_dims(mask_filename)
    print("\n\nOriginal box dimensions: %s\n" % box_dims)
    if args.max_dim:
        max_dim = np.max(box_dims)
        box_dims = np.array([max_dim]*3, dtype=int)

    print('\n\ndistance: %0.3f\nvec: %s' % (dist, vec))
    print('box_dims: %s + %d = %s' % (box_dims, args.padding, box_dims+args.padding))
    print('center location: %s\n' % (dist*vec))

    if args.output_name is not None and not args.query:
        new_box_dims = box_dims + args.padding
        print('\n saving new file as %s with new box %s\n...\n' % (args.output_name, new_box_dims))
        maps.extract_from_map(
            map_filename=mask_filename, distance_from_origin=dist,
            vec=vec, box_dims=new_box_dims, output_name=args.output_name,
            overwrite=args.overwrite)



if __name__=='__main__':
    entry_point()
