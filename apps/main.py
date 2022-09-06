import logging
import sys
import argparse

def identify_app(argv):

    parser = argparse.ArgumentParser(
        prog='EMProcess',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Main entry point for EMProcess apps.")

    parser.add_argument(
        "appname",
        choices={'create_subparticles', 'extract', 'formatting', 'mask', 'rebox', 'star'},
        help="Name of the application.")

    parser.add_argument(
        "appargs", nargs="*",
        help="Subsequent arguments to the app (add subcommand for more).")

#    helpstack = []
#    for h in ['--help', '-h']:
#        while h in argv and argv.index(h) != 1:
#            argv.remove(h)
#            helpstack.append(h)

    args = parser.parse_args([argv[1]])

    if args.appname == 'create_subparticles':
        from EMProcess.apps.create_subparticles import main
    elif args.appname == 'extract':
        from EMProcess.apps.extract import main
    elif args.appname == 'formatting':
        from EMProcess.apps.formatting import main
    elif args.appname == 'mask':
        from EMProcess.apps.mask import main
    elif args.appname == 'rebox':
        from EMProcess.apps.rebox import main
    elif args.appname == 'star':
        from EMProcess.apps.star import main

    args.main = main
#    args.appargs.extend(helpstack)

    return args


def main(argv=None):

    args = identify_app(argv)
    print(argv)

    try:
        if len(argv[2:]) == 0:
            args.main(['-h'])
        else:
            args.main(argv[2:])
    except Exception as e:
        message = ("An unexpected error has occurred")
        print(message, file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
