import argparse
import bath


def tuple(s):
    try:
        x, y = map(int, s.split(','))
        return (x, y)
    except:
        raise argparse.ArgumentTypeError("Argument must be 2-element pair x,y")


def info(args):
    """
    Print system info
    """
    import sys
    print('Python version:')
    print(sys.version)


def main():
    parser = argparse.ArgumentParser(
        description='Bath: CodeNeuro Neuron Segmentation',
        argument_default=argparse.SUPPRESS,
    )
    subcommands = parser.add_subparsers()

    # bath info
    cmd = subcommands.add_parser('info', description='print system info')
    cmd.set_defaults(func=info)

    # bath nmf <datasets> <data_directory> [<output_directory>] [<nmf args>]
    cmd = subcommands.add_parser('nmf', description='Non-negative Matrix Factorization', argument_default=argparse.SUPPRESS)
    cmd.add_argument('datasets', help='list of datasets by logical name (00.00, 04.01.test, etc)',
                     nargs='+')
    cmd.add_argument('base_dir', help='directory where the datasets reside')
    cmd.add_argument('-o', '--output_dir', help='directory where results file will be written')
    cmd.add_argument('-g', '--gaussian_blur', help='Sigma for gaussian blur filter.  If 0, then no gaussian filtering will be done.', type=float)
    cmd.add_argument('-k', '--n_components', help='number of components to estimate per block', type=int)
    cmd.add_argument('-t', '--threshold', help='value for thresholding (higher means more thresholding)', type=float)
    cmd.add_argument('-m', '--overlap', help='value for determining whether to merge (higher means fewer merges)', type=float)
    cmd.add_argument('-c', '--chunk_size', help='process images in chunks of this size; should be a comma-separated pair.', type=tuple)
    cmd.add_argument('-p', '--padding', help='add this much padding to each chunk; should be a comma-separated pair', type=tuple)
    cmd.add_argument('-i', '--merge_iter', help='number of iterations to perform when merging regions', type=int)
    cmd.add_argument('-v', '--verbose', help='if set, print progress messages', action="store_true")
    cmd.set_defaults(func=bath.nmf.main)

    # Each subcommand gives an `args.func`.
    # Call that function and pass the rest of `args` as kwargs.
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
