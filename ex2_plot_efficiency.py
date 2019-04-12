import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path


class ProgAbort(Exception):
    pass


class FatalError(Exception):
    def __init__(self, msg, code=2):
        self.msg = msg
        self.code = code
        super().__init__()


def plot(x_max, y_max, samples=50, output_path=None):
    fig, ax = plt.subplots(figsize=(3, 3))

    x = np.linspace(0, x_max, samples)
    y = x / (1 + x)

    ax.plot(x, y, c='g')

    ax.set_ylabel(r'$P_L/P_S$')
    ax.set_xlabel(r'$R_L/R_S$')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.grid()

    fig.tight_layout()
    if output_path is None:
        plt.show()
    else:
        if output_path.is_file():
            sys.stdout.write(
                'Do you wish to overwrite `%s`? Type lowercase YES\n'
                % output_path
            )
            buffer = input('> ').strip()
            if buffer != 'yes':
                raise ProgAbort()
        elif output_path.exists():
            raise FatalError(
                'cannot write graph to path `%s` - '
                'path exists and is not a regular file'
                % output_path
            )

        plt.savefig(str(output_path), format="svg")



def main(cmd, argv):
    parser = argparse.ArgumentParser(prog=cmd)
    parser.add_argument(
        'x_max',
        metavar='X_MAX',
        help='Maximum value of x',
        type=float
    )
    parser.add_argument(
        'y_max',
        metavar='Y_MAX',
        help='Maximum value of y',
        type=float
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT',
        help='Path to output file',
        type=Path,
        nargs='?'
    )

    opts = parser.parse_args(argv)

    try:
        plot(opts.x_max, opts.y_max, output_path=opts.output_path)
        return 0
    except ProgAbort:
        sys.stderr.write('Aborting\n')
        return 1
    except FatalError as excp:
        sys.stderr.write('ERROR: %s\n' %excp.msg)
        return excp.code

sys.exit(main(sys.argv[0], sys.argv[1:]))
