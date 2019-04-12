import csv
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


def get_data(path):
    with path.open('r') as fh:
        r_frac, r_frac_err, p_frac, p_frac_err = [0], [0], [0], [0]
        reader = csv.reader(fh)
        first = True
        for row in reader:
            if not first:
                r_frac.append(float(row[8]))
                r_frac_err.append(float(row[9]))
                p_frac.append(float(row[10]))
                p_frac_err.append(float(row[11]))
            first=False
        return {
            'resistance_frac': r_frac,
            'resistance_frac_err': r_frac_err,
            'power_frac': p_frac,
            'power_frac_err':p_frac_err
        }


def plot(data, x_max=None, y_max=None, samples=50, output_path=None):
    fig, ax = plt.subplots(figsize=(3, 3))

    if x_max is None:
        x_max = max(data['resistance_frac'])

    if y_max is None:
        y_max = max(data['power_frac']) + .1

    x = np.linspace(0, x_max, samples)
    y = 4*x / (1 + x)**2

    ax.errorbar(
        data['resistance_frac'],
        data['power_frac'],
        yerr=data['power_frac_err'],
        fmt='x',
        c='#BB0000',
        capsize=3
    )
    ax.plot(x, y, c='b')

    ax.set_ylabel(r'$P_L/P_{max}$')
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
        '-x', '--xmax',
        metavar='XMAX',
        help='Override maximum x',
        type=float
    )
    parser.add_argument(
        '-y', '--ymax',
        metavar='YMAX',
        help='Override maximum y',
        type=float
    )
    parser.add_argument(
        'data_path',
        metavar='DATA_FILE',
        help='Path to data file',
        type=Path,
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
        data = get_data(opts.data_path)
        plot(data, opts.xmax, opts.ymax, output_path=opts.output_path)
        return 0
    except ProgAbort:
        sys.stderr.write('Aborting\n')
        return 1
    except FatalError as excp:
        sys.stderr.write('ERROR: %s\n' %excp.msg)
        return excp.code

sys.exit(main(sys.argv[0], sys.argv[1:]))
