import argparse
import itertools
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class ProgAbort(Exception):
    pass


class FatalError(Exception):
    def __init__(self, msg, code=2):
        self.msg = msg
        self.code = code
        super().__init__()


def model_input(cycles, shift, duty=.5):
    t = np.linspace(0, 2*cycles, 1000, endpoint=False)
    return t, signal.square(t*np.pi, duty=duty) + shift


def model_output(init_cycles, cycles, shift, tc_frac):
    """
    Model the voltage of the capacitor
    based on the signal data and circuit configuration
    """
    v_pos = shift + 1
    v_neg = shift - 1

    def initial_charge(t):
        return v_pos * (1 - np.exp(-t))

    def rising(t, v_min):
        return (v_min - v_pos)*np.exp(-t) + v_pos

    def falling(t, v_max):
        return (v_max - v_neg)*np.exp(-t) + v_neg

    time_constant = 2 * tc_frac
    t = np.linspace(0, 1, 200, endpoint=True)
    norm_t = t / time_constant

    v_data, t_data = [], []
    v_init = initial_charge(norm_t)
    is_rising, v_max, v_min = False, v_init[-1], None

    if init_cycles == 0:
        v_data.extend(v_init)
        t_data.extend(t)
        r = range(1, 2*cycles)
    else:
        r = range(0, 2*cycles)
        for n in range(1, 2*init_cycles):
            if is_rising:
                v_max = rising(norm_t[-1], v_min)
            else:
                v_min = falling(norm_t[-1], v_max)
            is_rising = not is_rising


    for n in r:
        if is_rising:
            v_data.extend(rising(norm_t, v_min))
            v_max = v_data[-1]
        else:
            v_data.extend(falling(norm_t, v_max))
            v_min = v_data[-1]
        t_data.extend(n + t)
        is_rising = not is_rising

    return np.array(t_data), np.array(v_data)


def plot(init_cycles, cycles, shift, in_data, out_data, output_path=None):
    """
    Create plots
    """
    colour_wheel =['#329932',
            '#ff4444',
            'b',
            '#6a3d9a',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#67001f',
            '#b2182b',
            '#d6604d',
            '#f4a582',
            '#fddbc7',
            '#f7f7f7',
            '#d1e5f0',
            '#92c5de',
            '#4393c3',
            '#2166ac',
            '#053061']
    dash_styles = [[3,1],
            [1000,1],
            [2,1,10,1],
            [4, 1, 1, 1, 1, 1]]


    c_iter = itertools.cycle(colour_wheel)
    d_iter = itertools.cycle(dash_styles)

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(*in_data, label='Input', c=next(c_iter), dashes=next(d_iter))

    for tc_frac, data in out_data:
        x, y = data
        ax.plot(
            x, y,
            label='Output ($\\tau = %.3fT$)' % tc_frac,
            c=next(c_iter),
            dashes=next(d_iter)
        )

    ax.legend(loc='lower right')

    ax.set_xticklabels(list(range(0, cycles + 1)))
    ax.set_xticks(list(range(0, 2*cycles + 1, 2)))

    ax.set_xlim(left=0, right=cycles*2)
    ax.set_ylim(bottom=min(0,shift-1.5), top=max(0,shift+1.5))

    ax.grid(which='major', color='#999999')
    ax.set_xlabel('Time [$T$]')
    ax.set_ylabel('Voltage [$V_p$]')

    plt.tight_layout()

    if output_path:
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
    else:
        plt.show()


def main(cmd, argv):
    parser = argparse.ArgumentParser(prog=cmd)
    parser.add_argument(
        '-c', '--cycles',
        metavar='NCYCLES',
        help='Number of cycles',
        default=5,
        type=int
    )
    parser.add_argument(
        '-i', '--initial-cycles',
        metavar='NCYCLES',
        help='Number of initial cycles',
        default=0,
        type=int,
        dest='init_cycles'
    )
    parser.add_argument(
        '-s', '--shift',
        metavar='SHIFT',
        help='Voltage shift',
        default=0,
        type=int
    )
    parser.add_argument(
        'tc_frac_list',
        metavar='TC_RATIONS',
        help='Time constant to period rations',
    )
    parser.add_argument(
        'out_file',
        metavar='OUTPUT_FILE',
        help='Path to output file',
        nargs='?'
    )

    opts = parser.parse_args(argv)

    if opts.out_file is not None:
        opts.out_file = Path(opts.out_file)

    tc_fracs = sorted(float(x) for x in opts.tc_frac_list.split(','))

    try:
        sig_data = model_input(opts.cycles, opts.shift)
        model_data = [
            (f, model_output(opts.init_cycles, opts.cycles, opts.shift, f))
            for f in tc_fracs
        ]

        plot(opts.init_cycles, opts.cycles, opts.shift, sig_data, model_data, opts.out_file)
        return 0
    except ProgAbort:
        sys.stderr.write('Aborting\n')
        return 1
    except FatalError as excp:
        sys.stderr.write('ERROR: %s\n' %excp.msg)
        return excp.code

sys.exit(main(sys.argv[0], sys.argv[1:]))
