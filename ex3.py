import argparse
import configparser
import csv
import datetime
import itertools
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt


DATETIME_FORMAT_DEFAULT = "%d/%m/%Y  %H:%M:%S.%f"


class ProgAbort(Exception):
    pass


class FatalError(Exception):
    def __init__(self, msg, code=2):
        self.msg = msg
        self.code = code
        super().__init__()


def convert_sensor_time_data(t_data, fmt):
    """
    Convert sensor data string datetimes to seconds from
    our earliest datetime
    """

    def dt_to_seconds(datetime_0):
        def convert(dt_str):
            dt = datetime.datetime.strptime(dt_str, fmt)
            delta = dt - datetime_0
            return delta.seconds + delta.microseconds/10**6
        return convert

    datetime_0 = min(datetime.datetime.strptime(t[0], fmt) for t in t_data)
    convert = dt_to_seconds(datetime_0)

    return [
        np.fromiter((convert(dt) for dt in t_range), float)
        for t_range in t_data
    ]


def skip_comments(fh):
    """
    skip any comment line
    necessary helper as default csv reader includes these
    """
    for row in fh:
        raw = row.split('#')[0].strip()
        if raw:
            yield raw


def parse_config_file(path):
    """
    Parse the config file (ini format)
    """
    config = configparser.ConfigParser()
    config.read(str(path))
    cf = {}
    for name, section in config.items():
        if name != 'DEFAULT':
            data_file = Path(section['sensor_data'])
            if not data_file.is_absolute():
                data_file = path.parent / data_file
            data_file = data_file.resolve()

            section = config[name]
            cf[name] = dict(
                sensor_data = data_file,
                resistance = float(section['resistance']),
                capacitance = float(section['capacitance']),
                signal_col = int(section.get('signal_col', 0)),
                datetime_format_str = section.get(
                    'datetime_format_str', DATETIME_FORMAT_DEFAULT
                )
            )
    return cf


def sensor_file_to_data(path):
    """
    Pull data in from csv file. Data cols assumed to arranged 
    in format x_0, y_0, x_1, y_1,... x_N, y_N
    """
    x_data,y_data = [],[]
    with path.open('r') as fh:
        first_row = True
        for row in csv.reader(skip_comments(fh)):
            if first_row:
                entries = len(row)
                if entries % 2 != 0:
                    raise ValueError('csv must have even number of columns')
                else:
                    data_sets = entries // 2
                    x_data = [[] for _ in range(data_sets)]
                    y_data = [[] for _ in range(data_sets)]

                first_row = False

            xy_cycle = itertools.cycle((iter(x_data), iter(y_data)))
            for value in row:
                next(next(xy_cycle)).append(value)

    return x_data, y_data


def pythonize_data(ranges, native_type):
    """
    convert raw data from csv data to native python data types
    """
    return [
        np.fromiter((native_type(x) for x in r), native_type)
        for r in ranges
    ]


def _tightness(data_range, threshold, buckets):
    """
    establish what distance between data values
    is nominal
    """
    diff = np.fromiter(
        (
            abs(data_range[i + 1] - data_range[i])
            for i in range(0, len(data_range) - 1)
        ),
        float
    )
    max_diff = max(diff)
    min_diff = min(diff)
    delta = max_diff - min_diff
    step = delta/buckets

    buckets = [0 for n in range(buckets + 1)]
    for value in diff:
        buckets[math.floor(value / step)] += 1

    normalised_buckets = [n / len(data_range) for n in buckets]
    marker = 0
    for idx, n in enumerate(normalised_buckets):
        if n > threshold:
            marker = idx

    return (marker + 1) * step


def _cluster_data(data_range, tightness, threshold):
    """
    cluster data in data_range using distance between
    value and cluster avg value. Discard clusters
    of negligible size
    """
    safe_cluster_len = threshold * len(data_range)

    clusters = []
    cluster = [(0, data_range[0])]
    total = data_range[0]

    for idx, value in enumerate(data_range[1:]):
        if abs(total / len(cluster) - cluster[-1][1]) < tightness:
            cluster.append((idx, value))
            total += value
        else:
            if len(cluster) >= safe_cluster_len:
                clusters.append(cluster)
            cluster = [(idx, value)]
            total = value

    return clusters


def _aggregate_clusters(clusters):
    """
    Aggregate cluster data - return extremum,
    summed values, and number of points of cluster
    """
    return [
        {
            'cluster': c,
            'start': c[0][0],
            'end': c[-1][0],
            'total': sum(x[1] for x in c),
            'count': len(c),
            'max': max(x[1] for x in c),
            'min': min(x[1] for x in c),
        }
        for c in clusters
    ]


def _add_peak_and_trough_data(data, switch_diff):
    # first sweep
    counter, is_peak = 0, None

    def _update(x, is_peak):
        new_x = {'is_peak': is_peak}
        new_x.update(x)
        return new_x

    for idx, ag1, ag2 in zip(itertools.count(0), data[:-1], data[1:]):
        avg1 = ag1['total']/ag1['count']
        avg2 = ag2['total']/ag2['count']
        if abs(avg1 - avg2) > switch_diff:
            counter = idx
            is_peak = avg1 > avg2
            break

    new_data = [_update(x, is_peak) for x in data[:counter+1]]

    # second sweep
    for ag1, ag2 in zip(data[counter:-1], data[counter+1:]):
        avg1 = ag1['total']/ag1['count']
        avg2 = ag2['total']/ag2['count']
        if abs(avg1 - avg2) > switch_diff:
            is_peak = not is_peak
        new_data.append(_update(ag2, is_peak))

    return new_data


def _aggregate_peaks_and_troughs(data):
    """
    Take aggregated data and aggregate it further
    by merging all aggregate data for clusters
    on the same peak or trough
    """
    pntr = None
    new_data = []

    for agg in data:
        if pntr is not None and agg['is_peak'] == pntr['is_peak']:
            pntr['end'] = agg['end']
            pntr['total'] += agg['total']
            pntr['count'] += agg['count']
            pntr['max'] = max(pntr['max'], agg['max'])
            pntr['min'] = max(pntr['min'], agg['min'])
            pntr['cluster'].extend(agg['cluster'])
        else:
            new_data.append(agg.copy())
            pntr = agg

    return new_data


def _aggregate_min_max(data):
    """
    Reduce aggregate by removing max, min values
    and replacing them with a single spread value
    """
    def _helper(x):
        new = x.copy()
        new['spread'] = new['max'] - new['min']
        del new['min']
        del new['max']
        return new

    return [_helper(x) for x in data]


def _aggregate_average(data):
    """
    Reduce aggregate by removing total and count values
    and replacing them with a single average value
    """
    def _helper(x):
        new = x.copy()
        new['avg'] = new['total'] / new['count']
        del new['total']
        del new['count']
        return new

    return [_helper(x) for x in data]



def _combine_with_second_data_range(data, second_range):
    """
    Combined aggregated data with a secondary data
    range and replace the start & end values with
    the corresponding values from this range
    """

    def _helper(x):
        new = x.copy()
        new['start'] = second_range[x['start']]
        new['end'] = second_range[x['end']]
        return new
    return [_helper(x) for x in data]


def _waveform_data(data):
    peak_values = np.fromiter(
        (v[1] for x in data for v in x['cluster'] if x['is_peak']),
        float
    )
    trough_values = np.fromiter(
        (v[1] for x in data for v in x['cluster'] if not x['is_peak']),
        float
    )

    peak_time_data = np.fromiter(
        (x['end'] - x['start'] for x in data if x['is_peak']),
        float
    )
    trough_time_data = np.fromiter(
        (x['end'] - x['start'] for x in data if not x['is_peak']),
        float
    )

    peak_time_avg = np.mean(peak_time_data)
    trough_time_avg = np.mean(trough_time_data)
    period_avg = peak_time_avg + trough_time_avg

    return dict(
        peak_avg = np.mean(peak_values),
        trough_avg = np.mean(trough_values),
        peak_time_avg = peak_time_avg,
        trough_time_avg = trough_time_avg,
        period_avg = period_avg,
        freq_avg = 1 / period_avg,
        leading_peak = data[0]['is_peak'],
        pulse_data = data
    )


def analyse_signal(t_range, v_range):
    """
    Analyse a square signal and return a information about the
    waveform
    """
    tightness = _tightness(v_range, 0.005, 1000)
    clusters = _cluster_data(v_range, tightness, 0.005)

    max_value = max(x[1] for c in clusters for x in c)
    min_value = min(x[1] for c in clusters for x in c)
    switch_diff = (max_value - min_value) * 0.9

    agg =_aggregate_clusters(clusters)
    agg = _add_peak_and_trough_data(agg, switch_diff)
    agg = _aggregate_peaks_and_troughs(agg)
    agg = _aggregate_average(agg)
    agg = _aggregate_min_max(agg)
    combined = _combine_with_second_data_range(agg, t_range)
    data = _waveform_data(combined)
    return data


def rc_output_sim(time_constant, input_data, threshold, samples=50):
    """
    Model the output of an RC filter when it's supplied with a square
    wave input

    :param float time_constant:  Time constant of the RC circuit
    :param dict input_data:      Data from the analysis of the square wave
                                 input
    :param float threshold:      The model starts in an unsteady state. After
                                 sufficient iterations the model will stablise
                                 and the amplitude of output waveform will have
                                 little variance. Threshold controls how much
                                 variance controls is considered allowable.
                                 Should be between 0 and 1, anything above
                                 0.01 would be considered a lot of variance.
    :param int samples:          The number of sample points to use per
                                 half period

    """
    def _initial_charge(in_data):
        if in_data['leading_peak']:
            v0, t = in_data['peak_avg'], in_data['peak_time_avg']
        else:
            v0, t = in_data['trough_avg'], in_data['trough_time_avg']
        return v0 * (1 - np.exp(-t/time_constant))

    def _unsteady_state(in_data, extremum, threshold):
        high = -in_data['peak_time_avg']/time_constant
        low = -in_data['trough_time_avg']/time_constant
        smax = in_data['peak_avg']
        smin = in_data['trough_avg']
        if in_data['leading_peak']:
            v_max = extremum
            v_min = None
        else:
            v_min = extremum
            v_max = (v_min - smax)*np.exp(high) + smax

        n = 0
        while True:
            n += 1
            v_min_new = (v_max - smin)*np.exp(low) + smin
            v_max_new = (v_min_new - smax)*np.exp(high) + smax

            if (
                v_min is not None and
                abs(v_max_new - v_max) / v_max_new < threshold and
                abs(v_min_new - v_min) / v_min_new < threshold
            ):
                break
            else:
                v_max, v_min = v_max_new, v_min_new
        print(n)

        return v_min, v_max

    def _steady_state_factory(v_min, v_max, samples):
        def rising(sig, t0, t1):
            const = v_min - sig
            t_values = np.linspace(0, t1 - t0, samples)
            return t_values + t0, const*np.exp(-t_values/time_constant) + sig

        def falling(sig, t0, t1):
            const = v_max - sig
            t_values = np.linspace(0, t1 - t0, samples)
            return t_values + t0, const*np.exp(-t_values/time_constant) + sig
        return rising, falling

    v_extremum = _initial_charge(input_data)
    v_min, v_max = _unsteady_state(input_data, v_extremum, threshold)

    ss_funcs = itertools.cycle(_steady_state_factory(v_min, v_max, samples))
    if not input_data['leading_peak']:
        next(ss_funcs)

    v_data, t_data = [], []
    for info in input_data['pulse_data']:
        f = next(ss_funcs)
        t_range, v_range = f(info['avg'], info['start'], info['end'])
        v_data.extend(v_range)
        t_data.extend(t_range)

    return {'x':np.array(t_data), 'y':np.array(v_data)}


def plot(plot_data, output_dir, suffix, prefix, ext):
    """
    Create plots
    """
    def _make_plot(data, ax):
        title, data_ranges = data['title'], data['ranges']

        colour_wheel =['#329932', '#ff4444', 'b']
        dash_styles = [[3,1], [2,1,10,1], [4, 1, 1, 1, 1, 1]]
        c_iter = itertools.cycle(colour_wheel)
        d_iter = itertools.cycle(dash_styles)

        xmin = min(p['data']['x'][0] for p in data_ranges)
        xmax = max(p['data']['x'][-1] for p in data_ranges)

        ymin = np.floor(min(y for r in data_ranges for y in r['data']['y']))
        ymax = np.ceil(max(y for r in data_ranges for y in r['data']['y']))

        for r in data_ranges:
            ax.plot(
                r['data']['x'],
                r['data']['y'],
                label=r['label'],
                color=next(c_iter),
                dashes=next(d_iter)
            )

        ax.set_title(title)
        ax.legend(loc='upper right')

        ax.set_xlim(left=xmin, right=xmax)
        ax.set_ylim(bottom=ymin, top=ymax)

        ax.grid(which='major', color='#999999')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [V]')

    def _output_plot(path):
        if path:
            if path.is_file():
                sys.stdout.write(
                    'Do you wish to overwrite `%s`? Type lowercase YES\n'
                    % path
                )
                buffer = input('> ').strip()
                if buffer != 'yes':
                    raise ProgAbort()
            elif path.exists():
                raise FatalError(
                    'cannot write graph to path `%s` - '
                    'path exists and is not a regular file'
                    % path
                )

            plt.savefig(str(path), format="svg")

    def _build_path(src, dir, prefix, suffix, ext):
        name = '%s%s%s.%s' % (suffix, src.stem, prefix, ext)
        return dir / name

    def _aggregate_plots(plot_data):
        fig, axs = plt.subplots(ncols=1, nrows=len(plot_data),figsize=(7,9))
        for data, ax in zip(plot_data, axs):
            _make_plot(data, ax)

    if output_dir is None:
        _aggregate_plots(plot_data)
        plt.tight_layout()
        plt.show()
    else:
        for data in plot_data:
            fig, ax = plt.subplots(figsize=(7,3))
            _make_plot(data, ax)
            plt.tight_layout()
            out_path = _build_path(
                data['src'], output_dir, prefix, suffix, ext
            )
            _output_plot(out_path)


def get_plot_data(cf):
    """
    Use this experiments config to generate data
    for a plot
    """
    time_constant = cf['resistance'] * cf['capacitance']

    t_data_raw, v_data_raw = sensor_file_to_data(cf['sensor_data'])
    v_data = pythonize_data(v_data_raw, float)
    t_data = convert_sensor_time_data(
        t_data_raw, cf['datetime_format_str']
    )
    t_data_raw, v_data_raw = None, None

    output_idx = 1 if cf['signal_col'] is 0 else 1

    input_signal = {
        'x':t_data[cf['signal_col']],
        'y':v_data[cf['signal_col']]
    }
    output_signal = {
        'x':t_data[output_idx],
        'y':v_data[output_idx]
    }
    signal_info = analyse_signal(input_signal['x'], input_signal['y'])
    output_sim = rc_output_sim(
        time_constant,
        signal_info,
        threshold=0.00000000001
    )

    t = "RC circuit input/output ($\\tau$ = %.4fs, $f$=%.4fHz)" \
        % (time_constant, signal_info['freq_avg'])

    return {
        'title': t,
        'src': cf['sensor_data'],
        'ranges': [
            {'label':'Input', 'data':input_signal},
            {'label':'Output', 'data':output_signal},
            {'label':'Sim.', 'data':output_sim}
        ]
    }


def main(cmd, argv):
    parser = argparse.ArgumentParser(prog=cmd)
    parser.add_argument(
        '-s', '--suffix',
        help='Output suffix',
        default='',
    )
    parser.add_argument(
        '-p', '--prefix',
        help='Output prefix',
        default='',
    )
    parser.add_argument(
        '-e', '--ext',
        help='Overide file extension',
        default='svg'
    )
    parser.add_argument(
        'config_file',
        metavar='CONFIG_DATA_FILE',
        help='Path to config file'
    )
    parser.add_argument(
        'out_dir',
        metavar='OUTPUT_DIR',
        help='Path to output dir',
        nargs='?'
    )

    opts = parser.parse_args(argv)

    try:
        if opts.out_dir is not None:
            opts.out_dir = Path(opts.out_dir).resolve()

        config = parse_config_file(Path(opts.config_file))
        plot_data = []
        for n, c in config.items():
            pd = get_plot_data(c)
            pd['name'] = n
            plot_data.append(pd)

        plot(plot_data, opts.out_dir, opts.prefix, opts.suffix, opts.ext)
        return 0
    except ProgAbort:
        sys.stderr.write('Aborting\n')
        return 1
    except FatalError as excp:
        sys.stderr.write('ERROR: %s\n' %excp.msg)
        return excp.code

sys.exit(main(sys.argv[0], sys.argv[1:]))
