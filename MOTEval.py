# --------------------------------------------------------
# Compute metrics for trackers using ground-truth data
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200321
# Based on motmetrics (https://github.com/cheind/py-motmetrics/)
# --------------------------------------------------------

import argparse
from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path

import motmetrics as mm
from panda_utils import generate_mot_anno


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using ground-truth data.

Files
-----
All result files have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/tracks.json
    <GT_ROOT>/<SEQUENCE_1>/seqinfo.json
    <GT_ROOT>/<SEQUENCE_2>/tracks.json
    <GT_ROOT>/<SEQUENCE_2>/seqinfo.json
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('groundtruths', type=str, help='Directory containing ground truth files.')
    parser.add_argument('tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('--transfered', type=str, help='Directory containing transfered gt files', default='transfered')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use for matching between frames.')
    parser.add_argument('--id_solver', type=str, help='LAP solver to use for ID metrics. Defaults to --solver.')
    parser.add_argument('--exclude_id', dest='exclude_id', default=False, action='store_true',
                        help='Disable ID metrics')
    return parser.parse_args()


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing %s...', k)
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for %s, skipping.', k)

    return accs, names


def main():
    # pylint: disable=missing-function-docstring
    args = parse_args()

    # transfer ground truth file from PANDA format to MOTChallenge format
    generate_mot_anno(args.groundtruths, args.transfered)

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    gtfiles = glob.glob(os.path.join(args.transfered, '*.txt'))
    tsfiles = [f for f in glob.glob(os.path.join(args.tests, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    gt = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    metrics = list(mm.metrics.motchallenge_metrics)
    if args.exclude_id:
        metrics = [x for x in metrics if not x.startswith('id')]

    logging.info('Running metrics')

    if args.id_solver:
        mm.lap.default_solver = args.id_solver
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')

    '''we use MOTA, MOTP, IDF1, MT, ML, FP, FN, IDs, Frag to evaluate'''
    MOTA = summary['mota']['OVERALL']
    MOTP = 1 - summary['motp']['OVERALL']
    IDF1 = summary['idf1']['OVERALL']
    MT = summary['mostly_tracked']['OVERALL']
    ML = summary['mostly_lost']['OVERALL']
    FP = summary['num_false_positives']['OVERALL']
    FN = summary['num_misses']['OVERALL']
    IDs = summary['num_switches']['OVERALL']
    Frag = summary['num_fragmentations']['OVERALL']
    print(MOTA, MOTP, IDF1, MT, ML, FP, FN, IDs, Frag)


if __name__ == '__main__':
    main()
