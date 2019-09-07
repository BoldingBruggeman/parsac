#!/usr/bin/env python

from __future__ import print_function

# Import from standard Python library
import argparse

# Import personal custom stuff
from . import run
from . import result

def configure_argument_parser(parser):
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_run = subparsers.add_parser('run', help='Run the auto-calibration')
    run.configure_argument_parser(parser_run)
    parser_run.set_defaults(func=run.main)

    parser_plot = subparsers.add_parser('plot', help='Plot maximum likelihood for the various parameters being estimated')
    result.plot.configure_argument_parser(parser_plot)
    parser_plot.set_defaults(func=result.plot.main)

    parser_plotbest = subparsers.add_parser('plotbest', help='Plot statistical results - default for the best parameter set')
    result.plotbest.configure_argument_parser(parser_plotbest)
    parser_plotbest.set_defaults(func=result.plotbest.main)

    parser_animate_2d = subparsers.add_parser('animate_2d', help='Generate sequence of X-Y plots showing parameter variations - as .pngs')
    result.animate_2d.configure_argument_parser(parser_animate_2d)
    parser_animate_2d.set_defaults(func=result.animate_2d.main)

    parser_summary = subparsers.add_parser('summary', help='Summary of calibration')
    result.summary.configure_argument_parser(parser_summary)
    parser_summary.set_defaults(func=result.summary.main)

def main(args):
    print(args.subcommand)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
