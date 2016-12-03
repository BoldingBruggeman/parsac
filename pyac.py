#!/usr/bin/env python

import argparse

import client.run
import plot
import plotbest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PYthon AutoCalibration - pyac')
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run', help='Run the auto-calibration')
    client.run.configure_argument_parser(parser_run)
    parser_run.set_defaults(func=client.run.main)

    parser_plot = subparsers.add_parser('plot', help='Plot maximum likelihood for the various parameters being estimated')
    plot.configure_argument_parser(parser_plot)
    parser_plot.set_defaults(func=plot.main)

    parser_plotbest = subparsers.add_parser('plotbest', help='Plot statistical results - default for the best parameter set')
    plotbest.configure_argument_parser(parser_plotbest)
    parser_plotbest.set_defaults(func=plotbest.main)

    args = parser.parse_args()
    args.func(args)

