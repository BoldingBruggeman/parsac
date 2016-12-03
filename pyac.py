#!/usr/bin/env python

import argparse
import sys

import client.run
import plot
import plotbest

class pyac_cmds(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='PYthon AutoCalibration - pyac')
        subparsers = parser.add_subparsers()

        parser_run = subparsers.add_parser('run', help='Run the auto calibration')
        client.run.configure_argument_parser(parser_run)
        parser_run.set_defaults(func=client.run.main)

        parser_plot = subparsers.add_parser('plot', help='Plot maximum likelihood for the various parameters being estimated')
        client.run.configure_argument_parser(parser_plot)
        parser_run.set_defaults(func=plot.main)

        parser_plotbest = subparsers.add_parser('plotbest', help='Plot statistical results - default for the best parameter set')
        client.run.configure_argument_parser(parser_plotbest)
        parser_run.set_defaults(func=plotbest.main)

        args = parser.parse_args()
        args.func(args)

if __name__ == '__main__':
    pyac_cmds()

