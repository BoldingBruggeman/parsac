#!/usr/bin/env python

import argparse

import acpy.run
import plot
import plotbest
import animate_2d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoCalibration Python - acpy')
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run', help='Run the auto-calibration')
    acpy.run.configure_argument_parser(parser_run)
    parser_run.set_defaults(func=acpy.run.main)

    parser_plot = subparsers.add_parser('plot', help='Plot maximum likelihood for the various parameters being estimated')
    plot.configure_argument_parser(parser_plot)
    parser_plot.set_defaults(func=plot.main)

    parser_plotbest = subparsers.add_parser('plotbest', help='Plot statistical results - default for the best parameter set')
    plotbest.configure_argument_parser(parser_plotbest)
    parser_plotbest.set_defaults(func=plotbest.main)

    parser_animate_2d = subparsers.add_parser('animate_2d', help='Generate sequence of X-Y plots showing parameter variations - as .pngs')
    animate_2d.configure_argument_parser(parser_animate_2d)
    parser_animate_2d.set_defaults(func=animate_2d.main)

    args = parser.parse_args()
    args.func(args)

