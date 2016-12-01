#!/usr/bin/env python

import argparse
import sys

class pyac_cmds(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='PYthon AutoCalibration - pyac',
            usage='''pyac <command> [<args>]

The pyac commands are:
   run:      Run the auto calibration
   plot:     Plot maximum likelyhood for the various parameters being estimated
   plotbest: Plot statistical results - default for the best parameter set
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print 'Unrecognized command'
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def run(self):
        parser = argparse.ArgumentParser(description='Run the auto calibration')
        parser.add_argument('xmlfile',                 type=str, help='XML formatted configuration file')
        parser.add_argument('-m', '--method',          type=str, choices=('DE', 'fmin', 'galileo'), help='Optimization method: DE = Differential Evolution genetic algorithm, fmin = Nelder-Mead simplex, galileo = galileo genetic algorithm')
        parser.add_argument('-t', '--transport',       type=str, choices=('http', 'mysql'), help='Transport to use for server communication: http or mysql')
        parser.add_argument('-r', '--reportfrequency', type=int,    help='Time between result reports (seconds).')
        parser.add_argument('-i', '--interactive',     action='store_true', help='Whether to allow for user interaction (input from stdin) when making decisions')
        parser.add_argument('-n', '--ncpus',           type=int,    help='Number of CPUs to use (only for Differential Evolution genetic algorithm).')
        parser.add_argument('--tempdir',               type=str, help='Temporary directory for GOTM setups.')
        parser.add_argument('--ppservers',             type=str, help='Comma-separated list of names/IPs of Parallel Python servers to run on (only for Differential Evolution genetic algorithm).')
        parser.set_defaults(method='DE', transport=None, interactive=False, ncpus=None, ppservers=None, reportfrequency=None, tempdir=None, scenarios='.')

        args = parser.parse_args(sys.argv[2:])
        print 'Call the run module'

    def plot(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('xmlfile',       type=str, help='XML formatted configuration file')
        parser.add_argument('-r', '--range', type=float, help='Lower boundary for relative ln likelihood (always < 0)')
        parser.add_argument('--bincount', type=int, help='Target number of segments per ln likelihood marginal')
        parser.add_argument('-g', '--groupby', type=str, choices=('source', 'run'), help='What identifier to group the results by, i.e., "source" or "run".')
        parser.add_argument('-o', '--orderby', type=str, choices=('count', 'lnl'), help='What property to order the result groups by, i.e., "count" or "lnl".')
        parser.add_argument('--maxcount', type=int, help='Maximum number of series to plot')
        parser.add_argument('--constraint', type=str, action='append', nargs=3, help='Constraint on parameter (parameter name, minimum, maximum)', dest='constraints')
        parser.add_argument('-l', '--limit', type=int, help='Maximum number of results to read')
        parser.add_argument('--run', type=int, help='Run number')
        parser.add_argument('-u', '--update', action='store_true', help='Keep running and updating the figure with new results until the user quits with Ctrl-C')
        parser.set_defaults(range=None, bincount=25, orderby='count', maxcount=None, groupby='run', constraints=[], limit=-1, run=None, update=False)
        args = parser.parse_args(sys.argv[2:])
        print 'Call the plot module'

    def plotbest(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('xmlfile',       type=str, help='XML formatted configuration file')
        parser.add_argument('-r', '--rank',  type=int,   help='Rank of the result to plot (default = 1, i.e., the very best result)')
        parser.add_argument('-d', '--depth', type=float, help='Depth range to show (> 0)')
        parser.add_argument('-g', '--grid',  action='store_true', help='Whether to grid the observations.')
        parser.add_argument('--savenc',      type=str, help='Path to copy NetCDF output file to.')
        parser.add_argument('--simulationdir',type=str, help='Directory to run simulation in.')
        parser.set_defaults(rank=1, depth=None, grid=False, savenc=None, simulationdir=None)
        args = parser.parse_args(sys.argv[2:])
        print 'Call the plotbest module'

if __name__ == '__main__':
    pyac_cmds()

