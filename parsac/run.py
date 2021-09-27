#!/usr/bin/env python

# Import from standard Python library
from __future__ import print_function
import sys
import os.path
import argparse
import tempfile

# Import third party libraries
import numpy

# Import custom stuff
from . import service
from . import optimize
from . import job
from . import report

def configure_argument_parser(parser):
    parser.add_argument('xmlfile',                type=str, help='XML formatted configuration file')
    parser.add_argument('-m', '--method',         type=str, choices=('DE', 'fmin'), help='Optimization method: DE = Differential Evolution genetic algorithm, fmin = Nelder-Mead simplex (default: DE)', default='DE')
    parser.add_argument('-t', '--transport',      type=str, choices=('http', 'mysql'), help='Transport to use for server communication: http or mysql')
    parser.add_argument('-r', '--reportinterval', type=int, help='Time between result reports (seconds).')
    parser.add_argument('-i', '--interactive',    action='store_true', help='Whether to allow for user interaction (input from stdin) when making decisions')
    parser.add_argument('-q', '--quiet',          action='store_true', help='Suppress diagnostic messages')
    parser.add_argument('--tempdir',              type=str, help='Temporary directory to use for setups when using a parallelized optimization method (default: %s).' % tempfile.gettempdir())
    parser.add_argument('--maxfun',               type=int, help='Maximum number of function evaluations (or simulations) to perform (default: unlimited).')
    parser.add_argument('--maxiter',              type=int, help='Maximum number of iterations (Nelder-Mead simplex) or generations (Differential Evolution) to perform (default: unlimited).', default=sys.maxsize)

    de_options = parser.add_argument_group('Option specific to Differential Evolution (http://dx.doi.org/10.1023/A:1008202821328)')
    if service.parallel is not None:
        de_options.add_argument('-n', '--ncpus', type=int, help='Number of cores to use (default: use all available on the local machine).')
        de_options.add_argument('--ppservers',   type=str, help='Comma-separated list of names/IPs of Parallel Python servers to run on.')
        de_options.add_argument('--secret',      type=str, help='Parallel Python secret for authentication (only used in combination with ppservers argument).')
    de_options.add_argument('--F',  type=float, help='Scale factor for mutation (default: 0.5).', default=0.5)
    de_options.add_argument('--CR', type=float, help='Crossover probability (default: 0.9).', default=0.9)
    de_options.add_argument('--ftol', type=float, help='Difference in log likelihood that is acceptable for convergence. If the range in log likelihood values within the parameter population drops below this threshold, the optimization is terminated.')
    de_options.add_argument('--repeat', action='store_true', help='Start a new optimization whenever the last completes.')

def main(args):
    print('Reading configuration from %s...' % args.xmlfile)
    current_job = job.fromConfigurationFile(args.xmlfile, tempdir=args.tempdir, verbose=not args.quiet)

    with open(args.xmlfile) as f:
        xml = f.read()
    reporter = report.fromConfigurationFile(args.xmlfile, xml, allowedtransports=args.transport)

    # Configure result reporter
    reporter.interactive = args.interactive
    if args.reportinterval is not None:
        reporter.timebetweenreports = args.reportinterval

    opt = optimize.Optimizer(current_job, reportfunction=reporter.reportResult)

    try:
        while 1:
            logtransform = current_job.getParameterLogScale()
            if args.method == 'fmin':
                vals = opt.run(method=optimize.SIMPLEX, par_ini=current_job.createParameterSet(), transform=logtransform)
            elif args.method == 'DE':
                minpar, maxpar = current_job.getParameterBounds()

                popsize = 10*len(minpar)
                maxgen = args.maxiter
                if args.maxfun is not None:
                    maxgen = min(maxgen, int(numpy.ceil(args.maxfun/float(popsize))))
                print('Maximum number of generations for Differential Evolution (based on maxfun and maxiter): %i' % maxgen)
                startpoppath = 'startpop.dat'

                startpop = None
                if os.path.isfile(startpoppath):
                    # Retrieve cached copy of the observations
                    print('Reading initial population from file %s...' % startpoppath)
                    startpop = numpy.load(startpoppath)

                extra_args = {'parallelize': False}
                if service.parallel is not None:
                    extra_args.update(parallelize=True, ncpus=args.ncpus, ppservers=job.shared.parse_ppservers(args.ppservers), secret=args.secret)
                if args.ftol is not None:
                    extra_args.update(ftol=args.ftol, abstol=numpy.inf)
                vals = opt.run(method=optimize.DIFFERENTIALEVOLUTION, par_min=minpar, par_max=maxpar, popsize=popsize, maxgen=maxgen, F=args.F, CR=args.CR, initialpopulation=startpop, transform=logtransform, max_runtime=getattr(current_job, 'max_runtime', None), **extra_args)

                reporter.finalize()

            print('Best parameter set:')
            for parameter, value in zip(current_job.parameters, vals):
                print('  %s = %.6g' % (parameter.name, value))

            if args.method == 'fmin' or not args.repeat:
                break
    finally:
        reporter.finalize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
