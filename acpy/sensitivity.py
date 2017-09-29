#!/usr/bin/env python

# Import from standard Python library
import sys
import os
import argparse
import tempfile

# Import third party libraries
import numpy

# Import personal custom stuff
import service
import optimize
import job
import report

def configure_argument_parser(parser):
    parser.add_argument('xmlfile', type=str, help='XML formatted configuration file')
    parser.add_argument('N', type=int, help='Ensemble size')
    parser.add_argument('-t', '--transport', type=str, choices=('http', 'mysql'), help='Transport to use for server communication: http or mysql')
    parser.add_argument('-r', '--reportinterval', type=int, help='Time between result reports (seconds).')
    parser.add_argument('-i', '--interactive', action='store_true', help='Whether to allow for user interaction (input from stdin) when making decisions')
    parser.add_argument('--create_dirs', type=str, help='Create setup directories under the specified root.')
    parser.add_argument('--format', type=str, help='Format for subdirectory name (only in combination with --create_dirs).', default='%04i')

def main(args):
    assert args.create_dirs is not None, 'Currently --create_dirs argment must be provided'
    if not os.path.isdir(args.create_dirs):
        os.mkdir(args.create_dirs)

    print 'Reading configuration from %s...' % args.xmlfile
    current_job = job.fromConfigurationFile(args.xmlfile)

    with open(args.xmlfile) as f:
        xml = f.read()
    reporter = report.fromConfigurationFile(args.xmlfile, xml, allowedtransports=args.transport)

    # Configure result reporter
    reporter.interactive = args.interactive
    if args.reportinterval is not None:
        reporter.timebetweenreports = args.reportinterval

    names = current_job.getParameterNames()
    minpar, maxpar = current_job.getParameterBounds()
    try:
        import SALib.sample.saltelli
        problem = {'num_vars': len(names),
                   'names': names,
                   'bounds': list(zip(minpar, maxpar))
        }
        parameters = SALib.sample.saltelli.sample(problem, args.N)
        assert isinstance(current_job, job.program.Job)
        for i in xrange(parameters.shape[0]):
            current_job.simulationdir = os.path.join(args.create_dirs, args.format % i)
            current_job.initialized = False
            current_job.prepareDirectory(parameters[i, :])
    finally:
        reporter.finalize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
