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

#https://stackoverflow.com/questions/9505898/conditional-command-line-arguments-in-python-using-argparse

    parser.add_argument('xmlfile', type=str, help='XML formatted configuration file')
    subparsers = parser.add_subparsers(dest='subcommand')

    parser.add_argument('-t', '--transport', type=str, choices=('http', 'mysql'), help='Transport to use for server communication: http or mysql')
    parser.add_argument('-r', '--reportinterval', type=int, help='Time between result reports (seconds).')
    parser.add_argument('-i', '--interactive', action='store_true', help='Whether to allow for user interaction (input from stdin) when making decisions')

    parser_sample = subparsers.add_parser('sample')
#    parser_sample.add_argument('method', type=str, choices=('fast', 'ff', 'finite_diff', 'latin', 'saltelli', 'sobol_sequence'), help='Sampling method.', default='saltelli')
    parser_sample.add_argument('N', type=int, help='Ensemble size')
    parser_sample.add_argument('--create_dirs', type=str, help='Create setup directories under the specified root.')
    parser_sample.add_argument('--format', type=str, help='Format for subdirectory name (only in combination with --create_dirs).', default='%04i')

    parser_modelrun = subparsers.add_parser('modelrun')

#    parser_list = subparsers.add_parser('list')
#    parser_generate = subparsers.add_parser('generate')

    subparsers_sample = parser_sample.add_subparsers(dest='subsubcommand')
    subparser_sample_fast = subparsers_sample.add_parser('fast')
    subparser_sample_fast.add_argument('--M', type=int, help='The interference parameter', default=4)
    subparser_sample_ff = subparsers_sample.add_parser('ff')
    subparser_sample_finite_diff = subparsers_sample.add_parser('finite_diff')
    subparser_sample_finite_diff.add_argument('--delta', type=float, help='Finite difference step size (percent)', default=0.01)
    subparser_sample_latin = subparsers_sample.add_parser('latin')
    subparser_sample_saltelli = subparsers_sample.add_parser('saltelli')
#   Jorn - what is the best way to set a switch? With a default value of either True or False
    subparser_sample_saltelli.add_argument('--no_calc_second_order', action='store_true', help='Calculate second-order sensitivities', default=False)
    subparser_sample_sobol_sequence = subparsers_sample.add_parser('sobol_sequence')

    parser_analyze = subparsers.add_parser('analyze')
    subparsers_analyze = parser_analyze.add_subparsers(dest='subsubcommand')
    subparser_analyze_delta = subparsers_analyze.add_parser('delta')
    subparser_analyze_delta.add_argument('--num_resamples', type=int, help='The number of resamples when computing confidence intervals', default=10)
    subparser_analyze_delta.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)
    subparser_analyze_delta.add_argument('--print_to_console', action='store_true', help='Print results directly to console', default=False)

    subparser_analyze_dgsm = subparsers_analyze.add_parser('dgsm')
    subparser_analyze_dgsm.add_argument('--num_resamples', type=int, help='The number of resamples when computing confidence intervals', default=1000)
    subparser_analyze_dgsm.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)
    subparser_analyze_dgsm.add_argument('--print_to_console', action='store_true', help='Print results directly to console', default=False)

    subparser_analyze_fast = subparsers_analyze.add_parser('fast')
    subparser_analyze_fast.add_argument('--M', type=int, help='The interference parameter, i.e., the number of harmonics to sum in the Fourier series decomposition', default=4)
    subparser_analyze_fast.add_argument('--print_to_console', action='store_true', help='Print results directly to console', default=False)

    subparser_analyze_ff = subparsers_analyze.add_parser('ff')
    subparser_analyze_ff.add_argument('--second_order', action='store_true', help='Include interaction effects', default=False)
    subparser_analyze_ff.add_argument('--print_to_console', action='store_true', help='Print results directly to console', default=False)

    subparser_analyze_morris = subparsers_analyze.add_parser('morris')
    subparser_analyze_morris.add_argument('--num_resamples', type=int, help='The number of resamples when computing confidence intervals', default=1000)
    subparser_analyze_morris.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)
    subparser_analyze_morris.add_argument('--print_to_console', action='store_true', help='Print results directly to console', default=False)
    subparser_analyze_morris.add_argument('--grid_jump', type=int, help='The number of resamples when computing confidence intervals', default=2)
    subparser_analyze_morris.add_argument('--num_levels', type=int, help='The number of grid levels, must be identical to the value passed to SALib.sample.morris', default=4)

    subparser_analyze_rbd_fast = subparsers_analyze.add_parser('rbd_fast')
    subparser_analyze_rbd_fast.add_argument('--M', type=int, help='The interference parameter, i.e., the number of harmonics to sum in the Fourier series decomposition', default=4)
    subparser_analyze_rbd_fast.add_argument('--print_to_console', action='store_true', help='Print results directly to console', default=False)

    subparser_analyze_sobol = subparsers_analyze.add_parser('sobol')
    subparser_analyze_sobol.add_argument('--num_resamples', type=int, help='The number of resamples', default=100)
    subparser_analyze_sobol.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)
    subparser_analyze_sobol.add_argument('--print_to_console', action='store_true', help='Print results directly to console', default=False)
#   two extra parameters to Sobol - parallel=False, n_processors=Non

def main(args):
    print 'Reading configuration from %s...' % args.xmlfile
    current_job = job.fromConfigurationFile(args.xmlfile)

    names = current_job.getParameterNames()
    minpar, maxpar = current_job.getParameterBounds()
    SAlib_problem = {'num_vars': len(names),
                     'names': names,
                     'bounds': list(zip(minpar, maxpar))
                    }
#    print(len(names))
#    print(names)
#    print(list(zip(minpar, maxpar)))

#    print(args.subcommand)
    if args.subcommand == 'sample':
#        print(args.subsubcommand)
        assert args.create_dirs is not None, 'Currently --create_dirs argment must be provided'
        if not os.path.isdir(args.create_dirs):
            os.mkdir(args.create_dirs)

        Nensemble = args.N

        if args.subsubcommand == 'fast':
            from SALib.sample.fast_sampler import sample
            parameters = sample(SAlib_problem, args.N, args.M)
        elif args.subsubcommand == 'ff':
            assert args.subsubcommand is 'ff', 'Currently ff method is not supported'
            from SALib.sample.ff import sample
            parameters = sample(SAlib_problem, args.N, args.M)
        elif args.subsubcommand == 'finite_diff':
            from SALib.sample.finite_diff import sample
            parameters = sample(SAlib_problem, args.N, args.delta)
        elif args.subsubcommand == 'latin':
            from SALib.sample.latin import sample
            parameters = sample(SAlib_problem, args.N)
        elif args.subsubcommand == 'saltelli':
            from SALib.sample.saltelli import sample
            calc_second_order = not args.no_calc_second_order
            if calc_second_order:
                Nensemble = args.N*(len(names)+2)
            else:
                Nensemble = args.N*(2*len(names)+2)
            parameters = sample(SAlib_problem, args.N, calc_second_order)
        elif args.subsubcommand == 'sobol_sequence':
            assert args.subsubcommand is 'sobol_sequence', 'Currently ff method is not supported'
            from SALib.sample.sobol_sequence import sample

        print('Nensemble = %d generated' % (Nensemble))


        if args.create_dirs:
        # Only create setup directories
            assert isinstance(current_job, job.program.Job)
            scenariodir = current_job.scenariodir
            for i in xrange(parameters.shape[0]):
                current_job.scenariodir = scenariodir
                current_job.simulationdir = os.path.join(args.create_dirs, args.format % i)
                current_job.initialized = False
                current_job.prepareDirectory(parameters[i, :])

    if args.subcommand == 'analyze':
#        assert (args.method == 'sobol'), 'Currently only the sobol analyze method is supported'

        if args.subsubcommand == 'delta':
            from SALib.analyze.delta import analyze
            assert args.subsubcommand is 'delta', 'Currently delta method is not supported'
            analysis = analyze(SAlib_problem, X, Y, args.num_resamples, args.conf_level, args.print_to_console)
        elif args.subsubcommand == 'dgsm':
            from SALib.analyze.dgsm import analyze
            assert args.subsubcommand is 'dgsm', 'Currently dgsm method is not supported'
            analysis = analyze(SAlib_problem, X, Y, args.num_resamples, args.conf_level, args.print_to_console)
        elif args.subsubcommand == 'fast':
            from SALib.analyze.fast import analyze
            assert args.subsubcommand is 'fast', 'Currently fast method is not supported'
            analysis = analyze(SAlib_problem, Y, args.M, args.print_to_console)
        elif args.subsubcommand == 'ff':
            from SALib.analyze.ff import analyze
            assert args.subsubcommand is 'ff', 'Currently ff method is not supported'
            analysis = analyze(SAlib_problem, X, Y, args.print_to_console)
        elif args.subsubcommand == 'morris':
            from SALib.analyze.morris import analyze
            assert args.subsubcommand is 'morris', 'Currently morris method is not supported'
            analysis = analyze(SAlib_problem, X, Y, args.num_resamples, args.conf_level, args.print_to_console, args.grid_jump, args.num_level)
        elif args.subsubcommand == 'rbd_fast':
            from SALib.analyze.rbd_fast import analyze
            assert args.subsubcommand is 'rbd_fast', 'Currently rbd_fast method is not supported'
            analysis = analyze(SAlib_problem, X, Y, args.num_resamples, args.M, args.print_to_console)
        elif args.subsubcommand == 'sobol':
            from SALib.analyze.sobol import analyze
#KB
            calc_second_order = True
#KB
            analysis = analyze(SAlib_problem, Y, calc_second_order, args.num_resamples, args.conf_level, args.print_to_console)

        # Run the actual sensitivity analysis (yet to be implemented)
        with open(args.xmlfile) as f:
            xml = f.read()
        reporter = report.fromConfigurationFile(args.xmlfile, xml, allowedtransports=args.transport)

        # Configure result reporter
        reporter.interactive = args.interactive
        if args.reportinterval is not None:
            reporter.timebetweenreports = args.reportinterval

        try:
            pass
        finally:
            reporter.finalize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
