#!/usr/bin/env python

from __future__ import print_function

# Import from standard Python library
import sys
import os
import argparse
import tempfile
import cPickle

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

    parser_sample = subparsers.add_parser('sample')
#    parser_sample.add_argument('method', type=str, choices=('fast', 'ff', 'finite_diff', 'latin', 'saltelli', 'sobol_sequence'), help='Sampling method.', default='saltelli')
    parser_sample.add_argument('--create_dirs', type=str, help='Create setup directories under the specified root.')
    parser_sample.add_argument('--format', type=str, help='Format for subdirectory name (only in combination with --create_dirs).', default='%04i')

    parser_run = subparsers.add_parser('run')
    parser_run.add_argument('job_dir', help='Directory containing results per simulation (and sa_job.pickle)')
    parser_run.add_argument('-t', '--transport', choices=('http', 'mysql'), help='Transport to use for server communication: http or mysql')
    parser_run.add_argument('-r', '--reportinterval', type=int, help='Time between result reports (seconds).')
    parser_run.add_argument('-i', '--interactive', action='store_true', help='Whether to allow for user interaction (input from stdin) when making decisions')

#    parser_list = subparsers.add_parser('list')
#    parser_generate = subparsers.add_parser('generate')

    subparsers_sample = parser_sample.add_subparsers(dest='method')

    subparser_sample_fast = subparsers_sample.add_parser('fast')
    subparser_sample_fast.add_argument('N', type=int, help='The number of samples to generate')
    subparser_sample_fast.add_argument('--M', type=int, help='The interference parameter, i.e., the number of harmonics to sum in the Fourier series decomposition', default=4)

    subparser_sample_latin = subparsers_sample.add_parser('latin')
    subparser_sample_latin.add_argument('N', type=int, help='The number of samples to generate')

    subparser_analyze_morris = subparsers_sample.add_parser('morris')
    subparser_analyze_morris.add_argument('N', type=int, help='The number of trajectories to generate')
    subparser_analyze_morris.add_argument('num_levels', type=int, help='The number of grid levels')
    subparser_analyze_morris.add_argument('grid_jump', type=int, help='The grid jump size')
    subparser_analyze_morris.add_argument('--optimal_trajectories', type=int, help='The number of optimal trajectories to sample (between 2 and N)', default=None)
    subparser_analyze_morris.add_argument('--no_local_optimization', dest='local_optimization', action='store_false', help='Disable local optimization according to Ruano et al. (2012) Local optimization speeds up the process tremendously for bigger N and num_levels.')

    subparser_sample_saltelli = subparsers_sample.add_parser('saltelli')
    subparser_sample_saltelli.add_argument('N', type=int, help='The number of samples to generate')
    subparser_sample_saltelli.add_argument('--no_calc_second_order', dest='calc_second_order', action='store_false', help='Disable calculation of second-order sensitivities')

    subparser_sample_ff = subparsers_sample.add_parser('ff')

    #subparser_sample_finite_diff = subparsers_sample.add_parser('finite_diff')
    #subparser_sample_finite_diff.add_argument('--delta', type=float, help='Finite difference step size (percent)', default=0.01)

    #subparser_sample_sobol_sequence = subparsers_sample.add_parser('sobol_sequence')

    parser_analyze = subparsers.add_parser('analyze')
    parser_analyze.add_argument('job_dir', help='Directory containing results per simulation (and sa_job.pickle)')
    subparsers_analyze = parser_analyze.add_subparsers(dest='method')

    subparser_analyze_fast = subparsers_analyze.add_parser('fast')
    subparser_analyze_fast.add_argument('--print_to_console', action='store_true', help='Print results directly to console')

    subparser_analyze_rbd_fast = subparsers_analyze.add_parser('rbd_fast')
    subparser_analyze_rbd_fast.add_argument('--M', type=int, help='The interference parameter, i.e., the number of harmonics to sum in the Fourier series decomposition', default=10)
    subparser_analyze_rbd_fast.add_argument('--print_to_console', action='store_true', help='Print results directly to console')

    subparser_analyze_morris = subparsers_analyze.add_parser('morris')
    subparser_analyze_morris.add_argument('--num_resamples', type=int, help='The number of resamples when computing confidence intervals', default=1000)
    subparser_analyze_morris.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)
    subparser_analyze_morris.add_argument('--print_to_console', action='store_true', help='Print results directly to console')

    subparser_analyze_sobol = subparsers_analyze.add_parser('sobol')
    subparser_analyze_sobol.add_argument('--num_resamples', type=int, help='The number of resamples', default=100)
    subparser_analyze_sobol.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)
    subparser_analyze_sobol.add_argument('--print_to_console', action='store_true', help='Print results directly to console')
#   two extra parameters to Sobol - parallel=False, n_processors=Non

    subparser_analyze_delta = subparsers_analyze.add_parser('delta')
    subparser_analyze_delta.add_argument('--num_resamples', type=int, help='The number of resamples when computing confidence intervals', default=10)
    subparser_analyze_delta.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)
    subparser_analyze_delta.add_argument('--print_to_console', action='store_true', help='Print results directly to console')

    subparser_analyze_dgsm = subparsers_analyze.add_parser('dgsm')
    subparser_analyze_dgsm.add_argument('--num_resamples', type=int, help='The number of resamples when computing confidence intervals', default=1000)
    subparser_analyze_dgsm.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)
    subparser_analyze_dgsm.add_argument('--print_to_console', action='store_true', help='Print results directly to console')

    subparser_analyze_ff = subparsers_analyze.add_parser('ff')
    subparser_analyze_ff.add_argument('--second_order', action='store_true', help='Include interaction effects')
    subparser_analyze_ff.add_argument('--print_to_console', action='store_true', help='Print results directly to console')

def load_job(path):
    if not os.path.isdir(path):
        print('%s must be an existing directory' % path)
        return
    job_path = os.path.join(path, 'sa_job.pickle')        
    if not os.path.isfile(job_path):
        print('Directory %s does not contain %s and therefore cannot contain results of a sensitivity analysis.' % (path, os.path.basename(job_path)))
        return
    with open(job_path, 'rb') as f:
        return cPickle.load(f)

def main(args):
    print('Reading configuration from %s...' % args.xmlfile)
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

    if args.subcommand == 'sample':
        assert args.create_dirs is not None, 'Currently --create_dirs argment must be provided'
        if not os.path.isdir(args.create_dirs):
            os.mkdir(args.create_dirs)

        Nensemble = args.N

        if args.method == 'fast':
            from SALib.sample.fast_sampler import sample
            X = sample(SAlib_problem, args.N, args.M)
        elif args.method == 'latin':
            from SALib.sample.latin import sample
            X = sample(SAlib_problem, args.N)
        elif args.method == 'morris':
            from SALib.sample.morris import sample
            X = sample(SAlib_problem, args.N, args.num_levels, args.grid_jump, optimal_trajectories=args.optimal_trajectories, local_optimization=args.local_optimization)
        elif args.method == 'saltelli':
            from SALib.sample.saltelli import sample
            X = sample(SAlib_problem, args.N, calc_second_order=args.calc_second_order)
        elif args.method == 'ff':
            from SALib.sample.ff import sample
            X = sample(SAlib_problem)
        #elif args.method == 'finite_diff':
        #    from SALib.sample.finite_diff import sample
        #    X = sample(SAlib_problem, args.N, args.delta)
        #elif args.method == 'sobol_sequence':
        #    print('Sampler "sobol_sequence" is currently not supported'
        #    sys.exit(2)
        #    from SALib.sample.sobol_sequence import sample#
        else:
            print('Unknown sampler "%s" specified.' % args.method)

        print('Generated an ensemble with %i members' % (X.shape[0],))

        if args.create_dirs:
        # Only create setup directories
            assert isinstance(current_job, job.program.Job)
            scenariodir = current_job.scenariodir
            simulationdirs = [os.path.join(args.create_dirs, args.format % i) for i in xrange(X.shape[0])]
            for i, simulationdir in enumerate(simulationdirs):
                current_job.scenariodir = scenariodir
                current_job.simulationdir = simulationdir
                current_job.start(force=True)
                current_job.prepareDirectory(X[i, :])
            with open(os.path.join(args.create_dirs, 'sa_job.pickle'), 'wb') as f:
                cPickle.dump((args, X, simulationdirs), f, cPickle.HIGHEST_PROTOCOL)
    elif args.subcommand == 'run':
        job_info = load_job(args.job_dir)
        if not job_info:
            sys.exit(2)
        sampler_args, X, simulationdirs = job_info

        # Run the actual sensitivity analysis (yet to be implemented)
        with open(args.xmlfile) as f:
            xml = f.read()
        reporter = report.fromConfigurationFile(args.xmlfile, xml, allowedtransports=args.transport)

        # Configure result reporter
        reporter.interactive = args.interactive
        if args.reportinterval is not None:
            reporter.timebetweenreports = args.reportinterval

        try:
            for i, simulationdir in enumerate(simulationdirs):
                print('Running ensemble member %i in %s...' % (i, simulationdir))
                current_job.scenariodir = simulationdir
                returncode = current_job.run()
        finally:
            reporter.finalize()
    elif args.subcommand == 'analyze':
        job_info = load_job(args.job_dir)
        if not job_info:
            sys.exit(2)
        sampler_args, X, simulationdirs = job_info

        Y = numpy.empty((len(simulationdirs),))
        expression, ncpath = current_job.target
        print('Retrieving value of target expression for esah ensemble member...')
        for i, simulationdir in enumerate(simulationdirs):

            wrappednc = job.program.NcDict(os.path.join(simulationdir, ncpath))
            Y[i] = wrappednc.eval(expression)
            print('  - %i: %s' % (i, Y[i]))
            wrappednc.finalize()

        if args.method == 'fast':
            assert sampler_args.method == 'fast'
            from SALib.analyze.fast import analyze
            analysis = analyze(SAlib_problem, Y, sampler_args.M, args.print_to_console)
        elif args.method == 'rbd_fast':
            assert sampler_args.method == 'latin'
            from SALib.analyze.rbd_fast import analyze
            analysis = analyze(SAlib_problem, Y, X, M=args.M, print_to_console=args.print_to_console)
        elif args.method == 'morris':
            assert sampler_args.method == 'morris'
            from SALib.analyze.morris import analyze
            analysis = analyze(SAlib_problem, X, Y, num_resamples=args.num_resamples, conf_level=args.conf_level, print_to_console=args.print_to_console, grid_jump=sampler_args.grid_jump, num_levels=sampler_args.num_levels)
        elif args.method == 'sobol':
            assert sampler_args.method == 'saltelli'
            from SALib.analyze.sobol import analyze
            analysis = analyze(SAlib_problem, Y, calc_second_order=sampler_args.calc_second_order, num_resamples=args.num_resamples, conf_level=args.conf_level, print_to_console=args.print_to_console)
        elif args.method == 'delta':
            assert sampler_args.method == 'latin'
            from SALib.analyze.delta import analyze
            analysis = analyze(SAlib_problem, X, Y, num_resamples=args.num_resamples, conf_level=args.conf_level, print_to_console=args.print_to_console)
        elif args.method == 'dgsm':
            from SALib.analyze.dgsm import analyze
            analysis = analyze(SAlib_problem, X, Y, num_resamples=args.num_resamples, conf_level=args.conf_level, print_to_console=args.print_to_console)
        elif args.method == 'ff':
            assert sampler_args.method == 'ff'
            from SALib.analyze.ff import analyze
            analysis = analyze(SAlib_problem, X, Y, second_order=args.second_order, print_to_console=args.print_to_console)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
