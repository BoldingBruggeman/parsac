#!/usr/bin/env python

from __future__ import print_function

# Import from standard Python library
import sys
import os
import argparse
import tempfile
import cPickle
import xml.etree.ElementTree

# Import third party libraries
import numpy

# Import personal custom stuff
import service
import optimize
import job
import report

def configure_argument_parser(parser):
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_sample = subparsers.add_parser('sample')
    parser_sample.add_argument('xmlfile', type=str, help='XML formatted configuration file')
    parser_sample.add_argument('info', type=str, help='Path to save info to')
    parser_sample.add_argument('--dir', type=str, help='Directory under which setups per ensemble member are to be created/found. If this argument is not provided, the run step will creates temporary setup directories and run the model itself. If this argument is provided, the user is responsible for running the model in each of the created setup directories.')
    parser_sample.add_argument('--format', type=str, help='Format for subdirectory name (only in combination with --dir).', default='%04i')

    subparsers_sample = parser_sample.add_subparsers(dest='method')

    subparser_sample_fast = subparsers_sample.add_parser('fast')
    subparser_sample_fast.add_argument('N', type=int, help='The number of samples to generate')
    subparser_sample_fast.add_argument('--M', type=int, help='The interference parameter, i.e., the number of harmonics to sum in the Fourier series decomposition', default=4)

    subparser_sample_latin = subparsers_sample.add_parser('latin')
    subparser_sample_latin.add_argument('N', type=int, help='The number of samples to generate')

    subparser_sample_morris = subparsers_sample.add_parser('morris')
    subparser_sample_morris.add_argument('N', type=int, help='The number of trajectories to generate')
    subparser_sample_morris.add_argument('--num_levels', type=int, help='The number of grid levels', default=4)
    subparser_sample_morris.add_argument('--grid_jump', type=int, help='The grid jump size', default=2)
    subparser_sample_morris.add_argument('--optimal_trajectories', type=int, help='The number of optimal trajectories to sample (between 2 and N)', default=None)
    subparser_sample_morris.add_argument('--no_local_optimization', dest='local_optimization', action='store_false', help='Disable local optimization according to Ruano et al. (2012) Local optimization speeds up the process tremendously for bigger N and num_levels.')

    subparser_sample_saltelli = subparsers_sample.add_parser('saltelli')
    subparser_sample_saltelli.add_argument('N', type=int, help='The number of samples to generate')
    subparser_sample_saltelli.add_argument('--no_calc_second_order', dest='calc_second_order', action='store_false', help='Disable calculation of second-order sensitivities')

    subparser_sample_ff = subparsers_sample.add_parser('ff')

    parser_run = subparsers.add_parser('run')
    parser_run.add_argument('info', type=str, help='Path to output of the "sample" step')

    parser_analyze = subparsers.add_parser('analyze')
    parser_analyze.add_argument('info', type=str, help='Path to output of the "sample" step')
    parser_analyze.add_argument('--print_to_console', action='store_true', help='Print results directly to console')
    parser_analyze.add_argument('--select', nargs=2, help='This requires two values: N OUTPUTXML. Selects the N most sensitive parameters for a calibation run and save it to OUTPUTXML')
    subparsers_analyze = parser_analyze.add_subparsers(dest='method')

    subparser_analyze_fast = subparsers_analyze.add_parser('fast')

    subparser_analyze_rbd_fast = subparsers_analyze.add_parser('rbd_fast')
    subparser_analyze_rbd_fast.add_argument('--M', type=int, help='The interference parameter, i.e., the number of harmonics to sum in the Fourier series decomposition', default=10)

    subparser_analyze_morris = subparsers_analyze.add_parser('morris')
    subparser_analyze_morris.add_argument('--num_resamples', type=int, help='The number of resamples when computing confidence intervals', default=1000)
    subparser_analyze_morris.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)

    subparser_analyze_sobol = subparsers_analyze.add_parser('sobol')
    subparser_analyze_sobol.add_argument('--num_resamples', type=int, help='The number of resamples', default=100)
    subparser_analyze_sobol.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)
#   two extra parameters to Sobol - parallel=False, n_processors=Non

    subparser_analyze_delta = subparsers_analyze.add_parser('delta')
    subparser_analyze_delta.add_argument('--num_resamples', type=int, help='The number of resamples when computing confidence intervals', default=10)
    subparser_analyze_delta.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)

    subparser_analyze_dgsm = subparsers_analyze.add_parser('dgsm')
    subparser_analyze_dgsm.add_argument('--num_resamples', type=int, help='The number of resamples when computing confidence intervals', default=1000)
    subparser_analyze_dgsm.add_argument('--conf_level', type=float, help='The confidence interval level', default=0.95)

    subparser_analyze_ff = subparsers_analyze.add_parser('ff')
    subparser_analyze_ff.add_argument('--second_order', action='store_true', help='Include interaction effects')

def sample(SAlib_problem, args):
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
    else:
        print('Unknown sampler "%s" specified.' % args.method)
    print('Generated an ensemble with %i members' % (X.shape[0],))
    return X

def analyze(SAlib_problem, args, sample_args, X, Y):
    if args.method == 'fast':
        # https://dx.doi.org/10.1063/1.1680571
        # https://dx.doi.org/10.1080/00401706.1999.10485594
        assert sample_args.method == 'fast'
        import SALib.analyze.fast
        analysis = SALib.analyze.fast.analyze(SAlib_problem, Y, sample_args.M, args.print_to_console)
    elif args.method == 'rbd_fast':
        # https://dx.doi.org/10.1016/j.ress.2005.06.003
        # https://dx.doi.org/10.1016/j.ress.2009.11.005
        # https://dx.doi.org/10.1016/j.ress.2012.06.010
        # https://dx.doi.org/10.1080/19401493.2015.1112430
        assert sample_args.method == 'latin'
        import SALib.analyze.rbd_fast
        analysis = SALib.analyze.rbd_fast.analyze(SAlib_problem, Y, X, M=M, print_to_console=args.print_to_console)
    elif args.method == 'morris':
        # https://dx.doi.org/10.1080/00401706.1991.10484804
        # https://dx.doi.org/10.1016/j.envsoft.2006.10.004
        assert sample_args.method == 'morris'
        import SALib.analyze.morris
        analysis = SALib.analyze.morris.analyze(SAlib_problem, X, Y, num_resamples=args.num_resamples, conf_level=args.conf_level, print_to_console=args.print_to_console, grid_jump=sample_args.grid_jump, num_levels=sample_args.num_levels)
        sensitivities = analysis['mu_star']
    elif args.method == 'sobol':
        # https://dx.doi.org/10.1016/S0378-4754(00)00270-6
        # https://dx.doi.org/10.1016/S0010-4655(02)00280-1
        # https://dx.doi.org/10.1016/j.cpc.2009.09.018
        assert sample_args.method == 'saltelli'
        import SALib.analyze.sobol
        analysis = SALib.analyze.sobol.analyze(SAlib_problem, Y, calc_second_order=sample_args.calc_second_order, num_resamples=args.num_resamples, conf_level=args.conf_level, print_to_console=args.print_to_console)
        sensitivities = analysis['ST']
    elif args.method == 'delta':
        # https://dx.doi.org/10.1016/j.ress.2006.04.015
        # https://dx.doi.org/10.1016/j.ejor.2012.11.047
        assert sample_args.method == 'latin'
        import SALib.analyze.delta
        analysis = SALib.analyze.delta.analyze(SAlib_problem, X, Y, num_resamples=args.num_resamples, conf_level=args.conf_level, print_to_console=args.print_to_console)
    elif args.method == 'dgsm':
        import SALib.analyze.dgsm
        analysis = SALib.analyze.dgsm.analyze(SAlib_problem, X, Y, num_resamples=args.num_resamples, conf_level=args.conf_level, print_to_console=args.print_to_console)
    elif args.method == 'ff':
        assert sample_args.method == 'ff'
        import SALib.analyze.ff
        analysis = SALib.analyze.ff.analyze(SAlib_problem, X, Y, second_order=args.second_order, print_to_console=args.print_to_console)
    for name, sensitivity in sorted(zip(SAlib_problem['names'], sensitivities), cmp=lambda x, y: cmp(y[1], x[1])):
        print('%s: %s' % (name, sensitivity))
    return sensitivities

def undoLogTransform(values, logscale):
    return numpy.array([v if not log else 10.**v for log, v in zip(logscale, values)])

def save_info(path, info):
    with open(path, 'wb') as f:
        cPickle.dump(info, f, cPickle.HIGHEST_PROTOCOL)

def main(args):
    if args.subcommand != 'sample':
        print('Reading acpy/sa info from %s...' % args.info)
        with open(args.info, 'rb') as f:
            job_info = cPickle.load(f)
        args.xmlfile = job_info['sample_args'].xmlfile

    print('Reading configuration from %s...' % args.xmlfile)
    current_job = job.fromConfigurationFile(args.xmlfile)

    names = current_job.getParameterNames()
    minpar, maxpar = current_job.getParameterBounds()
    logscale = current_job.getParameterLogScale()

    # For parameters that having been marked for log-transformation,
    # transform their ranges now so that SAlib will operate in log-transformed space at all times.
    for i, log in enumerate(logscale):
        if log:
            minpar[i] = numpy.log10(minpar[i])
            maxpar[i] = numpy.log10(maxpar[i])

    SAlib_problem = {'num_vars': len(names),
                     'names': names,
                     'bounds': list(zip(minpar, maxpar))
                    }

    if args.subcommand == 'sample':
        # Only create setup directories
        X = sample(SAlib_problem, args)
        job_info = {'sample_args': args, 'X': X}

        if args.dir is not None:
            assert isinstance(current_job, job.program.Job)
            if not os.path.isdir(args.dir):
                os.mkdir(args.dir)
            scenariodir = current_job.scenariodir
            job_info['simulationdirs'] = [os.path.join(args.dir, args.format % i) for i in xrange(X.shape[0])]
            for i, simulationdir in enumerate(job_info['simulationdirs']):
                current_job.scenariodir = scenariodir
                current_job.simulationdir = simulationdir
                current_job.start(force=True)
                parameter_values = undoLogTransform(X[i, :], logscale)
                current_job.prepareDirectory(parameter_values)
        save_info(args.info, job_info)
    elif args.subcommand == 'run':
        X = job_info['X']
        if 'simulationdirs' in job_info:
            # We have created all setup directories during the sample setp. The user must have run to model in each.
            Y = numpy.empty((len(job_info['simulationdirs']),))
            expression, ncpath = current_job.target
            expression = compile(expression, '<string>', 'eval')
            print('Retrieving value of target expression for each ensemble member...')
            for i, simulationdir in enumerate(job_info['simulationdirs']):
                wrappednc = job.program.NcDict(os.path.join(simulationdir, ncpath))
                Y[i] = wrappednc.eval(expression)
                print('  - %i: %s' % (i, Y[i]))
                wrappednc.finalize()
        else:
            # We run the model ourselves.
            Y = current_job.evaluate_ensemble([undoLogTransform(X[i, :], logscale) for i in xrange(X.shape[0])])
            Y = numpy.array(Y)
        job_info['Y'] = Y
        print('Updating acpy/sa info in %s...' % args.info)
        save_info(args.info, job_info)
    elif args.subcommand == 'analyze':
        if 'Y' not in job_info:
            print('"analyze" step can only be used after "run" step')
            sys.exit(2)
        sensitivities = analyze(SAlib_problem, args, job_info['sample_args'], job_info['X'], job_info['Y'])
        if args.select is not None:
            n, path = args.select
            n = int(n)
            isort = numpy.argsort(sensitivities)
            print('Selecting top %i parameters:' % n)
            selected = set()
            for i in isort[-n:][::-1]:
                print('- %s' % (names[i],))
                selected.add(names[i])
            xml_tree = xml.etree.ElementTree.parse(args.xmlfile)
            parameters_xml = xml_tree.find('./parameters')
            children = list(parameters_xml)
            for ipar, element in enumerate(parameters_xml.findall('./parameter')):
                with job.shared.XMLAttributes(element, 'parameter %i' % (ipar+1)) as att:
                    name = current_job.getParameter(att).name
                if name not in selected:
                    element.tag = 'disabled_parameter'
            xml_tree.write(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
