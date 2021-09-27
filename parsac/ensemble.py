#!/usr/bin/env python

from __future__ import print_function

# Import from standard Python library
import sys
import os
import argparse
import tempfile
import xml.etree.ElementTree
import glob

# Import third party libraries
import numpy
import numpy.random

# Import personal custom stuff
from . import result
from . import job

def configure_argument_parser(parser):
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_sample = subparsers.add_parser('sample')
    parser_sample.add_argument('xmlfile', type=str, help='XML formatted configuration file')
    parser_sample.add_argument('N', type=int, help='ensemble size')
    parser_sample.add_argument('--gridsize', type=int, help='number of cells per parameter grid', default=20)
    parser_sample.add_argument('--plot', action='store_true', help='show histogram of ensemble members')
    parser_sample.add_argument('--dir', type=str, help='directory to create ensemble setups in (one per member)')
    parser_sample.add_argument('--scenario', type=str, help='Scenario directory to use as template for ensemble-member-specific setups (only in combination with --dir).')
    parser_sample.add_argument('--format', type=str, help='Format for subdirectory name (only in combination with --dir).', default='%04i')

    parser_run = subparsers.add_parser('run')
    parser_run.add_argument('xmlfile', type=str, help='XML formatted configuration file')
    parser_run.add_argument('dirs', nargs='+', help='Directories to run in (may contain wildcards)')
    parser_run.add_argument('-n', '--ncpus', type=int, help='Number of cores to use (default: use all available on the local machine).')
    parser_run.add_argument('--ppservers',   type=str, help='Comma-separated list of names/IPs of Parallel Python servers to run on.')
    parser_run.add_argument('--secret',      type=str, help='Parallel Python secret for authentication (only used in combination with ppservers argument).')

def get_weights_grid(current_job, results, gridsize):
    # Build parameter grid (one dimension per parameter)
    # We will use this to normalize a parameter set's probability of being selected
    # by the number of other parameter sets that fall within the same grid point.
    minpar, maxpar = current_job.getParameterBounds()
    logscale = current_job.getParameterLogScale()
    pargrid = numpy.empty((len(minpar), gridsize))
    for ipar, (left, right, log) in enumerate(zip(minpar, maxpar, logscale)):
        if log:
            pargrid[ipar, :] = numpy.logspace(numpy.log10(left), numpy.log10(right), pargrid.shape[1])
        else:
            pargrid[ipar, :] = numpy.linspace(left, right, pargrid.shape[1])

    # Determine where each result sits in our npar-dimensional parameter grid
    index2count = {}
    indices = []
    for iresult in range(results.shape[0]):
        inds = tuple([pargrid[ipar, :].searchsorted(value) for ipar, value in enumerate(results[iresult, :-1])])
        indices.append(inds)
        index2count[inds] = index2count.get(inds, 0) + 1

    # Calculate probability-of-being-chosen for each original parameter set,
    # based on log-likelihood and the proximity of other results (i..e, their co-occurence in the same grid cell)
    weights = numpy.ones((results.shape[0],))
    for iresult, ind in enumerate(indices):
        weights[iresult] /= index2count[ind]
    return weights

def get_weights_radius(current_job, results, M=10):
    # From each parameter set, find out the radius of the hypershere that includes the nearest
    # M parameter sets. The volume associated with that hypershere is an approximation of the
    # sampling density around that parameter set, and will be used to weight the associated PDF [likelihood] value.
    minpar, maxpar = current_job.getParameterBounds()
    logscale = current_job.getParameterLogScale()
    relparvalues = numpy.empty_like(results[:, :-1])
    for ipar, (minval, maxval, log) in enumerate(zip(minpar, maxpar, logscale)):
        values = results[:, ipar]
        if log:
            minval, maxval, values = numpy.log10(minval), numpy.log10(maxval), numpy.log10(values)
        relparvalues[:, ipar] = (values - minval) / (maxval - minval)
    weights = numpy.empty((results.shape[0],))
    for iresult in range(relparvalues.shape[0]):
        dist = numpy.sqrt(((relparvalues - relparvalues[iresult, :])**2).sum(axis=1))
        imindist = dist.argsort()[:M+1]
        weights[iresult] = dist[imindist[-1]]**len(minpar)
    return weights

def main(args):
    if args.subcommand == 'run':
        dirs = []
        for d in args.dirs:
            dirs.extend(glob.glob(d))
        current_job = job.fromConfigurationFile(args.xmlfile)
        current_job.runEnsemble(dirs, ncpus=args.ncpus, ppservers=args.ppservers, secret=args.secret)
        return

    current_result = result.Result(args.xmlfile)
    results = current_result.get()
    if results.shape[0] < args.N:
        print('An ensemble of %i members was requested, but only %i results were found.' % (args.N, results.shape[0]))
        sys.exit(1)

    # Calculate probability-of-being-chosen for each original parameter set,
    # based on log-likelihood and the proximity of other results (i..e, their co-occurence in the same grid cell)
    rel_likelihood = numpy.exp(results[:, -1] - results[:, -1].max())
    #weights2 = get_weights_grid(current_result.job, results, args.gridsize)
    weights = get_weights_radius(current_result.job, results)
    p = rel_likelihood*weights
    p /= p.sum()

    # Select ensemble members
    ipicked = numpy.random.choice(numpy.arange(results.shape[0]), size=args.N, p=p)
    ensemble = results[ipicked, :-1]
    for i in range(ensemble.shape[0]):
        print(ensemble[i, :])

    if args.dir:
        assert isinstance(current_result.job, job.program.Job)
        if args.scenario is not None:
            current_result.job.scenariodir = args.scenario
        current_result.job.prepareEnsembleDirectories(ensemble, args.dir, args.format)

    if args.plot:
        # Show histogram of ensemble members
        names = current_result.job.getParameterNames()
        from matplotlib import pyplot
        fig = pyplot.figure()
        npar = results.shape[1] - 1
        for i, name in enumerate(names):
            ax = fig.add_subplot(1, npar, i+1)
            ax.hist(results[ipicked, i], bins=50)
            ax.set_xlabel(name)
        fig = pyplot.figure()
        ax = fig.gca()
        ax.semilogy(weights/weights.sum(), label='weights')
        ax.semilogy(rel_likelihood/rel_likelihood.sum(), label='likelihood')
        ax.semilogy(p, label='sampling probability')
        ax.legend()
        ax.set_xlabel('calibration trial number')
        ax.set_ylabel('sampling weight for ensemble')
        pyplot.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
