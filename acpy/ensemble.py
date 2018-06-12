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
import numpy.random

# Import personal custom stuff
import acpy.result
import acpy.job

def configure_argument_parser(parser):
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_sample = subparsers.add_parser('sample')
    parser_sample.add_argument('xmlfile', type=str, help='XML formatted configuration file')
    parser_sample.add_argument('N', type=int, help='ensemble size')
    parser_sample.add_argument('--gridsize', type=int, help='number of cells per parameter grid', default=20)
    parser_sample.add_argument('--plot', action='store_true', help='show histogram of ensemble members')

def main(args):
    result = acpy.result.Result(args.xmlfile)

    results = result.get()
    npar = results.shape[1] - 1

    # Build parameter grid (one dimension per parameter)
    # We will use this to normalize a parameetr set's probability of beign selected
    # by the number fo other parameter sets that fall within the same grid point.
    minpar, maxpar = result.job.getParameterBounds()
    logscale = result.job.getParameterLogScale()
    pargrid = numpy.empty((npar, args.gridsize))
    for ipar, (left, right, log) in enumerate(zip(minpar, maxpar, logscale)):
        if log:
            pargrid[ipar, :] = numpy.logspace(numpy.log10(left), numpy.log10(right), pargrid.shape[1])
        else:
            pargrid[ipar, :] = numpy.linspace(left, right, pargrid.shape[1])

    # Determine where each result sits in our npar-dimensional parameter grid
    index2count = {}
    indices = []
    for iresult in xrange(results.shape[0]):
        inds = tuple([pargrid[ipar].searchsorted(results[iresult, ipar]) for ipar in xrange(npar)])
        indices.append(inds)
        index2count[inds] = index2count.get(inds, 0) + 1

    # Calculate probability-of-being-chosen for each original parameter set,
    # based on log-likelihood and the proximity of other results (i..e, their co-occurence in the same grid cell)
    weights = numpy.exp(results[:, -1] - results[:, -1].max())
    for iresult, ind in enumerate(indices):
        weights[iresult] /= index2count[ind]
    weights /= weights.sum()

    # Select ensemble members
    ipicked = numpy.random.choice(numpy.arange(results.shape[0]), size=args.N, p=weights)
    for i in ipicked:
        print(results[i, :-1])

    if args.plot:
        # Show histogram of ensemble members
        from matplotlib import pyplot
        fig = pyplot.figure()
        for i in xrange(npar):
            ax = fig.add_subplot(1, npar, i+1)
            ax.hist(results[ipicked, ipar], pargrid[ipar, :])
        pyplot.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
