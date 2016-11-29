#!/usr/bin/env python

# Import from standard Python library
import sys
import optparse

# Import third-party modules
import numpy
import pylab

# Import custom modules
import client.result
import client.job.idealized

parser = optparse.OptionParser()
parser.add_option('--database', type='string', help='Path to database (SQLite only)')
parser.add_option('-r', '--range', type='float', help='Lower boundary for relative ln likelihood (always < 0)')
parser.add_option('--bincount', type='int', help='Number of bins for ln likelihood marginals')
parser.add_option('-g', '--groupby', type='choice', choices=('source','run'), help='What identifier to group the results by, i.e., "source" or "run".')
parser.add_option('-o', '--orderby', type='choice', choices=('count','lnl'), help='What property to order the result groups by, i.e., "count" or "lnl".')
parser.add_option('--maxcount', type='int', help='Maximum number of series to plot')
parser.add_option('--constraint', type='string', action='append',nargs=3,help='Constraint on parameter (parameter name, minimum, maximum)',dest='constraints')
parser.add_option('-l', '--limit', type='int', help='Maximum number of results to read')
parser.add_option('--run', type='int', help='Run number')
parser.add_option('--start', type='int', help='Index of first frame to generate')
parser.add_option('--stop', type='int', help='Index of last frame to generate')
parser.add_option('--stride', type='int', help='Stride in number of frames')
parser.add_option('-n', type='int', help='Number of points in active parameter set.')
parser.add_option('-s', '--scatter', action='store_true', help='Use scatter plot in 2d, with points according to likelihood value.')
parser.set_defaults(range=None, bincount=25, orderby='count', maxcount=None, groupby='run', constraints=[], limit=-1, run=None, database=None, scenarios=None, scatter=False, n=20, start=0, stop=-1, stride=1)
(options, args) = parser.parse_args()

if len(args) < 1:
    print 'This script takes the path to the job configuration file (xml) as first argument.'
    sys.exit(2)

marginal = True

parbounds = dict([(name, (minimum, maximum)) for name, minimum, maximum in options.constraints])

result = client.result.Result(args[0], database=options.database)

if options.range is not None and options.range > 0:
    options.range = -options.range

parnames = result.job.getParameterNames()
parmin, parmax = result.job.getParameterBounds()
parlog = result.job.getParameterLogScale()
parrange = parmax-parmin

if len(args) < 3:
    print 'This script takes three required arguments: path to job configuration file (xml), parameter on x-axis, parameter on y-axis. The following parameters are available: %s' % ', '.join(parnames)
    sys.exit(2)

ix = parnames.index(args[1])
iy = parnames.index(args[2])
print 'x axis: %s (parameter %i)' % (args[1], ix)
print 'y axis: %s (parameter %i)' % (args[2], iy)

def update(fig=None):
    res, source2history, run2source = result.get(groupby=options.groupby, constraints=parbounds, run_id=options.run, limit=options.limit)

    maxlnl = res[:, -1].max()
    res[:, -1] -= maxlnl
    minlnl = res[:, -1].min()

    # Show best parameter set
    if marginal:
        pylab.figure(figsize=(8, 8))
        ax_main = pylab.subplot(2, 2, 2)
    else:
        pylab.figure(figsize=(8, 8))
        ax_main = pylab.gca()

    xs = numpy.linspace(parmin[ix]-parrange[ix]/10, parmax[ix]+parrange[ix]/10, 100)
    xs_c = 0.5*(xs[:-1]+xs[1:])
    ys = numpy.linspace(parmin[iy]-parrange[iy]/10, parmax[iy]+parrange[iy]/10, 100)
    ys_c = 0.5*(ys[:-1]+ys[1:])

    marg1_func, marg2_func = None, None 
    if isinstance(result.job, client.job.idealized.Job) and not options.scatter:
        zs = result.job.evaluateFitness((xs_c[:, numpy.newaxis], ys_c[numpy.newaxis, :]))
        marg1_func = lambda x: numpy.exp(-0.5*(x**2))
        marg2_func = lambda x: numpy.exp(-0.5*(x**2))
        pc = pylab.contourf(xs_c, ys_c, zs, 100)
    if options.scatter:
        vmin, vmax = res[:, -1].min(), res[:, -1].max()
        series_old = pylab.scatter(res[:, ix], res[:, iy], c=res[:, -1], vmin=vmin, vmax=vmax)
        series_new = pylab.scatter(res[:, ix], res[:, iy], c=res[:, -1], vmin=vmin, vmax=vmax)
    else:
        series_old, = pylab.plot(res[:, ix], res[:, iy], 'o', mfc='none', mec='k')
        series_new, = pylab.plot(res[:, ix], res[:, iy], 'ow', mec='k')
    pylab.grid()
    pylab.xlim(parmin[ix], parmax[ix])
    pylab.ylim(parmin[iy], parmax[iy])
    if parlog[ix]:
        ax_main.set_xscale('log')
    if parlog[iy]:
        ax_main.set_yscale('log')
    pylab.xticks(pylab.xlim())
    pylab.yticks(pylab.ylim())
    if marginal:
        ax = pylab.subplot(2, 2, 1)
        pylab.grid()
        pylab.xlim(0., minlnl)
        pylab.ylim(parmin[iy], parmax[iy])
        if parlog[iy]:
            ax.set_yscale('log')
        pylab.xticks(())
        pylab.yticks(())
        series_marg2, = pylab.plot(res[:, -1], res[:, iy], 'ok')
        if marg2_func is not None:
            pylab.plot(marg1_func(xs_c)-maxlnl, xs_c, '-k')

        ax = pylab.subplot(2, 2, 4)
        pylab.grid()
        pylab.xlim(parmin[ix], parmax[ix])
        if parlog[ix]:
            ax.set_xscale('log')
        pylab.ylim(0., minlnl)
        pylab.xticks(())
        pylab.yticks(())
        series_marg1, = pylab.plot(res[:, ix], res[:, -1], 'ok')
        if marg1_func is not None:
            pylab.plot(xs_c, marg1_func(xs_c)-maxlnl, '-k')
    for i in range(options.start%res.shape[0], (options.stop%res.shape[0])+1, options.stride):
        n = max(0, i-options.n)
        if options.scatter:
            series_old.remove()
            series_new.remove()
            series_old = ax_main.scatter(res[:n, ix], res[:n, iy], c=res[:n, -1], vmin=vmin, vmax=vmax, edgecolors='face')
            series_new = ax_main.scatter(res[n:i, ix], res[n:i, iy], c=res[n:i, -1], vmin=vmin, vmax=vmax)
        else:
            series_old.set_data(res[:n, ix], res[:n, iy])
            series_new.set_data(res[n:i, ix], res[n:i, iy])
        if marginal:
            series_marg2.set_data(res[:i, -1], res[:i, iy])
            series_marg1.set_data(res[:i, ix], res[:i, -1])
        path = '%i.png' % i
        pylab.savefig(path, dpi=96)
        print 'Saved %s.' % path

update()
