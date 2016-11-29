#!/usr/bin/env python

# Import from standard Python library
import sys
import optparse

# Import third-party modules
import numpy

# Try importing MatPlotLib
try:
    import pylab
except ImportError:
    pylab = None

# Import custom modules
import client.result

parser = optparse.OptionParser()
parser.add_option('--database', type='string', help='Path to database (SQLite only)')
parser.add_option('-r', '--range', type='float', help='Lower boundary for relative ln likelihood (always < 0)')
parser.add_option('--bincount', type='int', help='Minimum number of points per ln likelihood marginal')
parser.add_option('-g', '--groupby', type='choice', choices=('source', 'run'), help='What identifier to group the results by, i.e., "source" or "run".')
parser.add_option('-o', '--orderby', type='choice', choices=('count', 'lnl'), help='What property to order the result groups by, i.e., "count" or "lnl".')
parser.add_option('--maxcount', type='int', help='Maximum number of series to plot')
parser.add_option('--constraint', type='string', action='append', nargs=3, help='Constraint on parameter (parameter name, minimum, maximum)', dest='constraints')
parser.add_option('-l', '--limit', type='int', help='Maximum number of results to read')
parser.add_option('--run', type='int', help='Run number')
parser.add_option('-u', '--update', action='store_true', help='Keep running and updating the figure with new results until the user quits with Ctrl-C')
parser.set_defaults(range=None, bincount=25, orderby='count', maxcount=None, groupby='run', constraints=[], limit=-1, run=None, database=None, scenarios=None, update=False)
options, args = parser.parse_args()

if len(args) < 1:
    print 'This script takes one required argument: path to job configuration file (xml).'
    sys.exit(2)

if options.range is not None and options.range > 0:
    options.range = -options.range

parbounds = dict([(name, (minimum, maximum)) for name, minimum, maximum in options.constraints])

result = client.result.Result(args[0], database=options.database)

parnames = result.job.getParameterNames()
parmin, parmax = result.job.getParameterBounds()
parlog = result.job.getParameterLogScale()
npar = len(parnames)

def update(fig=None):
    res, source2history = result.get(groupby=options.groupby, constraints=parbounds, run_id=options.run, limit=options.limit)
    run2source = result.get_sources()

    # Sort by likelihood
    indices = res[:, -1].argsort()
    res = res[indices, :]

    # Show best parameter set
    maxlnl = res[-1, -1]
    minlnl = res[0, -1]
    iinc = res[:, -1].searchsorted(maxlnl-1.92)
    lbounds, rbounds = res[iinc:, :-1].min(axis=0), res[iinc:, :-1].max(axis=0)
    best = res[-1, :-1]
    outside = res[:iinc, :-1]
    print 'Best parameter set is # %i with ln likelihood = %.6g:' % (indices[-1], maxlnl)
    for ipar, parname in enumerate(parnames):
        # Get conservative confidence interval by extending it to the first point
        # from the boundary that has a likelihood value outside the allowed range.
        lvalid = outside[:, ipar] < lbounds[ipar]
        rvalid = outside[:, ipar] > rbounds[ipar]
        if lvalid.any():
            lbounds[ipar] = outside[lvalid, ipar].max()
        if rvalid.any():
            rbounds[ipar] = outside[rvalid, ipar].min()

        # Report estimate and confidence interval
        print '  %s = %.6g (%.6g - %.6g)' % (parname, best[ipar], lbounds[ipar], rbounds[ipar])

    # Create parameter bins for histogram
    parbinbounds = numpy.empty((npar, options.bincount+1))
    parbins = numpy.empty((npar, options.bincount))
    parbins[:, :] = 1.1*(minlnl-maxlnl)
    for ipar, (minimum, maximum, logscale) in enumerate(zip(parmin, parmax, parlog)):
        if logscale:
            parbinbounds[ipar, :] = numpy.logspace(numpy.log10(minimum), numpy.log10(maximum), options.bincount+1)
        else:
            parbinbounds[ipar, :] = numpy.linspace(minimum, maximum, options.bincount+1)

    group2maxlnl = dict([(s,curres[:, -1].max()) for s, curres in source2history.items()])

    # Order sources (runs or clients) according to counts or ln likelihood.
    if options.groupby is not None:
        print 'Points per %s:' % options.groupby
        sources = source2history.keys()
        if options.orderby == 'count':
            sources = sorted(sources, cmp=lambda x, y: cmp(len(source2history[y]), len(source2history[x])))
        else:
            sources = sorted(sources, cmp=lambda x, y: cmp(group2maxlnl[y], group2maxlnl[x]))
        if options.maxcount is not None and len(sources) > options.maxcount:
            sources[options.maxcount:] = []
        for source in sources:
            dat = source2history[source]
            label = source
            if options.groupby == 'run':
                label = '%s (%s)' % (source, run2source[source])
            print '  %s: %i points, best lnl = %.8g.' % (label, len(dat), group2maxlnl[source])

    if pylab is None:
        print 'MatPlotLib not found - skipping plotting.'
        return

    nrow = int(numpy.ceil(numpy.sqrt(0.5*npar)))
    ncol = int(numpy.ceil(npar/nrow))

    # Create the figure
    if fig is not None:
        pylab.figure(fig.number)
        for ipar in range(npar):
            pylab.subplot(nrow, ncol, ipar+1)
            pylab.cla()
    else:
        fig = pylab.figure(figsize=(12, 8))
        pylab.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.05, hspace=.3)
        if options.update:
            fig.canvas.mpl_connect('close_event', lambda evt: sys.exit(0))

    # Create subplots
    for ipar in range(npar):
        pylab.subplot(nrow, ncol, ipar+1)
        pylab.hold(True)

    # Approximate marginal by estimsting upper contour of cloud.
    marginals = []
    for ipar, (minimum, maximum, logscale) in enumerate(zip(parmin, parmax, parlog)):
        values = res[:, ipar]
        if logscale:
            values = numpy.log10(values)
        step = int(round(res.shape[0]/options.bincount))
        order = values.argsort()
        values, lnls = values[order], res[order, -1]
        xs, ys, i = [values[0]], [lnls[0]], 1
        while i < len(values):
            ilast = i + step
            slope = (lnls[i:ilast]-ys[-1])/(values[i:ilast]-xs[-1])
            i += slope.argmax()
            xs.append(values[i])
            ys.append(lnls[i])
            i += 1
            while i < len(values) and values[i] == xs[-1]:
                i += 1
        xs, ys = numpy.array(xs), numpy.array(ys)-maxlnl
        if logscale: xs = 10.**xs
        marginals.append((xs, ys))

    for source in sources:
        # Combine results for current source (run or client) into single array.
        curres = source2history[source]
        curres[:, -1] -= maxlnl

        # Determine label for current source.
        label = source
        if options.groupby == 'run':
            label = '%s (%s)' % (source, run2source[source])

        for ipar in range(npar):
            # Plot results for current source.
            pylab.subplot(nrow, ncol, ipar+1)
            pylab.plot(curres[:, ipar], curres[:, -1], '.', label=label)

            # Update histogram based on current source results.
            ind = parbinbounds[ipar, :].searchsorted(curres[:, ipar])-1
            for i, ibin in enumerate(ind):
                parbins[ipar, ibin] = max(parbins[ipar, ibin], curres[i, -1])

    # Put finishing touches on subplots
    for ipar, (name, minimum, maximum, logscale, lbound, rbound, marginal) in enumerate(zip(parnames, parmin, parmax, parlog, lbounds, rbounds, marginals)):
        ax = pylab.subplot(nrow, ncol, ipar+1)
        #pylab.legend(sources,numpoints=1)

        # Add title
        pylab.title(name)

        # Plot marginal
        pylab.hold(True)
        #x = numpy.concatenate((parbinbounds[ipar, 0:1], numpy.repeat(parbinbounds[ipar, 1:-1], 2), parbinbounds[ipar, -1:]), 0)
        #y = numpy.repeat(parbins[ipar, :], 2)
        #pylab.plot(x, y, '-k', label='_nolegend_')
        pylab.plot(marginal[0], marginal[1], '-k', label='_nolegend_')

        #pylab.plot(res[:,0],res[:,1],'o')

        # Set axes boundaries
        pylab.xlim(minimum, maximum)
        ymin = minlnl-maxlnl
        if options.range is not None:
            ymin = options.range
        pylab.ylim(ymin, 0)

        # Show confidence interval
        pylab.axvline(lbound, color='k', linestyle='--')
        pylab.axvline(rbound, color='k', linestyle='--')

        if logscale:
            ax.set_xscale('log')

    #pylab.legend(numpoints=1)
    if not options.update:
        pylab.savefig('estimates.png', dpi=300)

    # Show figure and wait until the user closes it.
    pylab.show()

    return fig

if options.update:
    if pylab is None:
        print 'MatPlotLib not found - cannot run in continuous update mode (-u/--update).'
        sys.exit(1)
    fig = None
    pylab.ion()
    count = result.count()
    while 1:
        fig = update(fig)
        print 'Waiting for new results...',
        while 1:
            pylab.pause(5.)
            newcount = result.count()
            if newcount != count:
                print '%i found.' % (newcount-count)
                count = newcount
                break
else:
    update()
