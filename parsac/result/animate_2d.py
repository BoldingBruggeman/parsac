#!/usr/bin/env python
from __future__ import print_function

def configure_argument_parser(parser):
    parser.add_argument('xmlfile',       type=str, help='XML formatted configuration file')
    parser.add_argument('x',       type=str, help='Parameter on X-axis')
    parser.add_argument('y',       type=str, help='Parameter on Y-axis')
    parser.add_argument('--constraint', type=str, action='append',nargs=3,help='Constraint on parameter (parameter name, minimum, maximum)',dest='constraints')
    parser.add_argument('-l', '--limit', type=int, help='Maximum number of results to read')
    parser.add_argument('--run', type=int, help='Run number')
    parser.add_argument('--start', type=int, help='Index of first frame to generate')
    parser.add_argument('--stop', type=int, help='Index of last frame to generate')
    parser.add_argument('--stride', type=int, help='Stride in number of frames')
    parser.add_argument('-n', type=int, help='Number of points in active parameter set.')
    parser.add_argument('-s', '--scatter', action='store_true', help='Use scatter plot in 2d, with points according to likelihood value.')
    parser.set_defaults(constraints=[], limit=-1, run=None, scatter=False, n=20, start=0, stop=-1, stride=1)

def main(args):
    # Import third-party modules
    import numpy
    import pylab

    # Import custom modules
    from .. import result
    from .. import job

    marginal = True

    parbounds = dict([(name, (minimum, maximum)) for name, minimum, maximum in args.constraints])

    current_result = result.Result(args.xmlfile)

    parnames = current_result.job.getParameterNames()
    parmin, parmax = current_result.job.getParameterBounds()
    parlog = current_result.job.getParameterLogScale()
    parrange = parmax-parmin

    ix = parnames.index(args.x)
    iy = parnames.index(args.y)
    print('x axis: %s (parameter %i)' % (args.x, ix))
    print('y axis: %s (parameter %i)' % (args.y, iy))

    def update(fig=None):
        res = current_result.get(constraints=parbounds, run_id=args.run, limit=args.limit)

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
        if isinstance(current_result.job, job.idealized.Job) and not args.scatter:
            zs = current_result.job.evaluate((xs_c[:, numpy.newaxis], ys_c[numpy.newaxis, :]))
            marg1_func = lambda x: numpy.exp(-0.5*(x**2))
            marg2_func = lambda x: numpy.exp(-0.5*(x**2))
            pc = pylab.contourf(xs_c, ys_c, zs, 100)
        if args.scatter:
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
        for i in range(args.start%res.shape[0], (args.stop%res.shape[0])+1, args.stride):
            n = max(0, i-args.n)
            if args.scatter:
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
            print('Saved %s.' % path)

    update()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)

