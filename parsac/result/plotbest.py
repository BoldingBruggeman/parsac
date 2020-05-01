#!/usr/bin/env python
from __future__ import print_function

def configure_argument_parser(parser):
    parser.add_argument('xmlfile',       type=str, help='XML formatted configuration file')
    parser.add_argument('-r', '--rank',  type=int,   help='Rank of the result to plot (default = 1, i.e., the very best result)')
    parser.add_argument('-d', '--depth', type=float, help='Depth range to show (> 0)')
    parser.add_argument('-g', '--grid',  action='store_true', help='Whether to grid the observations.')
    parser.add_argument('--savenc',      type=str, help='Path to copy NetCDF output file to.')
    parser.add_argument('--simulationdir',type=str, help='Directory to run simulation in.')
    parser.set_defaults(rank=1, depth=None, grid=False, savenc=None, simulationdir=None)

def main(args):
    import sys

    import numpy
    import pylab
    import matplotlib.cm
    import matplotlib.gridspec
    import matplotlib.colorbar

    from .. import result

    if args.grid:
        try:
            import scipy.interpolate
        except ImportError as e:
            print('Failed to import scipy.interpolate needed for --grid/-g. Error: %s' % e)
            sys.exit(1)

    if args.depth is not None and args.depth < 0:
        print('Depth argument must be positive, but is %.6g.' % args.depth)
        sys.exit(2)

    extravars = ()
    #extravars = (('nuh',True),)
    #extravars = [('mean_1',False),('mean_2',False),('var_1_1',False),('var_2_2',False),('cor_2_1',False)]
    #extravars = (('phytosize_mean_om',False),('phytosize_var_om',False))

    current_result = result.Result(args.xmlfile, simulationdir=args.simulationdir)

    parameters, lnl = current_result.get_best(args.rank)

    # Show best parameter set
    print('%ith best parameter set:' % args.rank)
    for name, value in zip(current_result.job.getParameterNames(), parameters):
        print('  %s = %.6g' % (name, value))
    print('Original ln likelihood = %.8g' % lnl)

    # Initialize the job (needed to load observations)
    current_result.job.start()

    # Build a list of all NetCDF variables that we want model results for.
    obsinfo = current_result.job.observations
    outputvars = [oi['outputvariable'] for oi in obsinfo]
    for vardata in extravars:
        if isinstance(vardata, (str, u''.__class__)):
            outputvars.append(vardata)
        else:
            outputvars.append(vardata[0])

    # Run and retrieve results.
    #returncode = current_result.job.controller.run(parameters,showoutput=True)
    #if returncode!=0:
    #    print('GOTM run failed - exiting.')
    #    sys.exit(1)
    #nc = NetCDFFile(ncpath,'r')
    #res = current_result.job.controller.getNetCDFVariables(nc,outputvars,addcoordinates=True)
    #nc.close()
    likelihood, model_values = current_result.job.evaluate2(parameters, return_model_values=True, show_output=True)
    print('Newly calculated ln likelihood = %.8g. Original value was %.8g.' % (likelihood, lnl))

    # # Copy NetCDF file
    # if args.savenc is not None:
    #     print('Saving NetCDF output to %s...' % args.savenc, end='')
    #     shutil.copyfile(ncpath,args.savenc)
    #     fout = open('%s.info' % args.savenc,'w')
    #     fout.write('job %i, %ith best parameter set\n' % (current_result.job.id,args.rank))
    #     fout.write('%s\n' % datetime.datetime.today().isoformat())
    #     fout.write('parameter values:\n')
    #     for i,val in enumerate(parameters_utf):
    #         pi = current_result.job.controller.externalparameters[i]
    #         fout.write('  %s = %.6g\n' % (pi['name'],val))
    #     fout.write('ln likelihood = %.8g\n' % lnl)
    #     fout.close()
    #     print(' done')

    hres = pylab.figure(figsize=(10, 9))
    herr = pylab.figure()
    hcor = pylab.figure(figsize=(2.5, 9))
    nrow = int(round(numpy.sqrt(len(obsinfo))))
    ncol = int(numpy.ceil(len(obsinfo)/float(nrow)))
    gs = matplotlib.gridspec.GridSpec(len(obsinfo), 11)
    for i, (oi, moddat) in enumerate(zip(obsinfo, model_values)):
        times, observed_values, zs = oi['times'], oi['values'], oi['zs']

        if zs is None:
            # Time series of scalar (dimensions: time)
            t_centers, all_model_data, model_data = moddat
        else:
            # Time series of vertically structured variable (dimensions: time, z)
            t_interfaces, z_interfaces, all_model_data, model_data = moddat

        modelmin = oi.get('modelminimum', None)
        if modelmin is not None:
            all_model_data[all_model_data < modelmin] = modelmin
            model_data[model_data < modelmin] = modelmin

        # If we do a relative fit, scale the model result to best match observations.
        if oi['relativefit']:
            if (model_data == 0.).all():
                print('ERROR: cannot calculate optimal scaling factor for %s because all model values equal zero.' % oi['outputvariable'])
                sys.exit(1)
            scale = (observed_values*model_data).sum()/(model_data**2).sum()
            print('Optimal model-to-observation scaling factor for %s = %.6g.' % (oi['outputvariable'], scale))
            model_data *= scale
            all_model_data *= scale
        elif oi['fixed_scale_factor'] is not None:
            model_data *= oi['fixed_scale_factor']
            all_model_data *= oi['fixed_scale_factor']

        # Create figure for model-data comparison
        pylab.figure(hres.number)

        # Plot model result
        if zs is not None:
            zmin, zmax = z_interfaces[:, 0].min(), z_interfaces[:, -1].max()
            z_centers = (z_interfaces[:-1, :-1] + z_interfaces[1:, :-1] + z_interfaces[:-1, 1:] + z_interfaces[1:, 1:])/4
            if args.depth is not None:
                zmin = -args.depth

            valid_obs = zs > zmin
            valid_mod = z_centers > zmin
            varrange = (min(all_model_data[valid_mod].min(), observed_values[valid_obs].min()), max(all_model_data[valid_mod].max(), observed_values[valid_obs].max()))
            #print(varrange)

            t_interfaces = pylab.date2num(t_interfaces)
            ax = pylab.subplot(gs[i, 0:5])
            pc = pylab.pcolormesh(t_interfaces, z_interfaces, all_model_data, vmin=varrange[0], vmax=varrange[1])
            pylab.ylim(zmin, zmax)
            pylab.xlim(t_interfaces[0, 0], t_interfaces[-1, 0])
            loc = matplotlib.dates.AutoDateLocator()
            ax.xaxis.set_major_formatter(matplotlib.dates.AutoDateFormatter(loc))
            ax.xaxis.set_major_locator(loc)
            pylab.grid(True)
            cb = pylab.colorbar(pc, cax=pylab.subplot(gs[i, -1]))
            cb.set_label(oi['outputvariable'])

            # Plot observations
            ax = pylab.subplot(gs[i, 5:-1])
            numtimes = pylab.date2num(times)
            if args.grid:
                gridded_observed_values = scipy.interpolate.griddata((numtimes, zs), observed_values, (t_interfaces, z_interfaces), 'linear')
                pylab.pcolormesh(t_interfaces, z_interfaces, gridded_observed_values, vmin=varrange[0], vmax=varrange[1])
            else:
                pylab.scatter(numtimes, zs, s=10, c=observed_values, vmin=varrange[0], vmax=varrange[1], edgecolors='none')
            pylab.ylim(zmin, zmax)
            pylab.xlim(t_interfaces[0, 0], t_interfaces[-1, 0])
            loc = matplotlib.dates.AutoDateLocator()
            ax.xaxis.set_major_formatter(matplotlib.dates.AutoDateFormatter(loc))
            ax.xaxis.set_major_locator(loc)
            pylab.gca().set_yticklabels(())
            pylab.grid(True)
        else:
            ax = pylab.subplot(gs[i, :])
            pylab.plot(pylab.date2num(t_centers), all_model_data, '-')
            pylab.plot(pylab.date2num(times), observed_values, '.')
            loc = matplotlib.dates.AutoDateLocator()
            ax.xaxis.set_major_formatter(matplotlib.dates.AutoDateFormatter(loc))
            ax.xaxis.set_major_locator(loc)
            pylab.ylabel(oi['outputvariable'])
            pylab.grid(True)

        # Plot model predictions vs. observations
        pylab.figure(hcor.number)
        #pylab.subplot(nrow,ncol,i+1)
        pylab.subplot(len(obsinfo), 1, i+1)
        pylab.plot(observed_values, model_data, '.')
        pylab.grid(True)
        mi, ma = min(observed_values.min(), model_data.min()), max(observed_values.max(), model_data.max())
        pylab.xlim(mi, ma)
        pylab.ylim(mi, ma)
        pylab.plot((mi, ma), (mi, ma), '-k')
        pylab.xlabel('observation')
        pylab.ylabel('model')

        # Plot histogram with errors.
        pylab.figure(herr.number)
        pylab.subplot(len(obsinfo), 1, i+1)
        diff = model_data-observed_values
        var_obs = ((observed_values-observed_values.mean())**2).mean()
        var_mod = ((model_data-model_data.mean())**2).mean()
        cov = ((observed_values-observed_values.mean())*(model_data-model_data.mean())).mean()
        cor = cov/numpy.sqrt(var_obs*var_mod)
        bias = (model_data-observed_values).mean()
        mae = numpy.mean(numpy.abs(diff))
        rmse = numpy.sqrt(numpy.mean(diff**2))
        #pylab.plot(diff,obs[:,1],'o')
        #pylab.figure()
        n, bins, patches = pylab.hist(diff, 100, density=True)
        pylab.xlabel('model - observation')
        print('%s:\n- bias: %.4g\n- mean absolute error = %.4g\n- rmse = %.4g\n- cor = %.4g\n- s.d. mod = %.4g\n- s.d. obs = %.4g' % (oi['outputvariable'], bias, mae, rmse, cor, numpy.sqrt(var_mod), numpy.sqrt(var_obs)))
        y = (1. / numpy.sqrt(2 * numpy.pi) / rmse) * numpy.exp((-0.5 / rmse**2) * bins**2)
        l = pylab.plot(bins, y, 'r--', linewidth=2)

    if len(extravars) > 0:
        pylab.figure()
        varcount = float(len(extravars))
        rowcount = int(numpy.ceil(numpy.sqrt(varcount)))
        colcount = int(numpy.ceil(varcount/rowcount))
        for i, vardata in enumerate(extravars):
            if isinstance(vardata, (str, u''.__class__)):
                varname = vardata
                logscale = False
            else:
                varname = vardata[0]
                logscale = vardata[1]
            modeldata = res[varname]
            if logscale:
                modeldata = numpy.log10(modeldata)
            pylab.subplot(rowcount, colcount, i+1)
            pylab.pcolormesh(tim_stag, z_stag, modeldata.T)
            pylab.ylim(-viewdepth, 0)
            pylab.colorbar()
            xax = pylab.gca().xaxis
            loc = matplotlib.dates.AutoDateLocator()
            xax.set_major_formatter(matplotlib.dates.AutoDateFormatter(loc))
            xax.set_major_locator(loc)
            pylab.grid(True)

    pylab.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
