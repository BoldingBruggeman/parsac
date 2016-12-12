#!/usr/bin/env python

import re,os.path
from Scientific.IO.NetCDF import NetCDFFile
import matplotlib.pylab,numpy
import acpy.gotmcontroller,acpy.run,acpy.optimizer

jobid = 25
scenpath = os.path.join('acpy','./scenarios/%i' % jobid)
obsdir = os.path.join('acpy',acpy.run.obsdir)

nmlfile = os.path.join(scenpath,'gotmrun.nml')

for year in range(2000,2006):
    print 'Running year %04i' % year
    nmls,nmlorder = acpy.gotmcontroller.parseNamelistFile(nmlfile)
    nmls['time']['start'] = '"%04i-01-01 00:00:00"' % year
    nmls['time']['stop'] = '"%04i-01-01 00:00:00"' % (year+1)
    acpy.gotmcontroller.writeNamelistFile(nmlfile,nmls,nmlorder)

    job = acpy.optimizer.Job(jobid,
                        scenpath,
                        gotmexe=acpy.run.exe,
                        transports=(acpy.transport.Dummy(),),
                        copyexe=True)

    job.verbose = False
    job.addObservation(os.path.join(obsdir,'pottemp.obs'),'temp',maxdepth=310,cache=False)
    job.addObservation(os.path.join(obsdir,'salt.obs'),'salt',maxdepth=310,cache=False)
    job.verbose = True

    job.initialize()

    obsinfo = job.observations
    outputvars = [oi['outputvariable'] for oi in obsinfo]

    # Run and retrieve results.
    ncpath = job.controller.run((),showoutput=False,returnncpath=True)
    if ncpath==None:
        print 'GOTM run failed - exiting.'
        sys.exit(1)
    nc = NetCDFFile(ncpath,'r')
    res = acpy.run.job.controller.getNetCDFVariables(nc,outputvars,addcoordinates=True)
    nc.close()

    # Shortcuts to coordinates
    tim_cent,z_cent,z1_cent = res['time_center'],res['z_center'],res['z1_center']
    tim_stag,z_stag,z1_stag = res['time_staggered'],res['z_staggered'],res['z1_staggered']

    # Find the depth index from where we start
    ifirstz = z_cent.searchsorted(-300)
    viewdepth = 300

    hres = matplotlib.pylab.figure()
    herr = matplotlib.pylab.figure()
    for i,oi in enumerate(obsinfo):
        modeldata = res[oi['outputvariable']]
        obsdata = oi['observeddata']

        modelmin = oi.get('modelminimum',None)
        if modelmin!=None: modeldata[modeldata<modelmin] = modelmin

        # Calculate model predictions on observation coordinates.
        pred = acpy.gotmcontroller.interp2(tim_cent,z_cent,modeldata,obsdata[:,0],obsdata[:,1])

        # If we do a relative fit, scale the model result to best match observations.
        if oi['relativefit']:
            if (pred==0.).all():
                print 'ERROR: cannot calculate optimal scaling factor for %s because all model values equal zero.' % oi['outputvariable']
                sys.exit(1)
            scale = sum(obsdata[:,2]*pred)/sum(pred*pred)
            print 'Optimal model-to-observation scaling factor for %s = %.6g.' % (oi['outputvariable'],scale)
            pred *= scale
            modeldata *= scale

        varrange = (min(modeldata[:,ifirstz:].min(),obsdata[:,2].min()),max(modeldata[:,ifirstz:].max(),obsdata[:,2].max()))

        diff = pred-obsdata[:,2]
        print '%s: mean absolute error = %.4g, s.d. = %.4g.' % (oi['outputvariable'],numpy.mean(numpy.abs(diff)),numpy.sqrt(numpy.mean(diff**2)))

        if False:
            # Create figure for model-data comparison
            matplotlib.pylab.figure(hres.number)

            # Plot model result
            matplotlib.pylab.subplot(len(obsinfo),2,i*2+1)
            matplotlib.pylab.pcolor(tim_stag,z_stag,modeldata.transpose())
            matplotlib.pylab.clim(varrange)
            matplotlib.pylab.ylim(-viewdepth,0)
            matplotlib.pylab.colorbar()
            matplotlib.pylab.xlim(tim_stag[0],tim_stag[-1])
            xax = matplotlib.pylab.gca().xaxis
            loc = matplotlib.dates.AutoDateLocator()
            xax.set_major_formatter(matplotlib.dates.AutoDateFormatter(loc))
            xax.set_major_locator(loc)
            matplotlib.pylab.grid(True)

            # Plot observations
            matplotlib.pylab.subplot(len(obsinfo),2,i*2+2)
            matplotlib.pylab.scatter(obsdata[:,0],obsdata[:,1],s=10,c=obsdata[:,2],cmap=matplotlib.cm.jet,vmin=varrange[0],vmax=varrange[1],faceted=False)
            matplotlib.pylab.ylim(-viewdepth,0)
            matplotlib.pylab.xlim(tim_stag[0],tim_stag[-1])
            matplotlib.pylab.clim(varrange)
            xax = matplotlib.pylab.gca().xaxis
            loc = matplotlib.dates.AutoDateLocator()
            xax.set_major_formatter(matplotlib.dates.AutoDateFormatter(loc))
            xax.set_major_locator(loc)
            matplotlib.pylab.grid(True)

            # Plot histogram with errors.
            matplotlib.pylab.figure(herr.number)
            matplotlib.pylab.subplot(len(obsinfo),1,i+1)
            #matplotlib.pylab.plot(diff,obs[:,1],'o')
            #matplotlib.pylab.figure()
            n, bins, patches = matplotlib.pylab.hist(diff, 100, normed=1)
            #y = matplotlib.pylab.normpdf(bins, 0., numpy.sqrt(ssq/len(diff)))
            #l = matplotlib.pylab.plot(bins, y, 'r--', linewidth=2)

if False: matplotlib.pylab.show()



    


