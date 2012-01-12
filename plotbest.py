#!/usr/bin/env python

import sys,math,optparse,shutil,datetime

# Import third-party modules
import numpy
import matplotlib.pylab,matplotlib.cm

import client.run,client.gotmcontroller
import mysqlinfo
from Scientific.IO.NetCDF import NetCDFFile

parser = optparse.OptionParser()
parser.add_option('-r', '--rank',  type='int',   help='Rank of the result to plot (default = 1, i.e., the very best result)')
parser.add_option('-d', '--depth', type='float', help='Depth range to show (> 0)')
parser.add_option('-g', '--grid',  action='store_true', help='Whether to grid the observations.')
parser.add_option('--savenc',      type='string', help='Path to copy NetCDF output file to.')
parser.set_defaults(rank=1,depth=None,grid=False,savenc=None)
(options, args) = parser.parse_args()

if len(args)<1:
    print 'No job identifier provided.'
    sys.exit(2)
jobid = int(args[0])

if options.depth!=None and options.depth<0:
    print 'Depth argument must be positive, but is %.6g.' % options.depth
    sys.exit(2)

extravars = ()
#extravars = (('nuh',True),)
#extravars = [('mean_1',False),('mean_2',False),('var_1_1',False),('var_2_2',False),('cor_2_1',False)]
#extravars = (('phytosize_mean_om',False),('phytosize_var_om',False))

# Connect to database and retrieve best parameter set.
db = mysqlinfo.connect(mysqlinfo.select)
c = db.cursor()
query = "SELECT `parameters`,`lnlikelihood` FROM `runs`,`results` WHERE `runs`.`id`=`results`.`run` AND `runs`.`job`='%i'" % jobid
c.execute(query)
res = [(strpars,lnlikelihood) for strpars,lnlikelihood in c if lnlikelihood!=None]

# Initialize the GOTM controller.
job = client.run.getJob(jobid)
job.controller.initialize()

res.sort(cmp=lambda x,y: cmp(x[1],y[1]))
parameters,lnl = res[-options.rank]
parameters = map(float,parameters.split(';'))


# Show best parameter set
print '%ith best parameter set:' % options.rank
parameters_utf = job.controller.untransformParameterValues(parameters)
for i,val in enumerate(parameters_utf):
    pi = job.controller.externalparameters[i]
    print '   %s = %.6g' % (pi['name'],val)
print 'ln likelihood = %.8g' % lnl

# Build a list of all NetCDF variables that we want model results for.
obsinfo = job.observations
outputvars = [oi['outputvariable'] for oi in obsinfo]
for vardata in extravars:
    if isinstance(vardata,basestring):
        outputvars.append(vardata)
    else:
        outputvars.append(vardata[0])

# Run and retrieve results.
ncpath = job.controller.run(parameters,showoutput=True,returnncpath=True)
if ncpath==None:
    print 'GOTM run failed - exiting.'
    sys.exit(1)
nc = NetCDFFile(ncpath,'r')
res = job.controller.getNetCDFVariables(nc,outputvars,addcoordinates=True)
nc.close()

# Copy NetCDF file
if options.savenc is not None:
    print 'Saving NetCDF output to %s...' % options.savenc,
    shutil.copyfile(ncpath,options.savenc)
    fout = open('%s.info' % options.savenc,'w')
    fout.write('job %i, %ith best parameter set\n' % (jobid,options.rank))
    fout.write('%s\n' % datetime.datetime.today().isoformat())
    fout.write('parameter values:\n')
    for i,val in enumerate(parameters_utf):
        pi = job.controller.externalparameters[i]
        fout.write('  %s = %.6g\n' % (pi['name'],val))
    fout.write('ln likelihood = %.8g\n' % lnl)
    fout.close()
    print ' done'

# Shortcuts to coordinates
tim_cent,z_cent,z1_cent = res['time_center'],res['z_center'],res['z1_center']
tim_stag,z_stag,z1_stag = res['time_staggered'],res['z_staggered'],res['z1_staggered']

# Find the depth index from where we start
ifirstz = 0
viewdepth = -z_stag[0]
if options.depth is not None:
    ifirstz = z_cent.searchsorted(-options.depth)
    viewdepth = options.depth

hres = matplotlib.pylab.figure(figsize=(10,9))
herr = matplotlib.pylab.figure()
hcor = matplotlib.pylab.figure(figsize=(2.5,9))
nrow = int(round(math.sqrt(len(obsinfo))))
ncol = int(math.ceil(len(obsinfo)/float(nrow)))
for i,oi in enumerate(obsinfo):
    modeldata = res[oi['outputvariable']]
    obsdata = oi['observeddata']

    modelmin = oi.get('modelminimum',None)
    if modelmin!=None: modeldata[modeldata<modelmin] = modelmin

    # Calculate model predictions on observation coordinates.
    pred = client.gotmcontroller.interp2(tim_cent,z_cent,modeldata,obsdata[:,0],obsdata[:,1])

    # If we do a relative fit, scale the model result to best match observations.
    if oi['relativefit']:
        if (pred==0.).all():
            print 'ERROR: cannot calculate optimal scaling factor for %s because all model values equal zero.' % oi['outputvariable']
            sys.exit(1)
        scale = sum(obsdata[:,2]*pred)/sum(pred*pred)
        print 'Optimal model-to-observation scaling factor for %s = %.6g.' % (oi['outputvariable'],scale)
        pred *= scale
        modeldata *= scale
    elif oi['fixed_scale_factor'] is not None:
        pred *= oi['fixed_scale_factor']
        modeldata *= oi['fixed_scale_factor']

    v = obsdata[:,1]>-viewdepth
    varrange = (min(modeldata[:,ifirstz:].min(),obsdata[v,2].min()),max(modeldata[:,ifirstz:].max(),obsdata[v,2].max()))

    # Create figure for model-data comparison
    matplotlib.pylab.figure(hres.number)

    # Plot model result
    matplotlib.pylab.subplot(len(obsinfo),2,i*2+1)
    matplotlib.pylab.pcolormesh(tim_stag,z_stag,modeldata.transpose())
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
    if options.grid:
        dt,dz,eps = 30,20,1e-6
        tgrid = numpy.arange(min(obsdata[:,0])-eps,max(obsdata[:,0]+eps),dt)
        zgrid = numpy.arange(min(obsdata[:,1])-eps,max(obsdata[:,1]+eps),dz)
        griddedobs = numpy.zeros((len(zgrid),len(tgrid)),dtype=numpy.float)
        counts = numpy.zeros((len(zgrid),len(tgrid)),dtype=numpy.int)
        for iobs in range(obsdata.shape[0]):
            it = tgrid.searchsorted(obsdata[iobs,0])-1
            iz = zgrid.searchsorted(obsdata[iobs,1])-1
            griddedobs[iz,it] += obsdata[iobs,2]
            counts[iz,it] += 1
        matplotlib.pylab.pcolormesh(tgrid,zgrid,numpy.ma.array(griddedobs/counts,mask=(counts==0)))
    else:
        matplotlib.pylab.scatter(obsdata[:,0],obsdata[:,1],s=10,c=obsdata[:,2],cmap=matplotlib.cm.jet,vmin=varrange[0],vmax=varrange[1],faceted=False)
    matplotlib.pylab.ylim(-viewdepth,0)
    matplotlib.pylab.xlim(tim_stag[0],tim_stag[-1])
    matplotlib.pylab.clim(varrange)
    xax = matplotlib.pylab.gca().xaxis
    loc = matplotlib.dates.AutoDateLocator()
    xax.set_major_formatter(matplotlib.dates.AutoDateFormatter(loc))
    xax.set_major_locator(loc)
    matplotlib.pylab.grid(True)

    # Plot model predictions vs. observations
    matplotlib.pylab.figure(hcor.number)
    #matplotlib.pylab.subplot(nrow,ncol,i+1)
    matplotlib.pylab.subplot(len(obsinfo),1,i+1)
    matplotlib.pylab.plot(obsdata[:,2],pred,'.')
    matplotlib.pylab.grid(True)
    mi,ma = min(obsdata[:,2].min(),pred.min()),max(obsdata[:,2].max(),pred.max())
    matplotlib.pylab.xlim(mi,ma)
    matplotlib.pylab.ylim(mi,ma)
    matplotlib.pylab.hold(True)
    matplotlib.pylab.plot((mi,ma),(mi,ma),'-k')

    # Plot histogram with errors.
    matplotlib.pylab.figure(herr.number)
    matplotlib.pylab.subplot(len(obsinfo),1,i+1)
    diff = pred-obsdata[:,2]
    var_obs = ((obsdata[:,2]-obsdata[:,2].mean())**2).mean()
    var_mod = ((pred-pred.mean())**2).mean()
    cov = ((obsdata[:,2]-obsdata[:,2].mean())*(pred-pred.mean())).mean()
    cor = cov/numpy.sqrt(var_obs*var_mod)
    #matplotlib.pylab.plot(diff,obs[:,1],'o')
    #matplotlib.pylab.figure()
    n, bins, patches = matplotlib.pylab.hist(diff, 100, normed=1)
    print '%s: mean absolute error = %.4g, rmse = %.4g., cor = %.4g, s.d. mod = %.4g, s.d. obs = %.4g' % (oi['outputvariable'],numpy.mean(numpy.abs(diff)),numpy.sqrt(numpy.mean(diff**2)),cor,numpy.sqrt(var_mod),numpy.sqrt(var_obs))
    #y = matplotlib.pylab.normpdf(bins, 0., numpy.sqrt(ssq/len(diff)))
    #l = matplotlib.pylab.plot(bins, y, 'r--', linewidth=2)

if len(extravars)>0:
    matplotlib.pylab.figure()
    varcount = float(len(extravars))
    rowcount = int(math.ceil(math.sqrt(varcount)))
    colcount = int(math.ceil(varcount/rowcount))
    for i,vardata in enumerate(extravars):
        if isinstance(vardata,basestring):
            varname = vardata
            logscale = False
        else:
            varname = vardata[0]
            logscale = vardata[1]
        modeldata = res[varname]
        if logscale: modeldata = numpy.log10(modeldata)
        matplotlib.pylab.subplot(rowcount,colcount,i+1)
        matplotlib.pylab.pcolormesh(tim_stag,z_stag,modeldata.transpose())
        matplotlib.pylab.ylim(-viewdepth,0)
        matplotlib.pylab.colorbar()
        xax = matplotlib.pylab.gca().xaxis
        loc = matplotlib.dates.AutoDateLocator()
        xax.set_major_formatter(matplotlib.dates.AutoDateFormatter(loc))
        xax.set_major_locator(loc)
        matplotlib.pylab.grid(True)

matplotlib.pylab.show()
