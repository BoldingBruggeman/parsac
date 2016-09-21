#!/usr/bin/env python

# Import from standard Python library
import sys,math,optparse,socket,pickle,os

# Import third-party modules
import numpy

# Import custom modules
import client.run

parser = optparse.OptionParser()
parser.add_option('--database',type='string',help='Path to database (SQLite only)')
parser.add_option('-r', '--range', type='float', help='Lower boundary for relative ln likelihood (always < 0)')
parser.add_option('--bincount', type='int', help='Number of bins for ln likelihood marginals')
parser.add_option('-g','--groupby',type='choice',choices=('source','run'),help='What identifier to group the results by, i.e., "source" or "run".')
parser.add_option('-o','--orderby',type='choice',choices=('count','lnl'),help='What property to order the result groups by, i.e., "count" or "lnl".')
parser.add_option('--maxcount',type='int',help='Maximum number of series to plot')
parser.add_option('--constraint',type='string',action='append',nargs=3,help='Constraint on parameter (parameter name, minimum, maximum)',dest='constraints')
parser.add_option('-l', '--limit', type='int', help='Maximum number of results to read')
parser.add_option('--run', type='int', help='Run number')
parser.set_defaults(range=None,bincount=25,orderby='count',maxcount=None,groupby='run',constraints=[],limit=-1,run=None,database=None,scenarios=None)
(options, args) = parser.parse_args()

if len(args)<1:
    print 'This script takes one required argument: path to job configuration file (xml).'
    sys.exit(2)

job = client.run.getJob(args[0])
jobid = os.path.splitext(os.path.basename(args[0]))[0]

if options.range is not None and options.range>0: options.range = -options.range

if options.database is None:
   import mysqlinfo
   db = mysqlinfo.connect(mysqlinfo.select)
else:
   import sqlite3
   db = sqlite3.connect(options.database)

# Build map from run identifier to source machine
c = db.cursor()
query = "SELECT `id`,`source`,`description` FROM `runs` WHERE `job`='%s'" % jobid
c.execute(query)
run2source = {}
source2fqn = {}
for run,source,description in c:
    # Chop of run@ string that is prepended if results arrive via MySQL
    if source.startswith('run@'): source=source[4:]

    # Try to resolve IP address, to get the host name.
    if source not in source2fqn:
        try:
            (fqn, aliaslist, ipaddrlist) = socket.gethostbyaddr(source)
        except:
            fqn = source
        source2fqn[source] = fqn
    else:
        fqn = source2fqn[source]

    # Link run identifier to source machine
    run2source[run] = source
c.close()

parconstraints = []
parnames = [p['name'] for p in job.controller.externalparameters]
for (name,minval,maxval) in options.constraints:
    minval,maxval = float(minval),float(maxval)
    i = parnames.index(name)
    parconstraints.append((i,minval,maxval))

# Retrieve all results
print 'Retrieving results...'
c = db.cursor()
runcrit = '`runs`.`id`'
if options.run is not None: runcrit = '%i' % options.run
query = "SELECT DISTINCT `results`.`id`,`parameters`,`lnlikelihood`,`%s` FROM `runs`,`results` WHERE `results`.`run`=%s AND `runs`.`job`='%s'" % (options.groupby,runcrit,jobid)
if options.limit!=-1: query += ' LIMIT %i' % options.limit
c.execute(query)
history = []
source2history = {}
group2maxlnl = {}
badcount = 0
i = 1
strlength = None
for rowid,strpars,lnlikelihood,group in c:
    # Make sure the lengths of all parameter strings match.
    if strlength is None:
        strlength = len(strpars)
    elif strlength!=len(strpars):
        print 'Skipping row %i because length of parameter string (%i) is less than expected (%i).' % (rowid,len(strpars),strlength)
        continue
    
    if lnlikelihood is None:
        badcount += 1
    else:
        valid = True
        try:
            parameters = numpy.array(strpars.split(';'),dtype=numpy.float)
        except ValueError:
            print 'Row %i: cannot parse "%s".' % (rowid,strpars)
            valid = False
        if valid and len(parameters)!=len(job.controller.externalparameters):
            print 'Row %i: Number of parameters (%i) does not match that of run (%i).' % (rowid,len(parameters),len(job.controller.externalparameters))
            valid = False
        if valid:
            for (ipar,minval,maxval) in parconstraints:
                if parameters[ipar]<minval or parameters[ipar]>maxval:
                    valid = False
                    break
        if valid:
            dat = (parameters,lnlikelihood)
            if lnlikelihood>group2maxlnl.get(group,-numpy.Inf): group2maxlnl[group]=lnlikelihood
            history.append(dat)
            source2history.setdefault(group,[]).append(dat)
    i += 1
db.close()
print 'Found %i results, of which %i were invalid.' % (len(history)+badcount,badcount)

# Stop if no results were found
if len(history)==0: sys.exit(0)

# Convert results into numpy array
res = numpy.empty((len(history),len(history[0][0])+1))
for i,(v,l) in enumerate(history):
    res[i,:-1] = v
    res[i,-1] = l

# Show best parameter set
indices = res[:,-1].argsort()
maxlnl = res[indices[-1],-1]
minlnl = res[indices[ 0],-1]
res = res[indices,:]
iinc = res[:,-1].searchsorted(maxlnl-1.92)
lbounds,rbounds = res[iinc:,:-1].min(axis=0),res[iinc:,:-1].max(axis=0)
best = res[-1,:-1]
outside = res[:iinc,:-1]
print 'Best parameter set is # %i with ln likelihood = %.6g:' % (indices[-1],maxlnl)
for ipar in range(res.shape[1]-1):
    # Get conservative confidence interval by extending it to the first point
    # from the boundary that has a likelihood value outside the allowed range.
    lvalid = outside[:,ipar]<lbounds[ipar]
    rvalid = outside[:,ipar]>rbounds[ipar]
    if lvalid.any(): lbounds[ipar] = outside[lvalid,ipar].max()
    if rvalid.any(): rbounds[ipar] = outside[rvalid,ipar].min()

    # Get parameter info
    pi = job.controller.externalparameters[ipar]

    # Undo log-transform (if any)
    if pi['logscale']:
        best[ipar] = math.pow(10.,best[ipar])
        lbounds[ipar] = 10.**lbounds[ipar]
        rbounds[ipar] = 10.**rbounds[ipar]

    # Report estimate and confidence interval
    print '   %s = %.6g (%.6g - %.6g)' % (pi['name'],best[ipar],lbounds[ipar],rbounds[ipar])

# Create parameter bins for histogram
npar = len(job.controller.externalparameters)
parbinbounds = numpy.empty((npar,options.bincount+1))
parbins = numpy.empty((npar,options.bincount))
parbins[:,:] = 1.1*(minlnl-maxlnl)
for ipar in range(npar):
    pi = job.controller.externalparameters[ipar]
    if pi['logscale']:
        parbinbounds[ipar,:] = numpy.linspace(math.log10(pi['minimum']),math.log10(pi['maximum']),options.bincount+1)
    else:
        parbinbounds[ipar,:] = numpy.linspace(pi['minimum'],pi['maximum'],options.bincount+1)

# Order sources (runs or clients) according to counts or ln likelihood.
print 'Points per %s:' % options.groupby
sources = source2history.keys()
if options.orderby=='count':
    sources = sorted(sources,cmp=lambda x,y: cmp(len(source2history[y]),len(source2history[x])))
else:
    sources = sorted(sources,cmp=lambda x,y: cmp(group2maxlnl[y],group2maxlnl[x]))
if options.maxcount is not None and len(sources)>options.maxcount: sources[options.maxcount:] = []
for source in sources:
    dat = source2history[source]
    label = source
    if options.groupby=='run': label = '%s (%s)' % (source,run2source[source])
    print '%s: %i points, best lnl = %.8g.' % (label,len(dat),group2maxlnl[source])

# Try importing MatPlotLib
try:
    import matplotlib.pylab
except ImportError:
    print 'MatPlotLib not found - skipping plotting.'
    sys.exit(1)
    
# Create the figure
matplotlib.pylab.figure(figsize=(12,8))
matplotlib.pylab.subplots_adjust(left=0.075,right=0.95,top=0.95,bottom=0.05,hspace=.3)

# Create subplots
nrow = math.ceil(math.sqrt(0.5*npar))
ncol = math.ceil(npar/nrow)
for ipar in range(npar):
    matplotlib.pylab.subplot(nrow,ncol,ipar+1)
    matplotlib.pylab.hold(True)
    
for source in sources:
    # Combine results for current source (run or client) into single array.
    dat = source2history[source]
    curres = numpy.zeros((len(dat),len(dat[0][0])+1))
    for i,(v,l) in enumerate(dat):
        if len(v)!=len(dat[0][0]): continue
        curres[i,:-1] = v
        curres[i,-1] = l
    curres[:,-1] -= maxlnl

    # Determine label for current source.
    label = source
    if options.groupby=='run': label = '%s (%s)' % (source,run2source[source])
    
    for ipar in range(npar):
        # Plot results for current source.
        matplotlib.pylab.subplot(nrow,ncol,ipar+1)
        matplotlib.pylab.plot(curres[:,ipar],curres[:,-1],'.',label=label)

        # Update histogram based on current source results.
        ind = parbinbounds[ipar,:].searchsorted(curres[:,ipar])-1
        for i,ibin in enumerate(ind):
            parbins[ipar,ibin] = max(parbins[ipar,ibin],curres[i,-1])

# Put finishing touches on subplots
for ipar in range(npar):
    pi = job.controller.externalparameters[ipar]
    
    matplotlib.pylab.subplot(nrow,ncol,ipar+1)
    #matplotlib.pylab.legend(sources,numpoints=1)

    # Add title
    matplotlib.pylab.title(pi['name'])

    # Plot marginal
    matplotlib.pylab.hold(True)
    x = numpy.concatenate((parbinbounds[ipar,0:1],numpy.repeat(parbinbounds[ipar,1:-1],2),parbinbounds[ipar,-1:]),0)
    y = numpy.repeat(parbins[ipar,:],2)
    matplotlib.pylab.plot(x,y,'-k',label='_nolegend_')

    #matplotlib.pylab.plot(res[:,0],res[:,1],'o')

    # Set axes boundaries
    minpar,maxpar = pi['minimum'],pi['maximum']
    lbound,rbound = lbounds[ipar],rbounds[ipar]
    if pi['logscale']:
        minpar,maxpar = math.log10(minpar),math.log10(maxpar)
        lbound,rbound = math.log10(lbound),math.log10(rbound)
    matplotlib.pylab.xlim(minpar,maxpar)
    ymin = minlnl-maxlnl
    if options.range is not None: ymin = options.range
    matplotlib.pylab.ylim(ymin,0)

    # Show confidence interval
    matplotlib.pylab.axvline(lbound,color='k',linestyle='--')
    matplotlib.pylab.axvline(rbound,color='k',linestyle='--')

#matplotlib.pylab.legend(numpoints=1)
matplotlib.pylab.savefig('estimates.png',dpi=300)
# Show figure and wait until the user closes it.
matplotlib.pylab.show()
