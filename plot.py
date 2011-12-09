# Import from standard Python library
import sys,math,optparse,socket,pickle

# Import third-party modules
import numpy
import matplotlib.pylab
import MySQLdb

# Import custom modules
import mysqlinfo
import client.run

parser = optparse.OptionParser()
parser.add_option('-r', '--range', type='float', help='Lower boundary for relative ln likelihood (always < 0)')
parser.add_option('--bincount', type='int', help='Number of bins for ln likelihood marginals')
parser.add_option('-g','--groupby',type='choice',choices=('source','run'),help='What identifier to group the results by, i.e., "source" or "run".')
parser.add_option('-o','--orderby',type='choice',choices=('count','lnl'),help='What property to order the result groups by, i.e., "count" or "lnl".')
parser.add_option('--maxcount',type='int',help='Maximum number of series to plot')
parser.add_option('--constraint',type='string',action='append',nargs=3,help='Constraint on parameter (parameter name, minimum, maximum)',dest='constraints')
parser.add_option('-l', '--limit', type='int', help='Maximum number of results to read')
parser.add_option('--run', type='int', help='Run number')
parser.set_defaults(range=None,bincount=25,orderby='count',maxcount=None,groupby='run',constraints=[],limit=-1,run=None)
(options, args) = parser.parse_args()

if len(args)<1:
    print 'No job identifier provided.'
    sys.exit(2)
jobid = int(args[0])

job = client.run.getJob(jobid)

if options.range!=None and options.range>0:
    print 'Range argument must be less than zero.'
    sys.exit(1)

db = MySQLdb.connect(host=mysqlinfo.host,user=mysqlinfo.viewuser,passwd=mysqlinfo.viewpassword,db=mysqlinfo.database)

# Build map from run identifier to source machine
c = db.cursor()
query = "SELECT `id`,`source`,`description` FROM `runs` WHERE `job`='%i'" % jobid
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
parnames = [p['name'] for p in job.controller.parameters]
for (name,minval,maxval) in options.constraints:
    minval,maxval = float(minval),float(maxval)
    i = parnames.index(name)
    parconstraints.append((i,minval,maxval))

# Retrieve all results
print 'Retrieving results...'
c = db.cursor()
runcrit = '`runs`.`id`'
if options.run is not None: runcrit = '%i' % options.run
query = "SELECT DISTINCT `parameters`,`lnlikelihood`,`%s` FROM `runs`,`results` WHERE `results`.`run`=%s AND `runs`.`job`='%i'" % (options.groupby,runcrit,jobid)
if options.limit!=-1: query += ' LIMIT %i' % options.limit
c.execute(query)
history = []
source2history = {}
group2maxlnl = {}
badcount = 0
i = 1
for strpars,lnlikelihood,group in c:
    if lnlikelihood==None:
        badcount += 1
    else:
        parameters = map(float,strpars.split(';'))
        valid = True
        for (ipar,minval,maxval) in parconstraints:
            if parameters[ipar]<minval or parameters[ipar]>maxval:
                valid = False
                break
        if not valid: continue
        assert len(parameters)==len(job.controller.parameters),'Row %i: Number of parameters (%i) does not match that of run (%i).' % (i,len(parameters),len(job.controller.parameters))
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
res = numpy.zeros((len(history),len(history[0][0])+1))
for i,(v,l) in enumerate(history):
    if len(v)!=len(history[0][0]): continue
    res[i,:-1] = v
    res[i,-1] = l

# Show best parameter set
indices = res[:,-1].argsort()
maxindex = indices[-1]
maxlnl = res[maxindex,-1]
minlnl = res[:,-1].min()
print 'Best parameter set is # %i with ln likelihood = %.6g:' % (maxindex,maxlnl)
for i in range(res.shape[1]-1):
    val = res[maxindex,i]
    pi = job.controller.parameters[i]
    val0,val2 = '',''
    for j in indices[-2::-1]:
        if val0=='' and res[j,i]<val and res[j,-1]<maxlnl-1.92: val0 = res[j,i]
        if val2=='' and res[j,i]>val and res[j,-1]<maxlnl-1.92: val2 = res[j,i]
        if val0!='' and val2!='': break
    if pi['logscale']:
        val = math.pow(10.,val)
        if val0!='': val0 = math.pow(10.,val0)
        if val2!='': val2 = math.pow(10.,val2)
    if val0!='': val0 = '%.6g' % val0
    if val2!='': val2 = '%.6g' % val2
    print '   %s = %.6g (%s - %s)' % (pi['name'],val,val0,val2)

# Create the figure
matplotlib.pylab.figure(figsize=(12,8))
matplotlib.pylab.subplots_adjust(left=0.075,right=0.95,top=0.95,bottom=0.05,hspace=.3)

# Set up subplots (one per parameter)
npar = len(job.controller.parameters)
nrow = math.ceil(math.sqrt(0.5*npar))
ncol = math.ceil(npar/nrow)
parbinbounds = numpy.empty((npar,options.bincount+1))
parbins = numpy.empty((npar,options.bincount))
parbins[:,:] = 1.1*(minlnl-maxlnl)
for ipar in range(npar):
    pi = job.controller.parameters[ipar]
    matplotlib.pylab.subplot(nrow,ncol,ipar+1)
    matplotlib.pylab.hold(True)
    if pi['logscale']:
        parbinbounds[ipar,:] = numpy.linspace(math.log10(pi['minimum']),math.log10(pi['maximum']),options.bincount+1)
    else:
        parbinbounds[ipar,:] = numpy.linspace(pi['minimum'],pi['maximum'],options.bincount+1)

# Plot likelihood values per parameters
print 'Points per %s:' % options.groupby
sources = source2history.keys()
if options.orderby=='count':
    sources = sorted(sources,cmp=lambda x,y: cmp(len(source2history[y]),len(source2history[x])))
else:
    sources = sorted(sources,cmp=lambda x,y: cmp(group2maxlnl[y],group2maxlnl[x]))
    
if options.maxcount!=None and len(sources)>options.maxcount: sources[options.maxcount:] = []
for source in sources:
    dat = source2history[source]
    curres = numpy.zeros((len(dat),len(dat[0][0])+1))
    for i,(v,l) in enumerate(dat):
        if len(v)!=len(dat[0][0]): continue
        curres[i,:-1] = v
        curres[i,-1] = l
    curres[:,-1] -= maxlnl
    label = source
    if options.groupby=='run': label = '%s (%s)' % (source,run2source[source])
    for ipar in range(npar):
        matplotlib.pylab.subplot(nrow,ncol,ipar+1)
        matplotlib.pylab.plot(curres[:,ipar],curres[:,-1],'.',label=label)
        ind = parbinbounds[ipar,:].searchsorted(curres[:,ipar])-1
        for i,ibin in enumerate(ind):
            parbins[ipar,ibin] = max(parbins[ipar,ibin],curres[i,-1])
    print '%s: %i points, best lnl = %.8g.' % (label,len(dat),group2maxlnl[source])

# Put finishing touches on subplots
for ipar in range(npar):
    pi = job.controller.parameters[ipar]
    
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
    if pi['logscale']: minpar,maxpar = math.log10(minpar),math.log10(maxpar)
    matplotlib.pylab.xlim(minpar,maxpar)
    lbound = minlnl-maxlnl
    if options.range!=None: lbound = options.range
    matplotlib.pylab.ylim(lbound,0)

#matplotlib.pylab.legend(numpoints=1)
matplotlib.pylab.savefig('estimates.png',dpi=300)
# Show figure and wait until the user closes it.
matplotlib.pylab.show()
