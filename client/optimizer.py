# Import from standard Python library (>= 2.4)
import sys,os.path,re,datetime,math,time,cPickle,hashlib,threading

# Import third-party modules
import numpy
import netCDF4

# Import personal custom stuff
import gotmcontroller,transport

# Regular expression for GOTM datetimes
datetimere = re.compile('(\d\d\d\d).(\d\d).(\d\d) (\d\d).(\d\d).(\d\d)\s*')

class RunTimeTransform(gotmcontroller.ParameterTransform):
    def __init__(self,ins,outs):
        self.expressions = []
        self.outvars = []
        for infile,namelist,variable,value in outs:
            self.outvars.append((infile,namelist,variable))
            self.expressions.append(value)
        self.outvars = tuple(self.outvars)
        
        self.innames = []
        bounds,logscale = {},{}
        for name,minval,maxval,haslogscale in ins:
            self.innames.append(name)
            bounds[name] = minval,maxval
            logscale[name] = haslogscale
        self.innames = tuple(self.innames)
            
        gotmcontroller.ParameterTransform.__init__(self,bounds,logscale)
        
    def getOriginalParameters(self):
        return self.outvars

    def getExternalParameters(self):
        return self.innames

    def undoTransform(self,values):
        workspace = dict(zip(self.innames,values))
        return tuple([eval(expr,workspace) for expr in self.expressions])

class attributes():
    def __init__(self,element,description):
        self.att = dict(element.attrib)
        self.description = description
        
    def get(self,name,type,default=None,required=None,minimum=None,maximum=None):
        value = self.att.pop(name,None)
        if value is None:
            # No value specified - use default or report error.
            if required is None: required = default is None
            if required: raise Exception('Attribute "%s" of %s is required.' % (name,self.description))
            value = default
        elif type is bool:
            # Boolean variable
            if value not in ('True','False'): raise Exception('Attribute "%s" of %s must have value "True" or "False".' % (name,self.description))
            value = value=='True'
        elif type in (float,int):
            # Numeric variable
            value = type(eval(value))
            if minimum is not None and value<minimum: raise Exception('The value of "%s" of %s must exceed %s.' % (name,self.description,minimum))
            if maximum is not None and value>maximum: raise Exception('The value of "%s" of %s must lie below %s.' % (name,self.description,maximum))
        else:
            # String
            value = type(value)
        return value
    def testEmpty(self):
        if self.att:
            print 'WARNING: the following attributes of %s are ignored: %s' % (self.description,', '.join(['"%s"' % k for k in self.att.keys()]))

class Job:
    verbose = True

    def __init__(self,jobid,scenariodir,gotmexe='./gotm.exe',copyexe=False,tempdir=None,simulationdir=None):
        self.initialized = False

        self.jobid = jobid

        # Create GOTM controller that takes care of setting namelist parameters, running GOTM, etc.
        self.controller = gotmcontroller.Controller(scenariodir,gotmexe,copyexe=copyexe,tempdir=tempdir,simulationdir=simulationdir)

        # Array to hold observation datasets
        self.observations = []

        # Whether to reject parameter sets outside of initial parameter ranges.
        # Rejection means returning a ln likelihood of negative infinity.
        self.checkparameterranges = True

    @staticmethod
    def fromConfigurationFile(path,**kwargs):
        import xml.etree.ElementTree
        tree = xml.etree.ElementTree.parse(path)

        jobid = os.path.splitext(os.path.basename(path))[0]

        # Allow overwrite of setup directory
        element = tree.find('setup')
        if element is not None:
           att = attributes(element,'the setup element')
           scenariodir = os.path.join(os.path.dirname(path),att.get('path',unicode))
           att.testEmpty()
        else:
           scenariodir = os.path.dirname(path)

        # Parse executable section
        element = tree.find('executable')
        if element is None: raise Exception('The root node must contain a single "executable" element.')
        att = attributes(element,'the executable element')
        exe = os.path.join(os.path.dirname(path),att.get('path',unicode))
        att.testEmpty()

        # Create job object
        job = Job(jobid,scenariodir,gotmexe=exe,copyexe=not hasattr(sys,'frozen'),**kwargs)
        
        # Parse parameters section
        for ipar,element in enumerate(tree.findall('parameters/parameter')):
            att = attributes(element,'parameter %i' % (ipar+1))
            if att.get('dummy',bool,default=False):
                job.controller.addDummyParameter('dummy',att.get('minimum',float,default=0.),att.get('maximum',float,default=1.))
            else:
                infile   = att.get('file',    unicode)
                namelist = att.get('namelist',unicode)
                variable = att.get('variable',unicode)
                att.description = 'parameter %s/%s/%s' % (infile,namelist,variable)
                minimum = att.get('minimum',float)
                maximum = att.get('maximum',float)
                job.controller.addParameter(infile,namelist,variable,
                                            minimum,maximum,
                                            logscale=att.get('logscale',bool,default=False))
            att.testEmpty()

        # Parse transforms
        for ipar,element in enumerate(tree.findall('parameters/transform')):
            att = attributes(element,'transform %i' % (ipar+1,))
            att.testEmpty()
            ins,outs = [],[]
            for iin,inelement in enumerate(element.findall('in')):
                att = attributes(inelement,'transform %i, input %i' % (ipar+1,iin+1))
                name = att.get('name',unicode)
                att.description = 'transform %i, input %s' % (ipar+1,name)
                ins.append((name,att.get('minimum',float),att.get('maximum',float),att.get('logscale',bool,default=False)))
                att.testEmpty()
            for iout,outelement in enumerate(element.findall('out')):
                att = attributes(outelement,'transform %i, output %i' % (ipar+1,iout+1))
                infile   = att.get('file',    unicode)
                namelist = att.get('namelist',unicode)
                variable = att.get('variable',unicode)
                att.description = 'transform %i, output %s/%s/%s' % (ipar+1,infile,namelist,variable)
                outs.append((infile,namelist,variable,att.get('value',unicode)))
                att.testEmpty()
            tf = RunTimeTransform(ins,outs)
            job.controller.addParameterTransform(tf)

        # Parse observations section
        n = 0
        for iobs,element in enumerate(tree.findall('observations/variable')):
            att = attributes(element,'observed variable %i' % (iobs+1))
            source = att.get('source',unicode)
            att.description = 'observation set %s' % source
            sourcepath = os.path.normpath(os.path.join(os.path.dirname(path),source))
            assert os.path.isfile(sourcepath),'Observation source file "%s" does not exist.' % sourcepath
            modelvariable = att.get('modelvariable',unicode)
            modelpath = att.get('modelpath',unicode)
            n += job.addObservation(sourcepath,modelvariable,modelpath,
                                    maxdepth          =att.get('maxdepth',           float,required=False,minimum=0.),
                                    mindepth          =att.get('mindepth',           float,required=False,minimum=0.),
                                    spinupyears       =att.get('spinupyears',        int,  default=0,     minimum=0),
                                    logscale          =att.get('logscale',           bool, default=False),
                                    relativefit       =att.get('relativefit',        bool, default=False),
                                    min_scale_factor  =att.get('minscalefactor',     float,required=False),
                                    max_scale_factor  =att.get('maxscalefactor',     float,required=False),
                                    fixed_scale_factor=att.get('constantscalefactor',float,required=False),
                                    minimum           =att.get('minimum',            float,default=0.1),
                                    sd                =att.get('sd',                 float,required=False,minimum=0.),
                                    cache=True)
            att.testEmpty()
        if n==0:
           raise Exception('No valid observations found within specified depth and timee range.')

        job.controller.processTransforms()

        return job
            
    def addObservation(self,observeddata,outputvariable,outputpath,spinupyears=0,relativefit=False,min_scale_factor=None,max_scale_factor=None,sd=None,maxdepth=None,mindepth=None,cache=True,fixed_scale_factor=None,logscale=False,minimum=None):
        sourcepath = None
        if mindepth is None: mindepth = -numpy.inf
        if maxdepth is None: maxdepth = self.controller.depth
        assert maxdepth>0, 'Argument "maxdepth" must be positive but is %.6g  (distance from surface in meter).' % maxdepth

        def getMD5(path):
            #print 'Calculating MD5 hash of %s...' % path
            f = open(path,'rb')
            m = hashlib.md5()
            while 1:
                block = f.read(m.block_size)
                if not block: break
                m.update(block)
            f.close()
            return m.digest()

        if isinstance(observeddata,basestring):
            
            # Observations are specified as path to ASCII file.
            sourcepath = observeddata
            md5 = getMD5(sourcepath)

            observeddata = None
            if cache and os.path.isfile(sourcepath+'.cache'):
                # Retrieve cached copy of the observations
                f = open(sourcepath+'.cache','rb')
                oldmd5 = cPickle.load(f)
                if oldmd5!=md5:
                    print 'Cached copy of %s is out of date - file will be reparsed.' % sourcepath
                else:
                    print 'Getting cached copy of %s...' % sourcepath
                    observeddata = cPickle.load(f)
                f.close()
                
            if observeddata is None:
                # Parse ASCII file and store observations as matrix.
                if self.verbose:
                    print 'Reading observations for variable "%s" from "%s".' % (outputvariable,sourcepath)
                if not os.path.isfile(sourcepath):
                    raise Exception('"%s" is not a file.' % sourcepath)
                obs = []
                f = open(sourcepath,'rU')
                iline = 1
                while 1:
                    line = f.readline()
                    if not line: break
                    if line[0]=='#': continue
                    datematch = datetimere.match(line)
                    if datematch is None:
                        raise Exception('Line %i does not start with time (yyyy-mm-dd hh:mm:ss). Line contents: %s' % (iline,line))
                    refvals = map(int,datematch.group(1,2,3,4,5,6)) # Convert matched strings into integers
                    curtime = datetime.datetime(*refvals)
                    data = line[datematch.end():].split()
                    if len(data)!=2:
                        raise Exception('Line %i does not contain two values (depth, observation) after the date + time, but %i values.' % (iline,len(data)))
                    depth,value = map(float,data)
                    obs.append((gotmcontroller.date2num(curtime),depth,value))
                    if self.verbose and iline%20000==0:
                        print 'Read "%s" upto line %i.' % (sourcepath,iline)
                    iline += 1
                f.close()
                observeddata = numpy.array(obs)

                # Try to store cached copy of observations
                if cache:
                    try:
                        f = open(sourcepath+'.cache','wb')
                        cPickle.dump(md5,         f,cPickle.HIGHEST_PROTOCOL)
                        cPickle.dump(observeddata,f,cPickle.HIGHEST_PROTOCOL)
                        f.close()
                    except Exception,e:
                        print 'Unable to store cached copy of observation file. Reason: %s' % e

            mindate = self.controller.start+datetime.timedelta(days=spinupyears*365)
            mint,maxt = gotmcontroller.date2num(mindate),gotmcontroller.date2num(self.controller.stop)
            if (observeddata[:,1]>0.).any(): print 'WARNING: %i of %i values above z=0 m.' % ((observeddata[:,1]>0.).sum(),observeddata.shape[0])
            valid = numpy.logical_and(numpy.logical_and(numpy.logical_and(observeddata[:,0]>=mint,observeddata[:,0]<=maxt),observeddata[:,1]>-maxdepth),observeddata[:,1]<=-mindepth)
            print '%i of %i observations lie within active time and depth range.' % (valid.sum(),observeddata.shape[0])
            observeddata = observeddata[valid,:]
            #print '  observation range: %s - %s' % (observeddata[:,2].min(),observeddata[:,2].max())
        else:
            assert False, 'Currently observations must be supplied as path to an 3-column ASCII file.'

        if logscale and minimum is None:
            raise Exception('For log scale fitting, the (relevant) minimum value must be specified.')

        self.observations.append({'outputvariable':outputvariable,
                                  'outputpath':outputpath,
                                  'observeddata':  observeddata,
                                  'relativefit':   relativefit,
                                  'min_scale_factor':min_scale_factor,
                                  'max_scale_factor':max_scale_factor,
                                  'fixed_scale_factor':fixed_scale_factor,
                                  'sd':            sd,
                                  'sourcepath':    sourcepath,
                                  'logscale':      logscale,
                                  'minimum':       minimum})

        return len(observeddata)

    def excludeObservationPeriod(self,start,stop):
        print 'Excluding observations between %s and %s.' % (str(start),str(stop))
        start = gotmcontroller.date2num(start)
        stop  = gotmcontroller.date2num(stop)
        for obsinfo in self.observations:
            obs = obsinfo['observeddata']
            rows = numpy.logical_or(obs[:,0]<start,obs[:,0]>stop)
            nrowprev = obs.shape[0]
            obsinfo['observeddata'] = obs[rows,:]
            nrow = obsinfo['observeddata'].shape[0]
            if nrow!=nrowprev:
                print 'Excluded %i observations of %s.' % (nrowprev-nrow,obsinfo['outputvariable'])
    
    def getObservationPaths(self):
        return [obsinfo['sourcepath'] for obsinfo in self.observations if obsinfo['sourcepath']!=None]

    def describe(self):
        obs = []
        for obsinfo in self.observations:
            # Copy key attributes of observation (but not the data matrix)
            infocopy = {}
            for key in ('sourcepath','outputvariable','relativefit','min_scale_factor','max_scale_factor','sd'):
                infocopy[key] = obsinfo[key]

            # Add attributes describing the data matrix
            data = obsinfo['observeddata']
            infocopy['observationcount'] = data.shape[0]
            infocopy['timerange']  = (float(data[:,0].min()),float(data[:,0].max()))
            infocopy['depthrange'] = (float(data[:,1].min()),float(data[:,1].max()))
            infocopy['valuerange'] = (float(data[:,2].min()),float(data[:,2].max()))
            obs.append(infocopy)
        info = self.controller.getInfo()
        info['observations'] = obs
        import pickle
        return pickle.dumps(info)
        
    def initialize(self):
        assert not self.initialized, 'Job has already been initialized.'

        self.controller.initialize()

        self.initialized = True

    def evaluateFitness(self,values):
        if not self.initialized: self.initialize()

        print 'Evaluating fitness with parameter set [%s].' % ','.join(['%.6g' % v for v in values])

        # If required, check whether all parameters are within their respective range.
        if self.checkparameterranges:
              errors = self.controller.checkParameters(values)
              if errors is not None:
                 print errors
                 return -numpy.Inf

        returncode = self.controller.run(values)

        if returncode!=0:
              # Run failed
              print 'Returning ln likelihood = negative infinity to discourage use of this parameter set.'
              self.reportResult(values,None,error='Run stopped prematurely')
              return -numpy.Inf

        resultroot = self.controller.scenariodir

        # Check if this is the first model run/evaluation of the likelihood.
        if not hasattr(self,'file2variables'):
            # This is the first time that we evaluate the likelihood.
            # Find a list of all NetCDF variables that we need.
            # Also find the coordinates in the result arrays and the weights that should be
            # used to interpolate to the observations.
            
            self.file2variables = {}
            file2re = {}
            for obsinfo in self.observations:
                obsvar,outputpath,obsdata = obsinfo['outputvariable'],obsinfo['outputpath'],obsinfo['observeddata']

                with netCDF4.Dataset(os.path.join(resultroot,outputpath)) as nc:
                   if outputpath not in file2re: file2re[outputpath] = re.compile('(?<!\w)('+'|'.join(nc.variables.keys())+')(?!\w)')  # variable name that is not preceded and followed by a "word" character
                   if outputpath not in self.file2variables: self.file2variables[outputpath] = set()

                   # Find variable names in expression.
                   curncvars = set(file2re[outputpath].findall(obsvar))
                   assert len(curncvars)>0,'No variables in found in NetCDF file %s that match %s.' % (outputpath,obsvar)
                   self.file2variables[outputpath] |= curncvars

                   # Check dimensions of all used NetCDF variables
                   firstvar,dimnames = None,None
                   for varname in curncvars:
                       curdimnames = tuple(nc.variables[varname].dimensions)
                       if dimnames is None:
                           firstvar,dimnames = varname,curdimnames
                           assert len(dimnames)==4, 'Do not know how to handle variables with != 4 dimensions. "%s" has %i dimensions.' % (varname,len(dimnames))
                           assert dimnames[0]=='time', 'Dimension 1 of variable %s must be time, but is "%s".' % (varname,dimnames[0])
                           assert dimnames[1] in ('z','z1'),'Dimension 2 of variable %s must be depth (z or z1), but is "%s".' % (varname,dimnames[1])
                           assert dimnames[-2:]==('lat','lon'), 'Last two dimensions of variable %s must be latitude and longitude, but are "%s".'  % (varname,dimnames[-2:])
                       else:
                           assert curdimnames==dimnames, 'Dimensions of %s %s do not match dimensions of %s %s. Cannot combine both in one expression.' % (varname,curdimnames,firstvar,dimnames)

                   print 'Calculating coordinates for linear interpolation to "%s" observations...' % obsvar

                   # Get reference date used in NetCDF file (according to COARDS convention).
                   dateref = gotmcontroller.getReferenceTime(nc)

                   # Get coordinates
                   time_vals = nc.variables['time'     ][:]/86400+gotmcontroller.date2num(dateref)
                   if dimnames[1]=='z':
                      h = nc.variables['h'][:,:,0,0]
                      z_vals = h.cumsum(axis=1)-h[0,:].sum()-h/2
                   else:
                      print 'Depth dimension %s not supported.' % dimnames[1]

                   # Get and cache information for interpolation from model grid to observations.
                   obsinfo['interp2_info'] = gotmcontroller.interp2_info(time_vals,z_vals,obsdata[:,0],obsdata[:,1])

        # Get all model variables that we need from the NetCDF file.
        file2vardata  = {}
        for path,variables in self.file2variables.items():
           with netCDF4.Dataset(os.path.join(resultroot,path)) as nc:
              file2vardata[path] = dict([(vn,nc.variables[vn][:,:,0,0]) for vn in variables])

        # Start with zero ln likelihood (likelihood of 1)
        lnlikelihood = 0.

        # Enumerate over the sets of observations.
        for obsinfo in self.observations:
            obsvar,outputpath,obsdata = obsinfo['outputvariable'],obsinfo['outputpath'],obsinfo['observeddata']

            # Get model predictions on observation coordinates,
            # using linear interpolation into result array.
            allvals = eval(obsvar,file2vardata[outputpath])
            modelvals = gotmcontroller.interp2_frominfo(allvals,obsinfo['interp2_info'])

            if not numpy.isfinite(modelvals).all():
                print 'WARNING: one or more model values for %s are not finite.' % obsvar
                print 'Returning ln likelihood = negative infinity to discourage use of this parameter set.'
                self.reportResult(values,None,error='Some model values for %s are not finite' % obsvar)
                return -numpy.Inf

            obsvals = obsdata[:,2]
            if obsinfo['logscale']:
                modelvals = numpy.log10(numpy.maximum(modelvals,obsinfo['minimum']))
                obsvals   = numpy.log10(numpy.maximum(obsvals,  obsinfo['minimum']))

            # If the model fit is relative, calculate the optimal model to observation scaling factor.
            scale = None
            if obsinfo['relativefit']:
                if obsinfo['logscale']:
                    # Optimal scale factor is calculated from optimal offset on a log scale.
                    scale = 10.**(obsvals.mean()-modelvals.mean())
                else:
                    # Calculate optimal scale factor.
                    if (modelvals==0.).all():
                        print 'WARNING: cannot calculate optimal scaling factor for %s because all model values equal zero.' % obsvar
                        print 'Returning ln likelihood = negative infinity to discourage use of this parameter set.'
                        self.reportResult(values,None,error='All model values for %s equal 0' % obsvar)
                        return -numpy.Inf
                    scale = (obsvals*modelvals).sum()/(modelvals*modelvals).sum()
                    if not numpy.isfinite(scale):
                        print 'WARNING: optimal scaling factor for %s is not finite.' % obsvar
                        print 'Returning ln likelihood = negative infinity to discourage use of this parameter set.'
                        self.reportResult(values,None,error='Optimal scaling factor for %s is not finite' % obsvar)
                        return -numpy.Inf

                # Report and check optimal scale factor.
                print 'Optimal model-to-observation scaling factor for %s = %.6g.' % (obsvar,scale)
                if obsinfo['min_scale_factor'] is not None and scale<obsinfo['min_scale_factor']:
                    print 'Clipping optimal scale factor to minimum = %.6g.' % obsinfo['min_scale_factor']
                    scale = obsinfo['min_scale_factor']
                elif obsinfo['max_scale_factor'] is not None and scale>obsinfo['max_scale_factor']:
                    print 'Clipping optimal scale factor to maximum = %.6g.' % obsinfo['max_scale_factor']
                    scale = obsinfo['max_scale_factor']
            elif obsinfo['fixed_scale_factor'] is not None:
                scale = obsinfo['fixed_scale_factor']

            # Apply scale factor if set
            if scale is not None:
                if obsinfo['logscale']:
                    modelvals += numpy.log10(scale)
                else:
                    modelvals *= scale

            # Calculate difference between model outcome and observations
            diff = modelvals-obsvals

            # Calculate sum of squares
            ssq = numpy.sum(diff*diff)

            # Add to likelihood, weighing according to standard deviation of current data.
            if obsinfo['sd'] is None:
                # No standard deviation specified: calculate the optimal s.d.
                print 'Using optimal s.d. for %s = %.6g.' % (obsvar,math.sqrt(ssq/len(diff)))
                lnlikelihood -= len(diff)/2.*math.log(ssq)
            else:
                # Use the specified standard deviation
                lnlikelihood -= ssq/2./obsinfo['sd']/obsinfo['sd']
            
        print 'ln Likelihood = %.6g.' % lnlikelihood
        self.reportResult(values,lnlikelihood)

        return lnlikelihood

    def reportResult(self,values,lnlikelihood,error=None):
        pass

class Reporter:
    @staticmethod
    def fromConfigurationFile(path,description,allowedtransports=None,interactive=True):
        import xml.etree.ElementTree
        tree = xml.etree.ElementTree.parse(path)

        jobid = os.path.splitext(os.path.basename(path))[0]

        # Parse transports section
        transports = []
        for itransport,element in enumerate(tree.findall('transports/transport')):
            att = attributes(element,'transport %i' % (itransport+1))
            type = att.get('type',unicode)
            if allowedtransports is not None and type not in allowedtransports: continue
            if type=='mysql':
                defaultfile = att.get('defaultfile',unicode,required=False)
                curtransport = transport.MySQL(server  =att.get('server',  unicode,required=(defaultfile is None)),
                                               user    =att.get('user',    unicode,required=(defaultfile is None)),
                                               password=att.get('password',unicode,required=(defaultfile is None)),
                                               database=att.get('database',unicode,required=(defaultfile is None)),
                                               defaultfile = defaultfile)
            elif type=='http':
                curtransport = transport.HTTP(server  =att.get('server',  unicode),
                                              path    =att.get('path',    unicode))
            elif type=='sqlite':
                curtransport = transport.SQLite()
            else:
                raise Exception('Unknown transport type "%s".' % type)
            att.testEmpty()
            transports.append(curtransport)

        return Reporter(jobid,description,transports,interactive=interactive)

    def __init__(self,jobid,description,transports=None,interactive=True):

        self.jobid = jobid
        self.description = description

        # Check transports
        assert transports is not None,'One or more transports must be specified.'
        validtp = []
        for transport in transports:
            if transport.available():
                validtp.append(transport)
            else:
                print 'Transport %s is not available.' % str(transport)
        if not validtp:
            print 'No transport available; exiting...'
            sys.exit(1)
        self.transports = tuple(validtp)

        # Last working transport (to be set at run time)
        self.lasttransport = None

        # Identifier for current run
        self.runid = None

        # Queue with results yet to be reported.
        self.resultqueue = []
        self.allowedreportfailcount = None
        self.allowedreportfailperiod = 3600   # in seconds
        self.timebetweenreports = 60 # in seconds
        self.reportfailcount = 0
        self.lastreporttime = time.time()
        self.reportedcount = 0
        self.nexttransportreset = 30
        self.queuelock = None

        # Whether to allow for interaction with user (via e.g. raw_input)
        self.interactive = interactive
        
    def reportRunStart(self):
        runid = None
        for transport in self.transports:
            if not transport.available(): continue
            try:
                runid = transport.initialize(self.jobid,self.description)
            except Exception,e:
                print 'Failed to initialize run over %s.\nReason: %s' % (str(transport),str(e))
                runid = None
            if runid is not None:
                print 'Successfully initialized run over %s.\nRun identifier = %i' % (str(transport),runid)
                self.lasttransport = transport
                break
            
        if runid is None:
            print 'Unable to initialize run. Exiting...'
            sys.exit(1)

        self.runid = runid

    def createReportingThread(self):
        self.queuelock = threading.Lock()
        self.reportingthread = ReportingThread(self)
        self.reportingthread.start()

    def reportResult(self,values,lnlikelihood,error=None):
        if self.queuelock is None: self.createReportingThread()

        if not numpy.isfinite(lnlikelihood): lnlikelihood = None
        
        # Append result to queue
        self.queuelock.acquire()
        self.resultqueue.append((values,lnlikelihood))
        self.queuelock.release()
        
    def flushResultQueue(self,maxbatchsize=100):
        # Report the start of the run, if that was not done before.
        if self.runid is None: self.reportRunStart()

        while 1:
            # Take current results from the queue.
            self.queuelock.acquire()
            batch = self.resultqueue[:maxbatchsize]
            del self.resultqueue[:maxbatchsize]
            self.queuelock.release()

            # If there are no results to report, do nothing.
            if len(batch)==0: return

            # Reorder transports, prioritizing last working transport
            # Once in a while we retry the different transports starting from the top.
            curtransports = []
            if self.reportedcount<self.nexttransportreset:
                if self.lasttransport is not None: curtransports.append(self.lasttransport)
            else:
                self.nexttransportreset += 30
                self.lasttransport = None
            for transport in self.transports:
                if self.lasttransport is None or transport is not self.lasttransport: curtransports.append(transport)

            # Try to report the results
            for transport in curtransports:
                success = True
                try:
                    transport.reportResults(self.runid,batch,timeout=5)
                except Exception,e:
                    print 'Unable to report result(s) over %s. Reason:\n%s' % (str(transport),str(e))
                    success = False
                if success:
                    print 'Successfully delivered %i result(s) over %s.' % (len(batch),str(transport))
                    self.lasttransport = transport
                    break

            if success:
                # Register success and continue to report any remaining results.
                self.reportedcount += len(batch)
                self.reportfailcount = 0
                self.lastreporttime = time.time()
                continue

            # If we arrived here, reporting failed.
            self.reportfailcount += 1
            print 'Unable to report %i result(s). Last report was sent %.0f s ago.' % (len(batch),time.time()-self.lastreporttime)

            # Put unreported results back in queue
            self.queuelock.acquire()
            batch += self.resultqueue
            self.resultqueue = batch
            self.queuelock.release()

            # If interaction with user is not allowed, leave the result in the queue and return.
            if not self.interactive: return
                
            # Check if the report failure tolerances (count and period) have been exceeded.
            exceeded = False
            if self.allowedreportfailcount is not None and self.reportfailcount>self.allowedreportfailcount:
                print 'Maximum number of reporting failures (%i) exceeded.' % self.allowedreportfailcount
                exceeded = True
            elif self.allowedreportfailperiod is not None and time.time()>(self.lastreporttime+self.allowedreportfailperiod):
                print 'Maximum period of reporting failure (%i s) exceeded.' % self.allowedreportfailperiod
                exceeded = True

            # If the report failure tolerance has been exceeded, ask the user whether to continue.
            if exceeded:
                resp = None
                while resp not in ('y','n'):
                    resp = raw_input('To report results, connectivity to the server should be restored. Continue for now (y/n)? ')
                if resp=='n': sys.exit(1)
                self.reportfailcount = 0
                self.lastreporttime = time.time()

            # We will tolerate this failure (the server-side script may be unavailable temporarily)
            print 'Queuing current result for later reporting.'
            return

class ReportingThread(threading.Thread):
    def __init__(self,job):
        threading.Thread.__init__(self)
        self.job = job
    
    def run(self):
        while 1:
            self.job.flushResultQueue()
            time.sleep(self.job.timebetweenreports)
            
