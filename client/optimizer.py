# Import from standard Python library (>= 2.4)
import sys,os.path,re,datetime,math,time

# Import third-party modules
import numpy

# Import personal custom stuff
import gotmcontroller,transport

# Regular expression for GOTM datetimes
datetimere = re.compile('(\d\d\d\d).(\d\d).(\d\d) (\d\d).(\d\d).(\d\d)\s*')

class Job:
    verbose = True

    def __init__(self,jobid,scenariodir,gotmexe='./gotm.exe',transports=None,copyexe=False,interactive=True):
        self.initialized = False

        self.jobid = jobid

        # Transport settings
        assert transports is not None,'One or more transports must be specified.'
        self.transports = transports

        # Last working transport (to be set runtime)
        self.lasttransport = None

        # Create GOTM controller that takes care of setting namelist parameters, running GOTM, etc.
        self.controller = gotmcontroller.Controller(scenariodir,gotmexe,copyexe=copyexe)

        # Array to hold observation datasets
        self.observations = []

        # Identifier for current run
        self.runid = None

        # Whether to reject parameter sets outside of initial parameter ranges.
        # Rejection means returning a ln likelihood of negative infinity.
        self.checkparameterranges = True

        # Queue with results yet to be reported.
        self.resultqueue = []
        self.allowedreportfailcount = None
        self.allowedreportfailperiod = 3600   # in seconds
        self.reportfailcount = 0
        self.lastreporttime = time.time()
        self.reportedcount = 0
        self.nexttransportreset = 30

        # Whether to allow for interaction with user (via e.g. raw_input)
        self.interactive = interactive

    @staticmethod
    def fromConfigurationFile(path,jobid,scenariodir,allowedtransports=None):
        import xml.etree.ElementTree
        tree = xml.etree.ElementTree.parse(path)
        
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

        # Parse executable section
        element = tree.find('executable')
        if element is None: raise Exception('The root node must contain a single "executable" element.')
        att = attributes(element,'the executable element')
        exe = att.get('path',unicode)
        att.testEmpty()

        # Parse transports section
        transports = []
        for itransport,element in enumerate(tree.findall('transports/transport')):
            att = attributes(element,'transport %i' % (itransport+1))
            type = att.get('type',unicode)
            if allowedtransports is not None and type not in allowedtransports: continue
            if type=='mysql':
                curtransport = transport.MySQL(server  =att.get('server',  unicode),
                                               user    =att.get('user',    unicode),
                                               password=att.get('password',unicode),
                                               database=att.get('database',unicode))
            elif type=='http':
                curtransport = transport.MySQL(server  =att.get('server',  unicode),
                                               path    =att.get('path',    unicode))
            else:
                raise Exception('Unknown transport type "%s".' % type)
            att.testEmpty()
            transports.append(curtransport)

        # Create job object
        job = Job(jobid,scenariodir,gotmexe=exe,transports=transports,copyexe=not hasattr(sys,'frozen'))
        
        # Parse parameters section
        for ipar,element in enumerate(tree.findall('parameters/parameter')):
            att = attributes(element,'parameter %i' % (ipar+1))
            if att.get('dummy',bool,default=False):
                job.controller.addDummyParameter('dummy',att.get('minimum',float,default=0.),att.get('maximum',float,default=1.))
                continue
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

        # Parse observations section
        for iobs,element in enumerate(tree.findall('observations/variable')):
            att = attributes(element,'observed variable %i' % (iobs+1))
            source = att.get('source',unicode)
            att.description = 'observation set %s' % source
            sourcepath = os.path.normpath(os.path.join(scenariodir,source))
            assert os.path.isfile(sourcepath),'Observation source file "%s" does not exist.' % sourcepath
            modelvariable = att.get('modelvariable',unicode)
            job.addObservation(sourcepath,modelvariable,
                               maxdepth          =att.get('maxdepth',           float,required=False,minimum=0.),
                               spinupyears       =att.get('spinupyears',        int,  default=0,     minimum=0),
                               logscale          =att.get('logscale',           bool, default=False),
                               relativefit       =att.get('relativefit',        bool, default=False),
                               min_scale_factor  =att.get('minscalefactor',     float,required=False),
                               max_scale_factor  =att.get('maxscalefactor',     float,required=False),
                               fixed_scale_factor=att.get('constantscalefactor',float,required=False),
                               sd                =att.get('sd',                 float,required=False,minimum=0.),
                               cache=False)
            att.testEmpty()

        return job
            
    def addObservation(self,observeddata,outputvariable,spinupyears=0,relativefit=False,min_scale_factor=None,max_scale_factor=None,sd=None,maxdepth=None,cache=True,fixed_scale_factor=None,logscale=False):
        sourcepath = None
        if maxdepth==None: maxdepth = self.controller.depth
        assert maxdepth>0, 'Argument "maxdepth" must be positive but is %.6g  (distance from surface in meter).' % maxdepth
        if isinstance(observeddata,basestring):
            
            # Observations are specified as path to ASCII file.
            sourcepath = observeddata

            if cache and os.path.isfile(sourcepath+'.cache'):
                # Retrieve cached copy of the observations
                print 'Getting cached copy of %s...' % sourcepath
                observeddata = numpy.load(sourcepath+'.cache')
            else:
                # Parse ASCII file and store observations as matrix.
                if self.verbose:
                    print 'Reading observations for variable "%s" from "%s".' % (outputvariable,sourcepath)
                if not os.path.isfile(sourcepath):
                    raise Exception('"%s" is not a file.' % sourcepath)
                mindate = self.controller.start+datetime.timedelta(days=spinupyears*365)
                obs = []
                f = open(sourcepath)
                iline = 1
                while True:
                    line = f.readline()
                    if line=='': break
                    if line[0]!='#':
                        datematch = datetimere.match(line)
                        if datematch==None:
                            raise Exception('Line %i does not start with time (yyyy-mm-dd hh:mm:ss). Line contents: %s' % (iline,line))
                        refvals = map(int,datematch.group(1,2,3,4,5,6)) # Convert matched strings into integers
                        curtime = datetime.datetime(*refvals)
                        if curtime>=mindate and curtime<=self.controller.stop:
                            data = line[datematch.end():].split()
                            if len(data)!=2:
                                raise Exception('Line %i does not contain two values (depth, observation) after the date + time, but %i values.' % (iline,len(data)))
                            depth,value = map(float,data)
                            if depth>-maxdepth:
                                obs.append((gotmcontroller.date2num(curtime),depth,value))
                    if self.verbose and iline%10000==0:
                        print 'Read "%s" upto line %i.' % (observeddata,iline)
                    iline += 1
                f.close()
                observeddata = numpy.array(obs)

                # Try to store cached copy of observations
                if cache:
                    try:
                        observeddata.dump(sourcepath+'.cache')
                    except Exception,e:
                        print 'Unable to store cached copy of observation file. Reason: %s' % e
        else:
            assert False, 'Currently observations must be supplied as path to an 3-column ASCII file.'

        if relativefit:
            assert not logscale,'Fitting observations on a log-scale and estimating the scale factor is currently not supported.'

        self.observations.append({'outputvariable':outputvariable,
                                  'observeddata':  observeddata,
                                  'relativefit':   relativefit,
                                  'min_scale_factor':min_scale_factor,
                                  'max_scale_factor':max_scale_factor,
                                  'fixed_scale_factor':fixed_scale_factor,
                                  'sd':            sd,
                                  'sourcepath':    sourcepath,
                                  'logscale':      logscale})

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
        self.initialized = True

        validtp = []
        for transport in self.transports:
            if transport.available():
                validtp.append(transport)
            else:
                print 'Transport %s is not available.' % str(transport)
        if not validtp:
            print 'No transport available; exiting...'
            sys.exit(1)
        self.transports = tuple(validtp)

        self.controller.initialize()

    def evaluateFitness(self,values):
        if not self.initialized: self.initialize()
        
        print 'Evaluating fitness with parameter set [%s].' % ','.join(['%.6g' % v for v in values])

        # If required, check whether all parameters are within their respective range.
        if self.checkparameterranges:
            errors = self.controller.checkParameters(values)
            if errors is not None:
                print errors
                return -numpy.Inf

        nc = self.controller.run(values)

        if nc is None:
            # Run failed
            print 'Returning ln likelihood = negative infinity to discourage use of this parameter set.'
            self.reportResult(values,None,error='Run stopped prematurely')
            return -numpy.Inf

        # Check if this is the first model run/evaluation of the likelihood.
        if not hasattr(self,'ncvariables'):
            # This is the first time that we evaluate the likelihood.
            # Find a list of all NetCDF variables that we need.
            # Also find the coordinates in the result arrays and the weights that should be
            # used to interpolate to the observations.
            
            self.ncvariables = set()
            varre = re.compile('(?<!\w)('+'|'.join(nc.variables.keys())+')(?!\w)')  # variable name that is not preceded and followed by a "word" character
            for obsinfo in self.observations:
                obsvar,obsdata = obsinfo['outputvariable'],obsinfo['observeddata']

                # Find variable names in expression.
                curncvars = set(varre.findall(obsvar))
                assert len(curncvars)>0,'No variables in found in NetCDF file that match %s.' % obsvar
                self.ncvariables |= curncvars

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
                z_vals    = nc.variables[dimnames[1]][:]

                # Get and cache information for interpolation from model grid to observations.
                obsinfo['interp2_info'] = gotmcontroller.interp2_info(time_vals,z_vals,obsdata[:,0],obsdata[:,1])

        # Get all model variables that we need from the NetCDF file.
        vardata = dict([(vn,nc.variables[vn][:,:,0,0]) for vn in self.ncvariables])

        # Close NetCDF file; we have all data.
        nc.close()
        
        # Start with zero ln likelihood (likelihood of 1)
        lnlikelihood = 0.

        # Enumerate over the sets of observations.
        for obsinfo in self.observations:
            obsvar,obsdata = obsinfo['outputvariable'],obsinfo['observeddata']

            # Get model predictions on observation coordinates,
            # using linear interpolation into result array.
            allvals = eval(obsvar,vardata)
            modelvals = gotmcontroller.interp2_frominfo(allvals,obsinfo['interp2_info'])

            obsvals = obsdata[:,2]
            if obsinfo['logscale']:
                valid = numpy.logical_and(modelvals>0.1,obsvals>0.1)
                modelvals = numpy.log10(modelvals[valid])
                obsvals   = numpy.log10(obsvals  [valid])

            # If the model fit is relative, calculate the optimal model to observation scaling factor.
            if obsinfo['relativefit']:
                if (modelvals==0.).all():
                    print 'WARNING: cannot calculate optimal scaling factor for %s because all model values equal zero.' % obsvar
                    print 'Returning ln likelihood = negative infinity to discourage use of this parameter set.'
                    self.reportResult(values,None,error='All model values for %s equal 0' % obsvar)
                    return -numpy.Inf
                if not numpy.isfinite(modelvals).all():
                    print 'WARNING: cannot calculate optimal scaling factor for %s because one or more model were not finite.' % obsvar
                    print 'Returning ln likelihood = negative infinity to discourage use of this parameter set.'
                    self.reportResult(values,None,error='Some model values for %s are not finite' % obsvar)
                    return -numpy.Inf
                scale = (obsvals*modelvals).sum()/(modelvals*modelvals).sum()
                if not numpy.isfinite(scale):
                    print 'WARNING: optimal scaling factor for %s is not finite.' % obsvar
                    print 'Returning ln likelihood = negative infinity to discourage use of this parameter set.'
                    self.reportResult(values,None,error='Optimal scaling factor for %s is not finite' % obsvar)
                    return -numpy.Inf
                print 'Optimal model-to-observation scaling factor for %s = %.6g.' % (obsvar,scale)
                if obsinfo['min_scale_factor'] is not None and scale<obsinfo['min_scale_factor']:
                    print 'Clipping optimal scale factor to minimum = %.6g.' % obsinfo['min_scale_factor']
                    scale = obsinfo['min_scale_factor']
                elif obsinfo['max_scale_factor'] is not None and scale>obsinfo['max_scale_factor']:
                    print 'Clipping optimal scale factor to maximum = %.6g.' % obsinfo['max_scale_factor']
                    scale = obsinfo['max_scale_factor']
                modelvals *= scale
            elif obsinfo['fixed_scale_factor'] is not None:
                modelvals *= obsinfo['fixed_scale_factor']

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

    def reportRunStart(self):
        runid = None
        for transport in self.transports:
            if not transport.available(): continue
            try:
                runid = transport.initialize(self.jobid,self.describe())
            except Exception,e:
                print 'Failed to initialize run over %s.\nReason: %s' % (str(transport),str(e))
                runid = None
            if runid!=None:
                print 'Successfully initialized run over %s.\nRun identifier = %i' % (str(transport),runid)
                self.lasttransport = transport
                break
            
        if runid==None:
            print 'Unable to initialize run. Exiting...'
            sys.exit(1)

        self.runid = runid

    def reportResult(self,values,lnlikelihood,error=None):
        # Report the start of the run, if that was not done before.
        if self.runid==None: self.reportRunStart()

        # Append result to queue
        self.resultqueue.append((values,lnlikelihood))

        # Flush the queue if needed.
        self.flushResultQueue()
        
    def flushResultQueue(self):
        # Reorder transports, prioritizing last working transport
        # Once in a while we retry the different transports starting from the top.
        curtransports = []
        if self.reportedcount<self.nexttransportreset:
            if self.lasttransport!=None: curtransports.append(self.lasttransport)
        else:
            self.nexttransportreset += 30
            self.lasttransport = None
        for transport in self.transports:
            if self.lasttransport==None or transport is not self.lasttransport: curtransports.append(transport)

        # Try to report the results
        for transport in curtransports:
            success = True
            try:
                transport.reportResults(self.runid,self.resultqueue,timeout=5)
            except Exception,e:
                print 'Unable to report result(s) over %s. Reason:\n%s' % (str(transport),str(e))
                success = False
            if success:
                print 'Successfully delivered %i result(s) over %s.' % (len(self.resultqueue),str(transport))
                self.lasttransport = transport
                break

        if success:
            # Flush the queue
            self.reportedcount += len(self.resultqueue)
            self.resultqueue = []
            self.reportfailcount = 0
            self.lastreporttime = time.time()
            return

        # If we arrived here, reporting failed.
        print 'Unable to report result(s).'
        self.reportfailcount += 1

        # If interaction with user is not allowed, leave the result in the queue and return.
        if not self.interactive: return
            
        # Check if the report failure tolerances (count and period) have been exceeded.
        exceeded = False
        if self.allowedreportfailcount!=None and self.reportfailcount>self.allowedreportfailcount:
            print 'Maximum number of reporting failures (%i) exceeded.' % self.allowedreportfailcount
            exceeded = True
        elif self.allowedreportfailperiod!=None and time.time()>(self.lastreporttime+self.allowedreportfailperiod):
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
