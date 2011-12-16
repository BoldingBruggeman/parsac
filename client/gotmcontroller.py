# Import from standard Python library (>= 2.4)
import sys,os,os.path,re,subprocess,shutil,time,datetime,tempfile,atexit,math

# Import third-party modules
import numpy
#from Scientific.IO.NetCDF import NetCDFFile
import netCDF4
NetCDFFile = netCDF4.Dataset

# Import personal custom stuff
import namelist

# Regular expression for GOTM datetimes
datetimere = re.compile('(\d\d\d\d).(\d\d).(\d\d) (\d\d).(\d\d).(\d\d)\s*')
gotmdatere = re.compile('\.{4}(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)')

# Determine if we are running on Windows
windows = (sys.platform=='win32')

# Generic parse routine for namelist file
# Returns dictionary linking namelist names to namelist data
# (another dictionary linking parameter names to parameter values)
def parseNamelistFile(path):
    nmls,nmlorder = {},[]
    nmlfile = namelist.NamelistFile(path)
    while True:
        try:
            nml = nmlfile.parseNextNamelist()
        except:
            break
        nmls[nml.name] = dict(nml)
        nmlorder.append(nml.name)
    return nmls,tuple(nmlorder)

class ParameterTransform:
    def __init__(self,bounds=None):
        if bounds is None: bounds = {}
        self.bounds = bounds

    def getOriginalParameters(self):
        assert False, 'getOriginalParameters must be implemented by class deriving from ParameterTransform'
        return ()   # Tuple of tuples with each three elements: namelist file, namelist name, parameter name

    def getExternalParameters(self):
        assert False, 'getExternalParameters must be implemented by class deriving from ParameterTransform'
        return ()   # List of parameter names

    def getExternalParameterBounds(self,name):
        assert name in self.bounds,'Boundaries for %s have not been set.' % name
        return self.bounds[name]

    def undoTransform(self,values):
        assert False, 'undoTransform must be implemented by class deriving from ParameterTransform'
        return ()   # Untransformed values

class SimpleTransform(ParameterTransform):
    def __init__(self,inpars,outpars,func,bounds=None):      
        ParameterTransform.__init__(self,bounds)
        for i in inpars: assert len(i)==3,'Input parameter must be specified as tuple with namelist filename, namelist name and parameter name. Current value: %s.' % i
        self.inpars = inpars
        self.outpars = outpars
        self.func = func

    def getOriginalParameters(self):
        return self.inpars

    def getExternalParameters(self):
        return self.outpars

    def undoTransform(self,values):
        return self.func(*values)

def writeNamelistFile(path,nmls,nmlorder):
    f = open(path,'w')
    for nml in nmlorder:
        data = nmls[nml]
        f.write('&%s\n' % nml)
        for var,val in data.iteritems():
            f.write('\t%s = %s,\n' % (var,val))
        f.write('/\n\n')
    f.close()

def getReferenceTime(nc,varname='time'):
    time_var = nc.variables[varname]
    fullunit = time_var.units
    if ' since ' not in fullunit:
      raise Exception('"units" attribute of variable "time" equals "%s", which does not follow COARDS convention. Problem: string does not contain " since ".' % (fullunit,))
    timeunit,reftime = fullunit.split(' since ')
    datematch = re.match(r'(\d\d\d\d)[-\/](\d{1,2})-(\d{1,2})\s*',reftime)
    if datematch==None:
        raise Exception('"units" attribute of variable "time" equals "%s", which does not follow COARDS convention. Problem: cannot parse date in "%s".' % (fullunit,reftime))
    year,month,day = map(int,datematch.group(1,2,3))
    hour,min,sec = 0,0,0
    reftime = reftime[datematch.end():]
    if len(reftime)>0:
        timematch = re.match(r'(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d*)?)\s*',reftime)
        if timematch==None:
            raise Exception('"units" attribute of variable "time" equals "%s", which does not follow COARDS convention. Problem: cannot parse time in "%s".' % (fullunit,reftime))
        hour,min,sec = map(int,timematch.group(1,2,3))
    return datetime.datetime(year,month,day,hour,min,sec)

# Stolen from MatPlotLib's _to_ordinalf
# (importing MatPlotLib means we would depend on 100s of files)
HOURS_PER_DAY = 24.
MINUTES_PER_DAY  = 60.*HOURS_PER_DAY
SECONDS_PER_DAY =  60.*MINUTES_PER_DAY
MUSECONDS_PER_DAY = 1e6*SECONDS_PER_DAY
def date2num(dt):
    """
    convert datetime to the Gregorian date as UTC float days,
    preserving hours, minutes, seconds and microseconds.  return value
    is a float
    """

    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        delta = dt.tzinfo.utcoffset(dt)
        if delta is not None:
            dt -= delta

    base =  float(dt.toordinal())
    if hasattr(dt, 'hour'):
        base += (dt.hour/HOURS_PER_DAY + dt.minute/MINUTES_PER_DAY +
                 dt.second/SECONDS_PER_DAY + dt.microsecond/MUSECONDS_PER_DAY
                 )
    return base

def interp2_info(x,y,X,Y):
    # Find upper/lower boundary indices for time.
    x_high = x.searchsorted(X)
    x_low  = x_high-1
    x_high = x_high.clip(min=0,max=x.shape[0]-1)
    x_low  = x_low.clip (min=0,max=x.shape[0]-1)

    # Find upper/lower boundary indices for depth.
    y_high = y.searchsorted(Y)
    y_low  = y_high-1
    y_high = y_high.clip(min=0,max=y.shape[0]-1)
    y_low  = y_low.clip (min=0,max=y.shape[0]-1)

    # Calculate the weight of each corner of the rectangle
    weight_lowlow   = abs(x[x_high]-X)*abs(y[y_high]-Y)
    weight_lowhigh  = abs(x[x_high]-X)*abs(Y-y[y_low])
    weight_highlow  = abs(X-x[x_low]) *abs(y[y_high]-Y)
    weight_highhigh = abs(X-x[x_low]) *abs(Y-y[y_low])
    totweight = weight_lowlow+weight_lowhigh+weight_highlow+weight_highhigh
    weight_lowlow   /= totweight
    weight_lowhigh  /= totweight
    weight_highlow  /= totweight
    weight_highhigh /= totweight

    return {'x_high':x_high,'x_low':x_low,
            'y_high':y_high,'y_low':y_low,
            'weight_lowlow':  weight_lowlow, 'weight_lowhigh': weight_lowhigh,
            'weight_highlow': weight_highlow,'weight_highhigh':weight_highhigh}

def interp2_frominfo(z,info):
    # Get model predictions on observation coordinates,
    # using linear interpolation into result array.
    x_low,x_high = info['x_low'],info['x_high']
    y_low,y_high = info['y_low'],info['y_high']
    return (z[x_low, y_low ]*info['weight_lowlow']  +
            z[x_low, y_high]*info['weight_lowhigh'] +
            z[x_high,y_low ]*info['weight_highlow'] +
            z[x_high,y_high]*info['weight_highhigh'])

def interp2(x,y,z,X,Y,info=None,returninfo=False):
    if info==None: info = interp2_info(x,y,X,Y)
    Z = interp2_frominfo(z,info)
    if returninfo:
        return Z,info
    else:
        return Z

class Controller:
    verbose = True

    def __init__(self,scenariodir,gotmexe='./gotm.exe',copyexe=False,tempdir=None):
        self.scenariodir = scenariodir
        self.gotmexe = os.path.realpath(gotmexe)
        self.copyexe = copyexe
        self.tempdir = tempdir
        self.parameters = []
        self.parametertransforms = []

        if self.tempdir is not None:
            self.tempdir = os.path.abspath(self.tempdir)

        # Check for existence of scenario directory
        # (we need it now already to find start/stop of simulation)
        if not os.path.isdir(self.scenariodir):
            raise Exception('GOTM scenario directory "%s" does not exist.' % self.scenariodir)

        # Parse file with namelists describing the main scenario settings.
        path = os.path.join(self.scenariodir,'gotmrun.nml')
        nmls,order = parseNamelistFile(path)
        assert 'output' in nmls, 'Cannot find namelist named "output" in "%s".' % path
        assert 'time'   in nmls, 'Cannot find namelist named "time" in "%s".' % path

        # Find start and stop of simulation.
        # These will be used to prune the observation table.
        datematch = datetimere.match(nmls['time']['start'][1:-1])
        assert datematch!=None,'Unable to parse start datetime in "%s".' % nmls['time']['start'][1:-1]
        self.start = datetime.datetime(*map(int,datematch.group(1,2,3,4,5,6)))
        datematch = datetimere.match(nmls['time']['stop'][1:-1])
        assert datematch!=None,'Unable to parse stop datetime in "%s".' % nmls['time']['stop'][1:-1]
        self.stop = datetime.datetime(*map(int,datematch.group(1,2,3,4,5,6)))

        # Find the (relative) path where the NetCDF file will be written.
        self.outputpath = os.path.join(nmls['output']['out_dir'][1:-1],nmls['output']['out_fn'][1:-1]+'.nc')

        self.depth = float(nmls['station']['depth'])

        self.initialized = False

    def addDummyParameter(self,name,minimum,maximum,logscale=False):
        self.addParameter(None,None,name,minimum,maximum,logscale)

    def addParameterTransform(self,transform):
        self.parametertransforms.append(transform)

    def addParameter(self,namelistfile,namelistname,name,minimum,maximum,logscale=False):
        if namelistfile!=None and not os.path.isfile(os.path.join(self.scenariodir,namelistfile)):
            raise Exception('"%s" is not present in scenario directory "%s".' % (namelistfile,self.scenariodir))
        if logscale and minimum<=0:
            raise Exception('Minimum for "%s" = %.6g, but that value cannot be used as this parameter is set to move on a log-scale.' % (name,minimum))
        if maximum<minimum:
            raise Exception('Maximum value (%.6g) for "%s" < minimum value (%.6g).' % (maximum,name,minimum))
        parinfo = {'namelistfile':namelistfile,
                   'namelistname':namelistname,
                   'name'        :name,
                   'minimum'     :minimum,
                   'maximum'     :maximum,
                   'logscale'    :logscale}
        self.parameters.append(parinfo)

    def getInfo(self):
        return {'parameters':self.externalparameters}

    def initialize(self):
        assert not self.initialized, 'Controller has already been initialized.'
        self.initialized = True

        self.externalparameters = list(self.parameters)
        self.namelistparameters = [(pi['namelistfile'],pi['namelistname'],pi['name']) for pi in self.parameters]
        for transform in self.parametertransforms:
            self.namelistparameters += transform.getOriginalParameters()
            for extpar in transform.getExternalParameters():
                minval,maxval = transform.getExternalParameterBounds(extpar)
                self.externalparameters.append({'namelistfile':'none',
                             'namelistname':'none',
                             'name':extpar,
                             'minimum':minval,
                             'maximum':maxval,
                             'logscale':False})

        # Check for presence of GOTM executable.
        if not os.path.isfile(self.gotmexe):
            raise Exception('Cannot locate GOTM executable at "%s".' % self.gotmexe)

        # Check for presence of custom temporary directory (if set)
        if self.tempdir is not None and not os.path.isdir(self.tempdir):
            raise Exception('Custom temporary directory "%s" does not exist.' % self.tempdir)

        # Create a temporary directory for the scenario on disk
        # (decreases runtime compared to network because GOTM can access observations faster)
        tempscenariodir = tempfile.mkdtemp(prefix='gotmopt',dir=self.tempdir)
        atexit.register(shutil.rmtree,tempscenariodir,True)
        print 'Copying files for model setup...'
        for name in os.listdir(self.scenariodir):
            if name.endswith('.nc'): continue
            srcname = os.path.join(self.scenariodir,name)
            dstname = os.path.join(tempscenariodir,name)
            shutil.copy(srcname, dstname)
        self.scenariodir = tempscenariodir

        if self.copyexe:
            print 'Copying GOTM executable...'
            dstname = os.path.join(self.scenariodir,os.path.basename(self.gotmexe))
            shutil.copy(self.gotmexe, dstname)
            self.gotmexe = dstname

        # Adjust output path to use temporary scenario directory
        self.outputpath = os.path.normpath(os.path.join(self.scenariodir,self.outputpath))

        self.namelistfiles,self.namelistorder = {},{}
        for nmlfile,nmlname,parname in self.namelistparameters:
            # If this is a dummy parameter, continue
            if nmlfile is None: continue
            
            # Update path to namelist file to match temporary scenario directory.
            path = os.path.join(self.scenariodir,nmlfile)

            # If we already read this namelist file for some other parameter, just continue.
            if path in self.namelistfiles: continue

            # Backup current namelist file
            icopy = 0
            while os.path.isfile(path+'.backup%02i' % icopy): icopy+=1
            shutil.copy(path,path+'.backup%02i' % icopy)

            # Read all namelist in the file, and store their data and order.
            nmls,nmlorder = parseNamelistFile(path)
            if nmlname not in nmls:
                raise Exception('Namelist "%s" does not exist in "%s".' % (nmlname,path))
            self.namelistfiles[path] = nmls
            self.namelistorder[path] = tuple(nmlorder)
            
    def setParameters(self,values):
        assert self.initialized, 'Job has not been initialized yet.'

        # Update the value of all untransformed namelist parameters
        for parinfo,value in zip(self.parameters,values):
            if parinfo['logscale']: value = math.pow(10.,value)
            if parinfo['namelistfile'] is not None:
                nmlpath = os.path.join(self.scenariodir,parinfo['namelistfile'])
                #print 'Setting %s/%s/%s to %s.' % (nmlpath,parinfo['namelistname'],parinfo['name'],value)
                self.namelistfiles[nmlpath][parinfo['namelistname']][parinfo['name']] = '%.8g' % value

        # Update namelist parameters that are governed by transforms
        ipar = len(self.parameters)
        for transform in self.parametertransforms:
            ext = transform.getExternalParameters()
            basevals = transform.undoTransform(values[ipar:ipar+len(ext)])
            for p,value in zip(transform.getOriginalParameters(),basevals):
                nmlpath = os.path.join(self.scenariodir,p[0])
                #print 'Setting %s/%s/%s to %s.' % (nmlpath,p[1],p[2],value)
                self.namelistfiles[nmlpath][p[1]][p[2]] = '%.8g' % value
            ipar += len(ext)

        # Write the new namelists to file.
        for nmlfile,nmls in self.namelistfiles.iteritems():
            f = open(nmlfile,'w')
            for nml in self.namelistorder[nmlfile]:
                data = nmls[nml]
                f.write('&%s\n' % nml)
                for var,val in data.iteritems():
                    f.write('\t%s = %s,\n' % (var,val))
                f.write('/\n\n')
            f.close()

    def createParameterSet(self):
        minvals,maxvals = self.getParameterBounds()
        values = []
        for minval,maxval in zip(minvals,maxvals):
            values.append((minval+maxval)/2.)
        return values

    def getParameterBounds(self,transformed=True):
        minvals,maxvals = [],[]
        for parinfo in self.parameters:
            minval,maxval = parinfo['minimum'],parinfo['maximum']
            minvals.append(minval)
            maxvals.append(maxval)
        for transform in self.parametertransforms:
            for extpar in transform.getExternalParameters():
                minval,maxval = transform.getExternalParameterBounds(extpar)
                minvals.append(minval)
                maxvals.append(maxval)
        if transformed:
            minvals,maxvals = self.transformParameterValues(minvals),self.transformParameterValues(maxvals)
        return tuple(minvals),tuple(maxvals)

    def untransformParameterValues(self,values):
        res = list(values)
        for i,parinfo in enumerate(self.parameters):
            if parinfo['logscale']: res[i] = math.pow(10.,res[i])
        return res

    def transformParameterValues(self,values):
        res = list(values)
        for i,parinfo in enumerate(self.parameters):
            if parinfo['logscale']: res[i] = math.log10(res[i])
        return res

    def checkParameters(self,values,valuesaretransformed=True):
        if valuesaretransformed: values = self.untransformParameterValues(values)
        bounds = self.getParameterBounds(transformed=False)
        i = 0
        for val,minval,maxval in zip(values,bounds[0],bounds[1]):
            if val<minval or val>maxval:
                return 'Parameter %i with value %.6g lies out of range (%.6g - %.6g), returning ln likelihood of -infinity.' % (i,val,minval,maxval)
            i+=1
        return None

    def run(self,values,showoutput=False,returnncpath=False):
        # Check number of parameters
        npar = len(self.getParameterBounds()[0])
        assert len(values)==npar, 'run was called with %i parameters, but the model was configured for %i parameters.' % (len(values),npar)
        
        # Transfer parameter values to GOTM scenario
        self.setParameters(values)

        # Take time and start GOTM
        time_start = time.time()
        print 'Starting model run...'
        if windows:
            # We start the process with low priority
            proc = subprocess.Popen(['start','/B','/WAIT','/LOW',self.gotmexe],shell=True,cwd=self.scenariodir,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen(self.gotmexe,cwd=self.scenariodir,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

        # GOTM is now running
        # Process GOTM output and show progress every now and then.
        nextprogress = .1
        span = float((self.stop-self.start).days)
        cleanfinish = False
        while 1:
            line = proc.stdout.readline()
            if line=='': break
            if showoutput:
                print line,
            else:
                datematch = gotmdatere.search(line)
                if datematch!=None:
                    refvals = map(int,datematch.group(1,2,3,4,5,6)) # Convert matched strings into integers
                    curtime = datetime.datetime(*refvals)
                    prog = (curtime-self.start).days/span
                    if prog>=nextprogress:
                        print '%i %% done.' % (prog*100.)
                        nextprogress += .1
            if ': Finished on ' in line: cleanfinish = True
        proc.wait()

        # Calculate and show elapsed time. Report error if GOTM did not complete gracefully.
        elapsed = time.time()-time_start
        print 'Model run took %.1f s.' % elapsed
        if not cleanfinish:
            print 'WARNING: model run stopped prematurely - an error must have occured.'
            return None

        if returnncpath: return self.outputpath
            
        # Open NetCDF file
        return NetCDFFile(self.outputpath)

    def getNetCDFVariables(self,nc,expressions,addcoordinates=False,ncvariables=None):
        # Check if this is the first model run/evaluation of the likelihood.
        if ncvariables==None:
            # Find a list of all NetCDF variables that we need.
            ncvariables = set()
            varre = re.compile('(?<!\w)('+'|'.join(nc.variables.keys())+')(?!\w)')  # variable name that is not preceded and followed by a "word" character
            for expr in expressions:
                # Find variable names in expression.
                curncvars = set(varre.findall(expr))
                ncvariables |= curncvars

                # Check dimensions of all used NetCDF variables
                firstvar,dimnames = None,None
                for varname in curncvars:
                    curdimnames = tuple(nc.variables[varname].dimensions)
                    if dimnames==None:
                        firstvar,dimnames = varname,curdimnames
                        assert len(dimnames)==4, 'Do not know how to handle variables with != 4 dimensions. "%s" has %i dimensions.' % (varname,len(dimnames))
                        assert dimnames[0]=='time', 'Dimension 1 of variable %s must be time, but is "%s".' % (varname,dimnames[0])
                        assert dimnames[1] in ('z','z1'),'Dimension 2 of variable %s must be depth (z or z1), but is "%s".' % (varname,dimnames[1])
                        assert dimnames[-2:]==('lat','lon'), 'Last two dimensions of variable %s must be latitude and longitude, but are "%s".'  % (varname,dimnames[-2:])
                    else:
                        assert curdimnames==dimnames, 'Dimensions of %s %s do not match dimensions of %s %s. Cannot combine both in one expression.' % (varname,curdimnames,firstvar,dimnames)

        # Get all model variables that we need.
        vardata = dict([(vn,nc.variables[vn][:,:,0,0]) for vn in ncvariables])

        res = {}
        for expr in expressions:
            res[expr] = eval(expr,vardata)

        if addcoordinates:
            tim = numpy.asarray(nc.variables['time'])
            z   = numpy.asarray(nc.variables['z'])
            z1  = numpy.asarray(nc.variables['z1'])

            tim = tim/86400.+date2num(getReferenceTime(nc))

            dtim = numpy.diff(tim)/2.
            tim_stag = numpy.zeros((len(tim)+1,))
            tim_stag[0 ] = tim[0]-dtim[0]
            tim_stag[1:-1] = tim[0:-1]+dtim
            tim_stag[-1] = tim[-1]+dtim[-1]

            z_stag = numpy.zeros((len(z1)+1,))
            z_stag[1:] = z1
            z_stag[0] = -self.depth

            z1_stag = numpy.zeros((len(z)+1,))
            z1_stag[0:-1] = z
            z1_stag[0] = -self.depth
            z1_stag[-1] = 0.

            res.update({'time_center':tim,'time_staggered':tim_stag,
                        'z_center'   :z,  'z_staggered'   :z_stag,
                        'z1_center'  :z1, 'z1_staggered'  :z1_stag})

        return res
