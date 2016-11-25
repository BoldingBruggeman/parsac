# Import from standard Python library (>= 2.4)
import sys
import os
import re
import subprocess
import shutil
import time
import datetime
import tempfile
import atexit
import math

# Import third-party modules
import numpy

# Import personal custom stuff
import namelist

# Regular expression for GOTM datetimes
datetimere = re.compile(r'(\d\d\d\d).(\d\d).(\d\d) (\d\d).(\d\d).(\d\d)\s*')
gotmdatere = re.compile(r'\.{4}(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)')

# Determine if we are running on Windows
windows = sys.platform == 'win32'

# Generic parse routine for namelist file
# Returns dictionary linking namelist names to namelist data
# (another dictionary linking parameter names to parameter values)
def parseNamelistFile(path):
    nmls, nmlorder = {}, []
    nmlfile = namelist.NamelistFile(path)
    while 1:
        try:
            nml = nmlfile.parseNextNamelist()
        except:
            break
        nmls[nml.name] = dict(nml)
        nmlorder.append(nml.name)
    return nmls, tuple(nmlorder)

class ParameterTransform:
    def __init__(self, bounds=None, logscale=None):
        if bounds is None: bounds = {}
        if logscale is None: logscale = {}
        self.bounds = bounds
        self.logscale = logscale

    def getOriginalParameters(self):
        assert False, 'getOriginalParameters must be implemented by class deriving from ParameterTransform'
        return ()   # Tuple of tuples with each three elements: namelist file, namelist name, parameter name

    def getExternalParameters(self):
        assert False, 'getExternalParameters must be implemented by class deriving from ParameterTransform'
        return ()   # List of parameter names

    def getExternalParameterBounds(self, name):
        assert name in self.bounds,'Boundaries for %s have not been set.' % name
        return self.bounds[name]

    def hasLogScale(self, name):
        return self.logscale.get(name, False)

    def undoTransform(self,values):
        assert False, 'undoTransform must be implemented by class deriving from ParameterTransform'
        return ()   # Untransformed values

class SimpleTransform(ParameterTransform):
    def __init__(self, inpars, outpars, func, bounds=None):
        ParameterTransform.__init__(self, bounds)
        for i in inpars:
            assert len(i) == 3, 'Input parameter must be specified as tuple with namelist filename, namelist name and parameter name. Current value: %s.' % i
        self.inpars = inpars
        self.outpars = outpars
        self.func = func

    def getOriginalParameters(self):
        return self.inpars

    def getExternalParameters(self):
        return self.outpars

    def undoTransform(self, values):
        return self.func(*values)

def writeNamelistFile(path, nmls, nmlorder):
    with open(path, 'w') as f:
        for nml in nmlorder:
            f.write('&%s\n' % nml)
            for name, value in nmls[nml].iteritems():
                f.write('\t%s = %s,\n' % (name, value))
            f.write('/\n\n')

class Controller:
    verbose = True

    def __init__(self,scenariodir, gotmexe='./gotm.exe', copyexe=False, tempdir=None, simulationdir=None):
        self.scenariodir = scenariodir
        self.gotmexe = os.path.realpath(gotmexe)
        self.copyexe = copyexe
        self.tempdir = tempdir
        self.simulationdir = simulationdir
        self.parameters = []
        self.parametertransforms = []

        if self.tempdir is not None:
            self.tempdir = os.path.abspath(self.tempdir)

        # Check for existence of scenario directory
        # (we need it now already to find start/stop of simulation)
        if not os.path.isdir(self.scenariodir):
            raise Exception('GOTM scenario directory "%s" does not exist.' % self.scenariodir)

        # Parse file with namelists describing the main scenario settings.
        path = os.path.join(self.scenariodir, 'gotmrun.nml')
        nmls, order = parseNamelistFile(path)
        assert 'time'   in nmls, 'Cannot find namelist named "time" in "%s".' % path

        # Find start and stop of simulation.
        # These will be used to prune the observation table.
        datematch = datetimere.match(nmls['time']['start'][1:-1])
        assert datematch is not None, 'Unable to parse start datetime in "%s".' % nmls['time']['start'][1:-1]
        self.start = datetime.datetime(*map(int, datematch.group(1,2,3,4,5,6)))
        datematch = datetimere.match(nmls['time']['stop'][1:-1])
        assert datematch is not None, 'Unable to parse stop datetime in "%s".' % nmls['time']['stop'][1:-1]
        self.stop = datetime.datetime(*map(int, datematch.group(1,2,3,4,5,6)))

        self.initialized = False

    def addDummyParameter(self, name, minimum, maximum, logscale=False):
        self.addParameter(None, None, name, minimum, maximum, logscale)

    def addParameterTransform(self, transform):
        self.parametertransforms.append(transform)

    def addParameter(self,namelistfile,namelistname,name,minimum,maximum,logscale=False):
        if namelistfile != None and not os.path.isfile(os.path.join(self.scenariodir, namelistfile)):
            raise Exception('"%s" is not present in scenario directory "%s".' % (namelistfile, self.scenariodir))
        if logscale and minimum <= 0:
            raise Exception('Minimum for "%s" = %.6g, but that value cannot be used as this parameter is set to move on a log-scale.' % (name,minimum))
        if maximum < minimum:
            raise Exception('Maximum value (%.6g) for "%s" < minimum value (%.6g).' % (maximum, name, minimum))
        parinfo = {'namelistfile':namelistfile,
                   'namelistname':namelistname,
                   'name'        :name,
                   'minimum'     :minimum,
                   'maximum'     :maximum,
                   'logscale'    :logscale}
        self.parameters.append(parinfo)

    def getInfo(self):
        return {'parameters': self.externalparameters}

    def processTransforms(self):
        self.externalparameters = list(self.parameters)
        self.namelistparameters = [(pi['namelistfile'], pi['namelistname'], pi['name']) for pi in self.parameters]
        for transform in self.parametertransforms:
            self.namelistparameters += transform.getOriginalParameters()
            for extpar in transform.getExternalParameters():
                minval, maxval = transform.getExternalParameterBounds(extpar)
                haslogscale = transform.hasLogScale(extpar)
                self.externalparameters.append({'namelistfile':'none',
                             'namelistname':'none',
                             'name':extpar,
                             'minimum':minval,
                             'maximum':maxval,
                             'logscale':haslogscale})

    def initialize(self):
        assert not self.initialized, 'Controller has already been initialized.'
        self.initialized = True

        self.processTransforms()

        # Check for presence of GOTM executable.
        if not os.path.isfile(self.gotmexe):
            raise Exception('Cannot locate GOTM executable at "%s".' % self.gotmexe)

        # Check for presence of custom temporary directory (if set)
        if self.tempdir is not None and not os.path.isdir(self.tempdir):
            raise Exception('Custom temporary directory "%s" does not exist.' % self.tempdir)

        if self.simulationdir is not None:
            # A specific directory in which to simulate has been provided.
            tempscenariodir = os.path.abspath(self.simulationdir)
            if not os.path.isdir(tempscenariodir): os.mkdir(tempscenariodir)
        else:           
            # Create a temporary directory for the scenario on disk
            # (decreases runtime compared to network because GOTM can access observations faster)
            tempscenariodir = tempfile.mkdtemp(prefix='gotmopt', dir=self.tempdir)
            atexit.register(shutil.rmtree, tempscenariodir, True)

        print 'Copying files for model setup...'
        for name in os.listdir(self.scenariodir):
            if name.endswith('.nc'):
               print '   skipping %s because it is a NetCDF file' % name
               continue
            srcname = os.path.join(self.scenariodir, name)
            if os.path.isdir(srcname):
               print '   skipping %s because it is a directory' % name
               continue
            dstname = os.path.join(tempscenariodir, name)
            shutil.copy(srcname, dstname)
        self.scenariodir = tempscenariodir

        if self.copyexe:
            print 'Copying GOTM executable...'
            dstname = os.path.join(self.scenariodir, os.path.basename(self.gotmexe))
            shutil.copy(self.gotmexe, dstname)
            self.gotmexe = dstname

        self.namelistfiles, self.namelistorder = {}, {}
        for nmlfile, nmlname, parname in self.namelistparameters:
            # If this is a dummy parameter, continue
            if nmlfile is None: continue
            
            # Update path to namelist file to match temporary scenario directory.
            path = os.path.join(self.scenariodir, nmlfile)

            # If we already read this namelist file for some other parameter, just continue.
            if path in self.namelistfiles: continue

            # Backup current namelist file
            icopy = 0
            while os.path.isfile(path+'.backup%02i' % icopy): icopy += 1
            shutil.copy(path,path+'.backup%02i' % icopy)

            # Read all namelist in the file, and store their data and order.
            nmls,nmlorder = parseNamelistFile(path)
            if nmlname not in nmls:
                raise Exception('Namelist "%s" does not exist in "%s".' % (nmlname, path))
            self.namelistfiles[path] = nmls
            self.namelistorder[path] = tuple(nmlorder)

    def setParameters(self, values):
        assert self.initialized, 'Job has not been initialized yet.'

        values = self.untransformParameterValues(values)

        # Update the value of all untransformed namelist parameters
        for parinfo, value in zip(self.parameters, values):
            if parinfo['namelistfile'] is not None:
                nmlpath = os.path.join(self.scenariodir,parinfo['namelistfile'])
                #print 'Setting %s/%s/%s to %s.' % (nmlpath,parinfo['namelistname'],parinfo['name'],value)
                self.namelistfiles[nmlpath][parinfo['namelistname']][parinfo['name']] = '%.15g' % value

        # Update namelist parameters that are governed by transforms
        ipar = len(self.parameters)
        for transform in self.parametertransforms:
            ext = transform.getExternalParameters()
            basevals = transform.undoTransform(values[ipar:ipar+len(ext)])
            for p, value in zip(transform.getOriginalParameters(), basevals):
                nmlpath = os.path.join(self.scenariodir, p[0])
                #print 'Setting %s/%s/%s to %s.' % (nmlpath,p[1],p[2],value)
                self.namelistfiles[nmlpath][p[1]][p[2]] = '%.15g' % value
            ipar += len(ext)

        # Write the new namelists to file.
        for nmlfile, nmls in self.namelistfiles.iteritems():
            with open(nmlfile, 'w') as f:
                for nml in self.namelistorder[nmlfile]:
                    f.write('&%s\n' % nml)
                    for name, value in nmls[nml].iteritems():
                        f.write('\t%s = %s,\n' % (name, value))
                    f.write('/\n\n')

    def createParameterSet(self):
        minvals,maxvals = self.getParameterBounds()
        values = []
        for minval,maxval in zip(minvals,maxvals):
            values.append((minval+maxval)/2.)
        return values

    def getParameterBounds(self, transformed=True):
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

    def untransformParameterValues(self, values):
        res = list(values)
        for i,parinfo in enumerate(self.externalparameters):
            if parinfo['logscale']: res[i] = math.pow(10.,res[i])
        return res

    def transformParameterValues(self, values):
        res = list(values)
        for i,parinfo in enumerate(self.externalparameters):
            if parinfo['logscale']: res[i] = math.log10(res[i])
        return res

    def checkParameters(self, values, valuesaretransformed=True):
        if valuesaretransformed: values = self.untransformParameterValues(values)
        bounds = self.getParameterBounds(transformed=False)
        for i, (val, minval, maxval) in enumerate(zip(values, bounds[0], bounds[1])):
            if val < minval or val > maxval:
                return 'Parameter %i with value %.6g lies out of range (%.6g - %.6g), returning ln likelihood of -infinity.' % (i, val, minval, maxval)
        return None

    def run(self, values, showoutput=False):
        # Check number of parameters
        npar = len(self.getParameterBounds()[0])
        assert len(values) == npar, 'run was called with %i parameters, but the model was configured for %i parameters.' % (len(values),npar)
        
        # Transfer parameter values to GOTM scenario
        self.setParameters(values)

        # Take time and start GOTM
        time_start = time.time()
        print 'Starting model run...'
        if windows:
            # We start the process with low priority
            proc = subprocess.Popen(['start', '/B', '/WAIT', '/LOW', self.gotmexe], shell=True, cwd=self.scenariodir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen(self.gotmexe, cwd=self.scenariodir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # GOTM is now running
        # Process GOTM output and show progress every now and then.
        if showoutput:
            while 1:
                line = proc.stdout.readline()
                if line == '': break
                if showoutput: print line,
        proc.communicate()

        # Calculate and show elapsed time. Report error if GOTM did not complete gracefully.
        elapsed = time.time()-time_start
        print 'Model run took %.1f s.' % elapsed
        if proc.returncode != 0: print 'WARNING: model run stopped prematurely - an error must have occured.'
        return proc.returncode

