# Import from standard Python library (>= 2.4)
from __future__ import print_function
import sys
import os.path
import re
import datetime
import time
import tempfile
import atexit
import shutil
import subprocess
import io
try:
    import cPickle as pickle
except ImportError:
    import pickle
import hashlib
import fnmatch

# Import third-party modules
import numpy
import netCDF4

try:
    import yaml
    import collections
    def dict_representer(dumper, data):
        return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())
    def dict_constructor(loader, node):
        return collections.OrderedDict(loader.construct_pairs(node))
    def none_representer(self, _):
        return self.represent_scalar('tag:yaml.org,2002:null', '')
    del yaml.loader.Loader.yaml_implicit_resolvers['o']
    del yaml.loader.Loader.yaml_implicit_resolvers['O']
    yaml.add_representer(collections.OrderedDict, dict_representer)
    yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, dict_constructor)
    yaml.add_representer(type(None), none_representer)
except ImportError:
    yaml = None

# Import custom packages
from . import namelist
from . import shared

# Regular expression for ISO 8601 datetimes
datetimere = re.compile(r'(\d\d\d\d).(\d\d).(\d\d) (\d\d).(\d\d).(\d\d)\s*')

def writeNamelistFile(path, nmls, nmlorder):
    if os.path.isfile(path):
        os.remove(path)
    with open(path, 'w') as f:
        for nml in nmlorder:
            f.write('&%s\n' % nml)
            for name, value in nmls[nml].items():
                f.write('\t%s = %s,\n' % (name, value))
            f.write('/\n\n')

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

def getMD5(path):
    #print('Calculating MD5 hash of %s...' % path)
    with open(path, 'rb') as f:
        m = hashlib.md5()
        while 1:
            block = f.read(m.block_size)
            if not block: break
            m.update(block)
    return m.digest()

class NamelistParameter(shared.Parameter):
    def __init__(self, job, att):
        self.file = os.path.normpath(att.get('file'))
        self.namelist = att.get('namelist')
        self.variable = att.get('variable')
        shared.Parameter.__init__(self, job, att, name='%s/%s/%s' % (self.file, self.namelist, self.variable))

        if not hasattr(job, 'namelistfiles'):
            job.namelistfiles, job.namelistorder = {}, {}

        self.own_file = self.file not in job.namelistfiles 
        if self.own_file:
            # Read all namelist in the file, and store their data and order.
            job.namelistfiles[self.file], job.namelistorder[self.file] = parseNamelistFile(os.path.join(job.scenariodir, self.file))

        if self.namelist not in job.namelistfiles[self.file]:
            raise Exception('Namelist "%s" is not present in "%s".' % (self.namelist, self.file))
        self.namelist_data = job.namelistfiles[self.file][self.namelist]
        if self.variable not in self.namelist_data:
            raise Exception('Variable "%s" is not present in namelist %s in "%s".' % (self.variable, self.namelist, self.file))

    def initialize(self):
        # Update path to namelist file to match temporary scenario directory.
        self.path = os.path.join(self.job.scenariodir, self.file)

        # If we already read this namelist file for some other parameter, just continue.
        if self.own_file:
            # Backup current namelist file
            icopy = 0
            while os.path.isfile(self.path+'.backup%02i' % icopy):
                icopy += 1
            shutil.copy(self.path, self.path+'.backup%02i' % icopy)

    def getValue(self):
        return self.namelist_data[self.variable]

    def setValue(self, value):
        self.namelist_data[self.variable] = '%.15g' % value

    def store(self):
        if self.own_file:
            writeNamelistFile(self.path, self.job.namelistfiles[self.file], self.job.namelistorder[self.file])

class YamlParameter(shared.Parameter):
    def __init__(self, job, att):
        self.file = os.path.normpath(att.get('file'))
        self.variable = att.get('variable')
        if yaml is None:
            raise Exception('Unable to handle parameter %s/%s because pyyaml package is not available.' % (self.file, self.variable)) 
        shared.Parameter.__init__(self, job, att, name='%s/%s' % (self.file, self.variable))

        if not hasattr(job, 'yamlfiles'):
            job.yamlfiles = {}

        self.own_file = self.file not in job.yamlfiles
        if self.own_file:
            # Read all namelist in the file, and store their data and order.
            with io.open(os.path.join(job.scenariodir, self.file), 'rU', encoding='utf-8') as f:
                job.yamlfiles[self.file] = yaml.safe_load(f)

        path_comps = self.variable.split('/')
        self.key = path_comps.pop()
        self.target_dict = job.yamlfiles[self.file]
        for i, comp in enumerate(path_comps):
            if comp not in self.target_dict:
                raise Exception('Variable "%s" not found in "%s" (key "%s" not found below /%s)' % (self.variable, self.file, comp, '/'.join(path_comps[:i])))
            self.target_dict = self.target_dict[comp]
        if self.key not in self.target_dict:
            raise Exception('Variable "%s" not found in "%s" (key "%s" not found below /%s)' % (self.variable, self.file, self.key, '/'.join(path_comps[:-1])))

    def initialize(self):
        # Update path to yaml file to match temporary scenario directory.
        self.path = os.path.join(self.job.scenariodir, self.file)

        # If we take charge of reading/writing the yaml file (i.e., we are the first yaml parameter referencing this file),
        # create a backup of it now.
        if self.own_file:
            icopy = 0
            while os.path.isfile(self.path+'.backup%02i' % icopy):
                icopy += 1
            shutil.copy(self.path, self.path+'.backup%02i' % icopy)

    def getValue(self):
        return self.target_dict[self.key]

    def setValue(self, value):
        self.target_dict[self.key] = float(value)

    def store(self):
        if self.own_file:
            if os.path.isfile(self.path):
                os.remove(self.path)
            with io.open(self.path, 'w', encoding='utf-8') as f:
                yaml.dump(self.job.yamlfiles[self.file], f, default_flow_style=False)

def readVariableFromTextFile(path, format, verbose=False, mindepth=-numpy.inf, maxdepth=numpy.inf):
    times, zs, values = [], [], []
    with open(path, 'rU') as f:
        for iline, line in enumerate(f):
            if verbose and (iline + 1) % 20000 == 0:
                print('Read "%s" upto line %i.' % (path, iline))
            if line.startswith('#'):
                continue
            datematch = datetimere.match(line)
            if datematch is None:
                raise Exception('Line %i does not start with time (yyyy-mm-dd hh:mm:ss). Line contents: %s' % (iline + 1, line))
            refvals = map(int, datematch.group(1, 2, 3, 4, 5, 6)) # Convert matched strings into integers
            curtime = datetime.datetime(*refvals)
            data = line[datematch.end():].rstrip('\n').split()
            if format == 0:
                # Depth-explicit variable (on each line: time, depth, value)
                if len(data) != 2:
                    raise Exception('Line %i does not contain two values (depth, observation) after the date + time, but %i values.' % (iline + 1, len(data)))
                z = float(data[0])
                if not numpy.isfinite(z):
                    raise Exception('Depth on line %i is not a valid number: %s.' % (iline + 1, data[0]))
                if -z < mindepth or -z > maxdepth:
                    continue
                zs.append(z)
            else:
                # Depth-independent variable (on each line: time, value)
                if len(data) != 1:
                    raise Exception('Line %i does not contain one value (observation) after the date + time, but %i values.' % (iline + 1, len(data)))
            times.append(curtime)
            value = float(data[-1])
            if not numpy.isfinite(value):
                raise Exception('Observed value on line %i is not a valid number: %s.' % (iline + 1, data[-1]))
            values.append(value)
    if format == 0:
        zs = numpy.array(zs)
    else:
        zs = None
    values = numpy.array(values)
    return times, zs, values

def readVariableFromNcFile(path, name, depth_name, verbose=False, mindepth=-numpy.inf, maxdepth=numpy.inf, time_name='time'):
    wrapped_nc = shared.NcDict(path)
    nctime = wrapped_nc.nc.variables[time_name]
    times = netCDF4.num2date(nctime[:], nctime.units, only_use_cftime_datetimes=False)
    values = wrapped_nc.eval(name)[...]
    assert values.ndim == 1
    zs = None if depth_name is None else wrapped_nc[depth_name][:]
    wrapped_nc.finalize()
    mask = numpy.ma.getmask(values)
    if mask is not numpy.ma.nomask:
        valid = ~mask
        times, values = times[valid], values[valid]
        if zs is not None:
            zs = zs[valid]
    return times, zs, values

class Job(shared.Job2):
    def __init__(self, job_id, xml_tree, root, copyexe=False, tempdir=None, simulationdir=None, verbose=True):
        # Allow overwrite of setup directory (default: directory with xml configuration file)
        element = xml_tree.find('setup')
        if element is not None:
            with shared.XMLAttributes(element, 'the setup element') as att:
                self.scenariodir = os.path.join(root, att.get('path', default='.'))
                self.exclude_files = att.get('exclude_files', default='*.nc')
                self.exclude_dirs = att.get('exclude_dirs', default='*')
        else:
            self.scenariodir = root
            self.exclude_files = '*.nc'
            self.exclude_dirs = '*'
        self.exclude_files = self.exclude_files.split(':')
        self.exclude_dirs = self.exclude_dirs.split(':')

        # Get executable path
        element = xml_tree.find('executable')
        if element is None:
            raise Exception('The root node must contain a single "executable" element.')
        with shared.XMLAttributes(element, 'the executable element') as att:
            self.use_shell = att.get('shell', bool, False)
            if self.use_shell:
                self.exe = att.get('path')
            else:
                self.exe = os.path.realpath(os.path.join(root, att.get('path')))
            self.max_runtime = att.get('max_runtime', int, required=False)

        self.simulationdir = simulationdir

        if tempdir is not None:
            tempdir = os.path.abspath(tempdir)
        self.tempdir = tempdir

        if copyexe is None:
            copyexe = not hasattr(sys, 'frozen')
        self.copyexe = copyexe
        self.symlink = False

        self.verbose = verbose

        #self.controller = gotmcontroller.Controller(scenariodir, path_exe, copyexe=not hasattr(sys, 'frozen'), tempdir=tempdir, simulationdir=simulationdir)

        # Initialize base class
        shared.Job.__init__(self, job_id, xml_tree, root)

        # Whether to reject parameter sets outside of initial parameter ranges.
        # Rejection means returning a ln likelihood of negative infinity.
        self.checkparameterranges = True

        self.statistics = []

        # If the XML contains a "target" element at root level, we will use that rather than compute a likelihood.
        self.targets = []
        for itarget, element in enumerate(xml_tree.findall('targets/target')):
            with shared.XMLAttributes(element, 'target %i' % (itarget + 1,)) as att:
                classname = att.get('class', required=False)
                if classname is not None:
                    try:
                        cls = self.get_class(classname, base=shared.Target)
                    except Exception as e:
                        raise Exception('Invalid class %s specified in "class" attribute of %s: %s' % (classname, att.description, e))
                else:
                    cls = shared.ExpressionTarget
                self.targets.append(cls(self, att))
        element = xml_tree.find('target')
        if element is not None:
            with shared.XMLAttributes(element, 'the target element') as att:
                self.targets.append(shared.ExpressionTarget(self, att))
        if self.targets:
            return

        # Look for additional statistics to save with result.
        for istatistic, element in enumerate(xml_tree.findall('extra_outputs/statistic')):
            with shared.XMLAttributes(element, 'statistic %i' % (istatistic + 1,)) as att:
                self.statistics.append((att.get('name'), att.get('expression')))

        # Array to hold observation datasets
        self.observations = []

        # Parse observations section
        n = 0
        for iobs, element in enumerate(xml_tree.findall('observations/variable')):
            with shared.XMLAttributes(element, 'observed variable %i' % (iobs+1)) as att:
                source = att.get('source')
                att.description = 'observation set %s' % source
                sourcepath = os.path.normpath(os.path.join(root, source))
                assert os.path.isfile(sourcepath), 'Observation source file "%s" does not exist.' % sourcepath
                modelvariable = att.get('modelvariable')
                modelpath = att.get('modelpath')
                file_format = att.get('format', default='profiles')
                n += self.addObservation(sourcepath, modelvariable, modelpath,
                    maxdepth          =att.get('maxdepth',           float, required=False, minimum=0.),
                    mindepth          =att.get('mindepth',           float, required=False, minimum=0.),
                    spinupyears       =att.get('spinupyears',        int, required=False, minimum=0),
                    logscale          =att.get('logscale',           bool, default=False),
                    relativefit       =att.get('relativefit',        bool, default=False),
                    min_scale_factor  =att.get('minscalefactor',     float, required=False),
                    max_scale_factor  =att.get('maxscalefactor',     float, required=False),
                    fixed_scale_factor=att.get('constantscalefactor',float, required=False),
                    minimum           =att.get('minimum',            float, default=0.1),
                    sd                =att.get('sd',                 float, required=False, minimum=0.),
                    file_format       ={'profiles':0, 'timeseries':1}[file_format],
                    variable          =att.get('variable', required=False),
                    depth_variable    =att.get('depth_variable', required=False),
                    cache=True)
        if n == 0:
            raise Exception('No valid observations found within specified depth and time range.')

    def getParameter(self, att):
        if att.get('dummy', bool, default=False):
            return shared.DummyParameter(self, att)
        strfile = att.get('file', required=False)
        if strfile is None:
            return shared.Parameter(self, att)
        elif strfile.endswith('.yaml'):
            return YamlParameter(self, att)
        return NamelistParameter(self, att)

    def getSimulationStart(self):
        raise NotImplementedError()

    def addObservation(self, observeddata, outputvariable, outputpath, spinupyears=None, relativefit=False, min_scale_factor=None, max_scale_factor=None, sd=None, maxdepth=None, mindepth=None, cache=True, fixed_scale_factor=None, logscale=False, minimum=None, file_format=0, variable=None, depth_variable=None):
        sourcepath = None
        if mindepth is None: mindepth = -numpy.inf
        if maxdepth is None: maxdepth = numpy.inf
        if maxdepth < 0: print('WARNING: maxdepth=%s, but typically should be positive (downward distance from surface in meter).' % maxdepth)
        assert maxdepth > mindepth, 'ERROR: maxdepth=%s should be greater than mindepth=%s' % (maxdepth, mindepth)

        assert isinstance(observeddata, (str, u''.__class__)), 'Currently observations must be supplied as path to an 3-column ASCII file.'

        # Observations are specified as path to ASCII file.
        sourcepath = observeddata
        md5 = getMD5(sourcepath)

        observeddata = None
        cache_path = sourcepath + ('' if variable is None else '.%s' % variable) + '.cache'
        if cache and os.path.isfile(cache_path):
            # Retrieve cached copy of the observations
            with open(cache_path, 'rb') as f:
                try:
                    oldmd5 = pickle.load(f)
                    if oldmd5 != md5:
                        print('Cached copy of %s is out of date - file will be reparsed.' % sourcepath)
                    else:
                        print('Loading cached copy of %s...' % sourcepath)
                        observeddata = pickle.load(f)
                except:
                    print('Failed to load cached copy of %s - file will be reparsed.' % sourcepath)

        if not isinstance(observeddata, tuple):
            # Parse ASCII file and store observations as matrix.
            if self.verbose:
                print('Reading observations for variable "%s" from "%s".' % (outputvariable, sourcepath))
            if not os.path.isfile(sourcepath):
                raise Exception('"%s" is not a file.' % sourcepath)
            if sourcepath.endswith('.nc'):
                if variable is None:
                    raise Exception('"variable" attribute must be provided since "%s" is a NetCDF file.' % sourcepath)
                times, zs, values = readVariableFromNcFile(sourcepath, variable, depth_variable, self.verbose, mindepth, maxdepth)
            else:
                times, zs, values = readVariableFromTextFile(sourcepath, file_format, self.verbose, mindepth, maxdepth)

            # Try to store cached copy of observations
            if cache:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(md5, f, pickle.HIGHEST_PROTOCOL)
                        pickle.dump((times, zs, values), f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to store cached copy of observation file. Reason: %s' % e)
        else:
            times, zs, values = observeddata

        if logscale and minimum is None:
            raise Exception('For log scale fitting, the (relevant) minimum value must be specified.')

        # Remove observations that lie within spin-up period (or before simulation start).
        if spinupyears is not None:
            start = self.getSimulationStart()
            obsstart = datetime.datetime(start.year+spinupyears, start.month, start.day)
            valid = numpy.array([t >= obsstart for t in times], dtype=bool)
            if not valid.any():
                print('Skipping "%s" because it has no observations after %s.' % (obsstart.strftime('%Y-%m-%d'),))
                return
            if zs is not None:
                zs = zs[valid]
            values = values[valid]
            times = [t for t, v in zip(times, valid) if v]

        self.observations.append({'outputvariable': outputvariable,
                                  'outputpath': outputpath,
                                  'times': times,
                                  'zs': zs,
                                  'values': values,
                                  'relativefit': relativefit,
                                  'min_scale_factor': min_scale_factor,
                                  'max_scale_factor': max_scale_factor,
                                  'fixed_scale_factor': fixed_scale_factor,
                                  'sd': sd,
                                  'sourcepath': sourcepath,
                                  'logscale': logscale,
                                  'minimum': minimum})

        return len(values)

    def getObservationPaths(self):
        return [obsinfo['sourcepath'] for obsinfo in self.observations if obsinfo['sourcepath'] is not None]

    def describe(self):
        obs = []
        for obsinfo in self.observations:
            # Copy key attributes of observation (but not the data matrix)
            infocopy = {}
            for key in ('sourcepath', 'outputvariable', 'relativefit', 'min_scale_factor', 'max_scale_factor', 'sd'):
                infocopy[key] = obsinfo[key]

            # Add attributes describing the data matrix
            times, zs, values = obsinfo['times'], obsinfo['zs'], obsinfo['values']
            infocopy['observationcount'] = len(times)
            infocopy['timerange'] = (min(times), max(times))
            infocopy['depthrange'] = (-float(zs.max()), -float(zs.min()))
            infocopy['valuerange'] = (float(values.min()), float(values.max()))
            obs.append(infocopy)
        parameter_info = [parameter.getInfo() for parameter in self.parameters]
        return pickle.dumps({'parameters':parameter_info, 'observations':obs})

    def on_start(self):
        # Check for presence of executable.
        if not self.use_shell:
            if not os.path.isfile(self.exe):
                raise Exception('Cannot locate executable at "%s".' % self.exe)

        # Check for presence of custom temporary directory (if set)
        if self.tempdir is not None and not os.path.isdir(self.tempdir):
            raise Exception('Custom temporary directory "%s" does not exist.' % self.tempdir)

        if self.simulationdir is not None:
            # A specific directory in which to simulate has been provided.
            tempscenariodir = os.path.abspath(self.simulationdir)
            if not os.path.isdir(tempscenariodir):
                os.mkdir(tempscenariodir)
        else:
            # Create a temporary directory for the scenario on disk
            # (decreases runtime compared to network because executable can access observations faster)
            tempscenariodir = tempfile.mkdtemp(prefix='gotmopt', dir=self.tempdir)
            atexit.register(shutil.rmtree, tempscenariodir, True)

        print('Copying files for model setup to %s...' % tempscenariodir)
        for name in os.listdir(self.scenariodir):
            srcname = os.path.abspath(os.path.join(self.scenariodir, name))
            isdir = os.path.isdir(srcname)
            exclude_patterns = self.exclude_dirs if isdir else self.exclude_files
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(name, pattern):
                    print('   skipping %s because it matches one of the patterns in exclude_%s (%s)' % (name, 'dirs' if isdir else 'files', ':'.join(exclude_patterns)))
                    break
            else:
                dstname = os.path.join(tempscenariodir, name)
                copy_function = shutil.copy2 if not self.symlink else os.symlink
                if isdir:
                    shutil.copytree(srcname, dstname, copy_function=copy_function)
                else:
                    copy_function(srcname, dstname)
        self.scenariodir = tempscenariodir

        if not self.use_shell:
            if self.copyexe:
                print('Copying %s executable...' % os.path.basename(self.exe))
                dstname = os.path.join(self.scenariodir, os.path.basename(self.exe))
                shutil.copy(self.exe, dstname)
                self.exe = dstname

        for parameter in self.parameters:
            parameter.initialize()

        for function in self.functions:
            function.initialize()

        for target in self.targets:
            target.initialize()

        self.statistics = [(name, compile(statistic, '<string>', 'eval')) for name, statistic in self.statistics]

    def prepareDirectory(self, values):
        assert self.started

        # Update the value of all untransformed parameters
        for parameter, value in zip(self.parameters, values):
            parameter.setValue(value)

        # Apply any custom functions
        for fnc in self.functions:
            fnc.apply()

        # Update namelist parameters that are governed by transforms
        #ipar = len(self.parameters)
        #for transform in self.parametertransforms:
        #    ext = transform.getExternalParameters()
        #    basevals = transform.undoTransform(values[ipar:ipar+len(ext)])
        #    for p, value in zip(transform.getOriginalParameters(), basevals):
        #        nmlpath = os.path.join(self.scenariodir, p[0])
        #        #print('Setting %s/%s/%s to %s.' % (nmlpath,p[1],p[2],value))
        #        self.namelistfiles[nmlpath][p[1]][p[2]] = '%.15g' % value
        #    ipar += len(ext)

        # Save updated namelist/YAML files.
        for parameter in self.parameters:
            parameter.store()

    def evaluate2(self, values, extra_outputs=None, return_model_values=False, show_output=False):
        assert self.started

        if self.verbose:
            print('Evaluating fitness with parameter set [%s].' % ','.join(['%.6g' % v for v in values]))

        # If required, check whether all parameters are within their respective range.
        if self.checkparameterranges:
            for parameter, value in zip(self.parameters, values):
                if value < parameter.minimum or value > parameter.maximum:
                    errors = 'Parameter %s with value %.6g out of range (%.6g - %.6g), returning ln likelihood of -infinity.' % (parameter.name, value, parameter.minimum, parameter.maximum)
                    return -numpy.inf

        self.prepareDirectory(values)

        returncode = run_program(self.exe, self.scenariodir, use_shell=self.use_shell, show_output=show_output, verbose=self.verbose)

        if returncode != 0:
            # Run failed
            print('Returning ln likelihood = negative infinity to discourage use of this parameter set.')
            #self.reportResult(values, None, error='Run stopped prematurely')
            return -numpy.inf

        if getattr(self, 'observations', None) is None:
            results = []
            for target in self.targets:
                results.append(target.getValue(self.scenariodir))
            return results[0] if len(results) == 1 else results

        outputpath2nc = {}
        for obsinfo in self.observations:
            ncpath = os.path.join(self.scenariodir, obsinfo['outputpath'])
            if not os.path.isfile(ncpath):
                raise Exception('Output file "%s" was not created.' % ncpath)
            outputpath2nc[obsinfo['outputpath']] = shared.NcDict(ncpath)

        # Check if this is the first model run/evaluation of the likelihood.
        if not getattr(self, 'processed_expressions', False):
            # This is the first time that we evaluate the likelihood.
            # Find a list of all NetCDF variables that we need.
            # Also find the coordinates in the result arrays and the weights that should be
            # used to interpolate to the observations.

            self.processed_expressions = True
            for obsinfo in self.observations:
                # Get time indices (for left side of bracket for linear interpolation)
                # This also eliminates points outside the simulated period.
                print('Calculating weights for linear interpolation to "%s" observations...' % obsinfo['outputvariable'], end='')
                wrappednc = outputpath2nc[obsinfo['outputpath']]
                time_units = wrappednc.nc.variables['time'].units
                time_vals = wrappednc['time']
                if 'itimes' not in obsinfo:
                    itimes_left, iweights_left = [], []
                    valid = numpy.zeros((len(obsinfo['times']),), dtype=bool)
                    numtimes = netCDF4.date2num(obsinfo['times'], time_units)
                    for i, numtime in enumerate(numtimes):
                        iright = time_vals.searchsorted(numtime)
                        if iright == 0 or iright >= len(time_vals): continue
                        valid[i] = True
                        itimes_left.append(iright - 1)
                        iweights_left.append((time_vals[iright] - numtime) / (time_vals[iright] - time_vals[iright - 1]))
                    obsinfo['times'] = [t for t, v in zip(obsinfo['times'], valid) if v]
                    obsinfo['numtimes'] = numtimes[valid]
                    obsinfo['itimes'] = numpy.array(itimes_left, dtype=int)
                    obsinfo['time_weights'] = numpy.array(iweights_left, dtype=float)
                    obsinfo['values'] = obsinfo['values'][valid]
                    if obsinfo['zs'] is not None:
                        obsinfo['zs'] = obsinfo['zs'][valid]

                # Compile expression
                obsinfo['outputexpression'] = compile(obsinfo['outputvariable'], '<string>', 'eval')

                # Retrieve data and check dimensions
                wrappednc.dimensions = set()
                data = wrappednc.eval(obsinfo['outputexpression'])
                if data.shape[0] != time_vals.size:
                    raise Exception('The first dimension of %s should have length %i (= number of time points), but has length %i.' % (obsinfo['outputvariable'], time_vals.size, data.shape[0]))
                if obsinfo['zs'] is not None:
                    assert data.ndim == 2, 'Expected two dimensions (time, depth) in %s, but found %i dimensions.' % (obsinfo['outputvariable'], data.ndim)
                    depth_dimensions = wrappednc.dimensions.intersection(('z', 'z1', 'zi'))
                    assert len(depth_dimensions) > 0, 'No depth dimension (z, zi or z1) used by expression %s' % obsinfo['outputvariable']
                    assert len(depth_dimensions) <= 1, 'More than one depth dimension (%s) used by expression %s' % (', '.join(depth_dimensions), obsinfo['outputvariable'])
                    obsinfo['depth_dimension'] = depth_dimensions.pop()
                else:
                    assert data.ndim == 1, 'Expected only one dimension (time) in %s, but found %i dimensions.' % (obsinfo['outputvariable'], data.ndim)

                print('done.')

        # Start with zero ln likelihood (likelihood of 1)
        lnlikelihood = 0.

        # Enumerate over the sets of observations.
        if return_model_values:
            model_values = []
        if extra_outputs is not None:
            namespace = dict([(name, getattr(numpy, name)) for name in dir(numpy)])
        for obsinfo in self.observations:
            obsvar, outputpath, obsvals = obsinfo['outputvariable'], obsinfo['outputpath'], obsinfo['values']
            wrappednc = outputpath2nc[outputpath]

            # Get model predictions for current variable or expression.
            all_values_model = wrappednc.eval(obsinfo['outputexpression'])

            if obsinfo['zs'] is not None:
                # Get model depth coordinates (currently expresses depth as distance from current surface elevation!)
                h = wrappednc['h']
                h = h.reshape(h.shape[:2])
                h_cumsum = h.cumsum(axis=1)
                if obsinfo['depth_dimension'] == 'z':
                    # Centres (all)
                    zs_model = h_cumsum - h_cumsum[:, -1, numpy.newaxis] - h/2
                else:
                    # Interfaces (all except bottom)
                    zs_model = h_cumsum - h_cumsum[:, -1, numpy.newaxis]

                # Interpolate in depth (extrapolates if required)
                modelvals = numpy.empty_like(obsvals)
                previous_numtime = None
                for i, (numtime, ileft, weight, z) in enumerate(zip(obsinfo['numtimes'], obsinfo['itimes'], obsinfo['time_weights'], obsinfo['zs'])):
                    if previous_numtime != numtime:
                        zprof     = weight*zs_model        [ileft, :] + (1-weight)*zs_model[ileft+1, :]
                        valueprof = weight*all_values_model[ileft, :] + (1-weight)*all_values_model[ileft+1, :]
                        previous_numtime = numtime
                    jright = min(max(1, zprof.searchsorted(z)), len(zprof)-1)
                    z_weight = (z - zprof[jright-1]) / (zprof[jright] - zprof[jright-1])
                    modelvals[i] = (1-z_weight)*valueprof[jright-1] + z_weight*valueprof[jright]
            else:
                modelvals = obsinfo['time_weights']*all_values_model[obsinfo['itimes']] + (1-obsinfo['time_weights'])*all_values_model[obsinfo['itimes']+1]

            if not numpy.isfinite(modelvals).all():
                print('WARNING: one or more model values for %s are not finite.' % obsvar)
                print('Returning ln likelihood = negative infinity to discourage use of this parameter set.')
                #self.reportResult(values,None,error='Some model values for %s are not finite' % obsvar)
                for wrappednc in outputpath2nc.values():
                    wrappednc.finalize()
                return -numpy.inf

            if return_model_values:
                time_units = wrappednc.nc.variables['time'].units
                t_centers = wrappednc['time']
                if obsinfo['zs'] is not None:
                    if obsinfo['depth_dimension'] == 'z':
                        # Centres (all)
                        z_interfaces = numpy.hstack((-h_cumsum[:, -1, numpy.newaxis], h_cumsum-h_cumsum[:, -1, numpy.newaxis]))
                    else:
                        # Interfaces (all except bottom)
                        zs_model = h_cumsum-h_cumsum[:, -1, numpy.newaxis]
                    z_interfaces2 = numpy.empty((z_interfaces.shape[0]+1, z_interfaces.shape[1]))
                    delta_z_interfaces = numpy.diff(z_interfaces, axis=0) / 2
                    z_interfaces2[0,   :] = z_interfaces[0,  :] - delta_z_interfaces[0,:]
                    z_interfaces2[1:-1,:] = z_interfaces[:-1,:] + delta_z_interfaces
                    z_interfaces2[-1,  :] = z_interfaces[-1, :] + delta_z_interfaces[-1,:]

                    half_delta_time = numpy.diff(t_centers) / 2
                    tim_stag = numpy.zeros((len(t_centers) + 1,))
                    tim_stag[0 ] = t_centers[0] - half_delta_time[0]
                    tim_stag[1:-1] = t_centers[:-1] + half_delta_time
                    tim_stag[-1] = t_centers[-1] + half_delta_time[-1]
                    t_interfaces = numpy.repeat(netCDF4.num2date(tim_stag, time_units, only_use_cftime_datetimes=False)[:, numpy.newaxis], z_interfaces2.shape[1], axis=1)
                    model_values.append((t_interfaces, z_interfaces2, all_values_model, modelvals))
                else:
                    model_values.append((netCDF4.num2date(t_centers, time_units, only_use_cftime_datetimes=False), all_values_model, modelvals))

            if obsinfo['logscale']:
                modelvals = numpy.log10(numpy.maximum(modelvals, obsinfo['minimum']))
                obsvals   = numpy.log10(numpy.maximum(obsvals, obsinfo['minimum']))

            # If the model fit is relative, calculate the optimal model to observation scaling factor.
            scale = None
            if obsinfo['relativefit']:
                if obsinfo['logscale']:
                    # Optimal scale factor is calculated from optimal offset on a log scale.
                    scale = 10.**(obsvals.mean()-modelvals.mean())
                else:
                    # Calculate optimal scale factor.
                    if (modelvals == 0.).all():
                        print('WARNING: cannot calculate optimal scaling factor for %s because all model values equal zero.' % obsvar)
                        print('Returning ln likelihood = negative infinity to discourage use of this parameter set.')
                        for wrappednc in outputpath2nc.values():
                            wrappednc.finalize()
                        #self.reportResult(values, None, error='All model values for %s equal 0' % obsvar)
                        return -numpy.inf
                    scale = (obsvals*modelvals).sum()/(modelvals**2).sum()
                    if not numpy.isfinite(scale):
                        print('WARNING: optimal scaling factor for %s is not finite.' % obsvar)
                        print('Returning ln likelihood = negative infinity to discourage use of this parameter set.')
                        #self.reportResult(values, None, error='Optimal scaling factor for %s is not finite' % obsvar)
                        for wrappednc in outputpath2nc.values():
                            wrappednc.finalize()
                        return -numpy.inf

                # Report and check optimal scale factor.
                print('Optimal model-to-observation scaling factor for %s = %.6g.' % (obsvar, scale))
                if obsinfo['min_scale_factor'] is not None and scale < obsinfo['min_scale_factor']:
                    print('Clipping optimal scale factor to minimum = %.6g.' % obsinfo['min_scale_factor'])
                    scale = obsinfo['min_scale_factor']
                elif obsinfo['max_scale_factor'] is not None and scale > obsinfo['max_scale_factor']:
                    print('Clipping optimal scale factor to maximum = %.6g.' % obsinfo['max_scale_factor'])
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
            diff = modelvals - obsvals

            # Calculate sum of squares
            ssq = (diff**2).sum()
            n = len(diff)

            # Add to likelihood, weighing according to standard deviation of current data.
            sd = obsinfo['sd']
            if sd is None:
                # No standard deviation specified: calculate the optimal s.d.
                sd = numpy.sqrt(ssq/(n-1))
                print('Using optimal s.d. for %s = %.6g.' % (obsvar, sd))

            # Note: assuming normally distributed errors, and omitting constant terms in the log likelihood = -n*ln(2*pi)/2
            lnlikelihood += -n*numpy.log(sd)-ssq/2/sd/sd

            if extra_outputs is not None:
                for name, fn in self.statistics:
                    value = eval(fn, {'x': obsvals, 'y': modelvals}, namespace)
                    extra_outputs.setdefault(name, []).append(value)

        print('ln Likelihood = %.6g.' % lnlikelihood)

        for wrappednc in outputpath2nc.values():
            wrappednc.finalize()
        if return_model_values:
            return lnlikelihood, model_values
        return lnlikelihood

    def prepareEnsembleDirectories(self, ensemble, root, format='%04i', symlink=False):
        ensemble = numpy.asarray(ensemble)
        self.symlink = symlink
        if not os.path.isdir(root):
            os.makedirs(root)
        scenariodir, targets = self.scenariodir, getattr(self, 'targets', [])
        dir_paths = [os.path.join(root, format % i) for i in range(ensemble.shape[0])]
        for i, simulationdir in enumerate(dir_paths):
            self.simulationdir = simulationdir
            self.start(force=True)
            self.prepareDirectory(ensemble[i, :])
            self.scenariodir = scenariodir
            self.targets = targets
        return dir_paths

    def runEnsemble(self, directories, ncpus=None, ppservers=(), socket_timeout=600, secret=None, show_output=False):
        import pp
        if self.verbose:
            print('Starting Parallel Python server...')
        if ncpus is None:
            ncpus = 'autodetect'
        ppservers = shared.parse_ppservers(ppservers)
        job_server = pp.Server(ncpus=ncpus, ppservers=ppservers, socket_timeout=socket_timeout, secret=secret)
        if ppservers:
            if self.verbose:
                print('Giving Parallel Python 10 seconds to connect to: %s' % (', '.join(ppservers)))
            time.sleep(10)
            if self.verbose:
                print('Running on:')
                for node, ncpu in job_server.get_active_nodes().items():
                    print('   %s: %i cpus' % (node, ncpu))
        nworkers = sum(job_server.get_active_nodes().values())
        if self.verbose:
            print('Total number of cpus: %i' % nworkers)
        if nworkers == 0:
            raise Exception('No cpus available; exiting.')
        jobs = []
        for rundir in directories:
            localexe = os.path.join(self.scenariodir, os.path.basename(self.exe))
            exe = localexe if os.path.isfile(localexe) else self.exe
            job = job_server.submit(run_program, (exe, rundir, self.use_shell, show_output), modules=('time', 'subprocess'))
            jobs.append(job)
        for job, rundir in zip(jobs, directories):
            job()
            print('Processed %s...' % rundir)

def run_program(exe, rundir, use_shell=False, show_output=True, verbose=True):
    if verbose:
        time_start = time.time()
        print('Starting model run...')

    args = [exe]
    if exe.endswith('.py'):
        args = [sys.executable] + args
        use_shell = False
    if sys.platform == 'win32':
        # We start the process with low priority
        IDLE_PRIORITY_CLASS = 0x00000040
        proc = subprocess.Popen(args, cwd=rundir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=IDLE_PRIORITY_CLASS, shell=use_shell, universal_newlines=True)
    else:
        proc = subprocess.Popen(args, cwd=rundir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=use_shell, universal_newlines=True)

    # Simulation is now running
    if show_output:
        while 1:
            line = proc.stdout.readline()
            if line == '':
                break
            print(line, end='')
    stdout_data, _ = proc.communicate()

    # Calculate and show elapsed time. Report error if executable did not complete gracefully.
    if verbose:
        elapsed = time.time() - time_start
        print('Model run took %.1f s.' % elapsed)
    if proc.returncode != 0:
        last_output = '\n'.join(['> %s' % l for l in stdout_data.rsplit('\n', 10)[-10:]])
        print('WARNING: %s returned non-zero code %i - an error must have occured. Last output:\n%s' % (os.path.basename(exe), proc.returncode, last_output))
    return proc.returncode
