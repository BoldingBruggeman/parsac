from __future__ import print_function
import os.path

import numpy

try:
    from .. import optimize
except ValueError:
    import optimize

job_path = None
def run_ensemble_member(new_job_path, parameter_values):
    global job_path, job

    if 'job_path' not in globals() or job_path != new_job_path:
        # This worker is being run for the first time with the specified job.
        # Store the job object, which will be reused for all consequentive runs
        # as long as the job identifier remaisn the same.
        # This job object is uninitialized, so the call to job.evaluateFitness
        # will force initialization for this worker only.
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        with open(new_job_path, 'rb') as f:
            job = pickle.load(f)
        job_path = new_job_path
        job.start()
    result = job.evaluate(parameter_values)
    return parameter_values, result

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

class RunTimeTransform(ParameterTransform):
    def __init__(self, ins, outs):
        self.expressions = []
        self.outvars = []
        for infile, namelist, variable, value in outs:
            self.outvars.append((infile, namelist, variable))
            self.expressions.append(value)
        self.outvars = tuple(self.outvars)

        self.innames = []
        bounds, logscale = {}, {}
        for name, minval, maxval, haslogscale in ins:
            self.innames.append(name)
            bounds[name] = minval, maxval
            logscale[name] = haslogscale
        self.innames = tuple(self.innames)

        ParameterTransform.__init__(self, bounds, logscale)

    def getOriginalParameters(self):
        return self.outvars

    def getExternalParameters(self):
        return self.innames

    def undoTransform(self, values):
        workspace = dict(zip(self.innames, values))
        return tuple([eval(expr, workspace) for expr in self.expressions])

class XMLAttributes():
    def __init__(self, element, description):
        self.att = dict(element.attrib)
        self.description = description
        self.unused = set(self.att.keys())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.unused:
            print('WARNING: the following attributes of %s are ignored: %s' % (self.description, ', '.join(['"%s"' % k for k in self.unused])))

    def get(self, name, type=None, default=None, required=None, minimum=None, maximum=None):
        value = self.att.get(name, None)
        if value is None:
            # No value specified - use default or report error.
            if required is None:
                required = default is None
            if required:
                raise Exception('Attribute "%s" of %s is required.' % (name, self.description))
            value = default
        elif type is bool:
            # Boolean variable
            if value not in ('True', 'False'):
                raise Exception('Attribute "%s" of %s must have value "True" or "False".' % (name, self.description))
            value = value == 'True'
        elif type in (float, int):
            # Numeric variable
            value = type(eval(value))
            if minimum is not None and value < minimum:
                raise Exception('The value of "%s" of %s must exceed %s.' % (name, self.description, minimum))
            if maximum is not None and value > maximum:
                raise Exception('The value of "%s" of %s must lie below %s.' % (name, self.description, maximum))
        elif type is not None:
            value = type(value)
        self.unused.discard(name)
        return value

class Parameter(object):
    def __init__(self, job, att, name=None, default_minimum=None, default_maximum=None):
        self.job = job
        att.description = 'parameter %s' % name
        if name is None:
            self.name = att.get('name')
        else:
            self.name = name
        self.minimum = att.get('minimum', float, default_minimum)
        self.maximum = att.get('maximum', float, default_maximum)
        self.logscale = att.get('logscale', bool, default=False)
        if self.logscale and self.minimum <= 0:
            raise Exception('Minimum for "%s" = %.6g, but should be positive as this parameter is configured to vary on a log scale.' % (self.name, self.minimum))
        if self.maximum < self.minimum:
            raise Exception('Maximum value (%.6g) for "%s" < minimum value (%.6g).' % (self.maximum, self.name, self.minimum))

    def initialize(self):
        pass

    def setValue(self, value):
        pass

    def store(self):
        pass

    def getInfo(self):
        return {'name': self.name, 'minimum': self.minimum, 'maximum': self.maximum, 'logscale': self.logscale}

class DummyParameter(Parameter):
    def __init__(self, job, att):
        Parameter.__init__(self, job, att, name='dummy', default_minimum=0.0, default_maximum=1.0)

class Job(optimize.OptimizationProblem):
    def __init__(self, job_id, xml_tree, root, **kwargs):
        self.started = False
        self.id = job_id

        self.parameters = []
        for ipar, element in enumerate(xml_tree.findall('parameters/parameter')):
            with XMLAttributes(element, 'parameter %i' % (ipar+1)) as att:
                self.parameters.append(self.getParameter(att))

        # Parse transforms
        for ipar, element in enumerate(xml_tree.findall('parameters/transform')):
            with XMLAttributes(element, 'transform %i' % (ipar+1,)) as att:
                pass
            ins, outs = [], []
            for iin, inelement in enumerate(element.findall('in')):
                with XMLAttributes(inelement, 'transform %i, input %i' % (ipar+1, iin+1)) as att:
                    name = att.get('name')
                    att.description = 'transform %i, input %s' % (ipar+1, name)
                    ins.append((name, att.get('minimum', float), att.get('maximum', float), att.get('logscale', bool, default=False)))
            for iout, outelement in enumerate(element.findall('out')):
                with XMLAttributes(outelement, 'transform %i, output %i' % (ipar+1, iout+1)) as att:
                    infile = att.get('file')
                    namelist = att.get('namelist')
                    variable = att.get('variable')
                    att.description = 'transform %i, output %s/%s/%s' % (ipar+1, infile, namelist, variable)
                    outs.append((infile, namelist, variable, att.get('value')))
            tf = RunTimeTransform(ins, outs)
            self.controller.addParameterTransform(tf)

    def start(self, force=False):
        if not self.started or force:
            self.on_start()
            self.started = True

    def evaluateFitness(self, parameter_values):
        return self.evaluate(parameter_values)

    # optionally to be implemented by derived classes
    def on_start(self):
        pass

    # to be implemented by derived classes
    def evaluate(self, parameter_values):
        raise NotImplementedError('Classes deriving from Job must implement "evaluate"')

    def evaluate_ensemble(self, ensemble, ncpus=None, ppservers=(), socket_timeout=600, secret=None, verbose=False):
        results = []
        try:
            import pp
        except ImportError:
            assert ncpus in (1, None) and not ppservers, 'ParallelPython not found. Unable to run ensemble in parallel.'
            ncpus = 1
        if ncpus != 1 or ppservers:
            import time
            import uuid
            try:
                import cPickle as pickle
            except ImportError:
                import pickle
            import atexit

            # Create job server and give it time to connect to nodes.
            if verbose:
                print('Starting Parallel Python server...')
            if ncpus is None:
                ncpus = 'autodetect'
            job_server = pp.Server(ncpus=ncpus, ppservers=ppservers, socket_timeout=socket_timeout, secret=secret)
            if ppservers:
                if verbose:
                    print('Giving Parallel Python 10 seconds to connect to: %s' % (', '.join(ppservers)))
                time.sleep(10)
                if verbose:
                    print('Running on:')
                    for node, ncpu in job_server.get_active_nodes().items():
                        print('   %s: %i cpus' % (node, ncpu))

            # Make sure the population size is a multiple of the number of workers
            nworkers = sum(job_server.get_active_nodes().values())
            if verbose:
                print('Total number of cpus: %i' % nworkers)
            if nworkers == 0:
                raise Exception('No cpus available; exiting.')

            jobpath = os.path.abspath('%s.ppjob' % uuid.uuid4())
            with open(jobpath, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            atexit.register(os.remove, jobpath)

            ppjobs = []
            print('Submitting %i jobs for parallel processing...' % len(ensemble))
            for member in ensemble:
                ppjob = job_server.submit(run_ensemble_member, (jobpath, member))
                ppjobs.append(ppjob)
            for ppjob in ppjobs:
                member, result = ppjob()
                results.append(result)
        else:
            self.start()
            for member in ensemble:
                result = self.evaluate(member)
                results.append(result)
        return results

    def getParameter(self, att):
        if att.get('dummy', bool, default=False):
            return DummyParameter(self, att)
        return Parameter(self, att)

    def getParameterNames(self):
        return tuple([parameter.name for parameter in self.parameters])

    def getParameterBounds(self):
        return numpy.array([parameter.minimum for parameter in self.parameters], dtype=float), numpy.array([parameter.maximum for parameter in self.parameters], dtype=float)

    def getParameterLogScale(self):
        return tuple([parameter.logscale for parameter in self.parameters])

    def createParameterSet(self):
        return numpy.array([0.5*(parameter.minimum+parameter.maximum) for parameter in self.parameters], dtype=float)

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
