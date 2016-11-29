import os.path

import numpy

try:
    from .. import optimize
except ValueError:
    import optimize

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

    def get(self, name, type, default=None, required=None, minimum=None, maximum=None):
        value = self.att.get(name, None)
        if value is None:
            # No value specified - use default or report error.
            if required is None: required = default is None
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
        else:
            # String
            value = type(value)
        self.unused.discard(name)
        return value

    def testEmpty(self):
        if self.unused:
            print 'WARNING: the following attributes of %s are ignored: %s' % (self.description, ', '.join(['"%s"' % k for k in self.unused]))

class Parameter(object):
    def __init__(self, name, att, default_minimum=None, default_maximum=None):
        att.description = 'parameter %s' % name
        self.name = name
        self.minimum = att.get('minimum', float, default_minimum)
        self.maximum = att.get('maximum', float, default_maximum)
        self.logscale = att.get('logscale', bool, default=False)
        if self.logscale and self.minimum <= 0:
            raise Exception('Minimum for "%s" = %.6g, but should be positive as this parameter is configured to vary on a log scale.' % (self.name, self.minimum))
        if self.maximum < self.minimum:
            raise Exception('Maximum value (%.6g) for "%s" < minimum value (%.6g).' % (self.maximum, self.name, self.minimum))

    def getInfo(self):
        return {'name': self.name, 'minimum': self.minimum, 'maximum': self.maximum, 'logscale': self.logscale}

class DummyParameter(Parameter):
    def __init__(self, att):
        Parameter.__init__(self, 'dummy', att, default_minimum=0.0, default_maximum=1.0)

class Job(optimize.OptimizationProblem):
    def __init__(self, job_id, xml_tree, root, **kwargs):
        self.initialized = False
        self.id = job_id

        self.parameters = []
        for ipar, element in enumerate(xml_tree.findall('parameters/parameter')):
            att = XMLAttributes(element, 'parameter %i' % (ipar+1))
            self.parameters.append(self.getParameter(att))
            att.testEmpty()

        # Parse transforms
        for ipar, element in enumerate(xml_tree.findall('parameters/transform')):
            att = XMLAttributes(element, 'transform %i' % (ipar+1,))
            att.testEmpty()
            ins, outs = [], []
            for iin, inelement in enumerate(element.findall('in')):
                att = XMLAttributes(inelement, 'transform %i, input %i' % (ipar+1, iin+1))
                name = att.get('name', unicode)
                att.description = 'transform %i, input %s' % (ipar+1, name)
                ins.append((name, att.get('minimum', float), att.get('maximum', float), att.get('logscale', bool, default=False)))
                att.testEmpty()
            for iout, outelement in enumerate(element.findall('out')):
                att = XMLAttributes(outelement, 'transform %i, output %i' % (ipar+1, iout+1))
                infile = att.get('file', unicode)
                namelist = att.get('namelist', unicode)
                variable = att.get('variable', unicode)
                att.description = 'transform %i, output %s/%s/%s' % (ipar+1, infile, namelist, variable)
                outs.append((infile, namelist, variable, att.get('value', unicode)))
                att.testEmpty()
            tf = RunTimeTransform(ins, outs)
            self.controller.addParameterTransform(tf)

    def getParameter(self, att):
        if att.get('dummy', bool, default=False):
            return DummyParameter(att)
        return Parameter(att.get('name', unicode), att)

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
