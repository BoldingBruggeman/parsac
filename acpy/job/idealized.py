from __future__ import print_function

import numpy

from . import shared

class Job(shared.Job):
    def __init__(self, job_id, xml_tree, root, **kwargs):
        shared.Job.__init__(self, job_id, xml_tree, root)

        element = xml_tree.find('target')
        if element is None:
            element = xml_tree.find('fitness')
            if element is not None:
                print('WARNING: XML file does not contain "target" element; using the deprecated "fitness" element instead.')
        if element is None:
            raise Exception('The root node must contain a single "target" element.')
        with shared.XMLAttributes(element, 'the target element') as att:
            self.expression = att.get('expression')

        self.basedict = {}
        for name in dir(numpy):
            obj = getattr(numpy, name)
            if isinstance(obj, numpy.ufunc):
                self.basedict[name] = obj

        self.parameter_names = self.getParameterNames()

    def on_start(self):
        self.expression = compile(self.expression, '<string>', 'eval')

    def evaluate(self, parameter_values):
        for name, value in zip(self.parameter_names, parameter_values):
            self.basedict[name] = value
        return eval(self.expression, {}, self.basedict)
