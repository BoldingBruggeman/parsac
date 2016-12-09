import numpy

import shared

class Job(shared.Job):
    def __init__(self, job_id, xml_tree, root, **kwargs):
        shared.Job.__init__(self, job_id, xml_tree, root)

        element = xml_tree.find('fitness')
        if element is None:
            raise Exception('The root node must contain a single "fitness" element.')
        att = shared.XMLAttributes(element, 'the fitness element')
        self.expression = att.get('expression', unicode)

        self.basedict = {}
        for name in dir(numpy):
            obj = getattr(numpy, name)
            if isinstance(obj, numpy.ufunc):
                self.basedict[name] = obj

        self.parameter_names = self.getParameterNames()

    def evaluateFitness(self, parameter_values):
        for name, value in zip(self.parameter_names, parameter_values):
            self.basedict[name] = value
        return eval(self.expression, {}, self.basedict)
