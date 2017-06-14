import sys

import shared

class Job(shared.Job):
    def __init__(self, job_id, xml_tree, root, **kwargs):
        shared.Job.__init__(self, job_id, xml_tree, root)

        element = xml_tree.find('function')
        if element is None:
            raise Exception('The root node must contain a single "function" element.')
        with shared.XMLAttributes(element, 'the function element') as att:
            self.module, self.name = att.get('name', unicode).rsplit('.', 1)
            self.path = att.get('path', unicode, default=root)
        self.parameter_names = self.getParameterNames()
        self.function = None

    def evaluateFitness(self, parameter_values):
        if self.function is None:
            if self.path is not None:
                sys.path.append(self.path)
            module = __import__(self.module)
            self.function = getattr(module, self.name)
        info = dict(zip(self.parameter_names, parameter_values))
        return self.function(**info)
