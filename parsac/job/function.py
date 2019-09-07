import sys
import os.path

from . import shared

class Job(shared.Job):
    def __init__(self, job_id, xml_tree, root, **kwargs):
        shared.Job.__init__(self, job_id, xml_tree, root)

        element = xml_tree.find('function')
        if element is None:
            raise Exception('The root node must contain a single "function" element.')
        with shared.XMLAttributes(element, 'the function element') as att:
            self.module, self.name = att.get('name').rsplit('.', 1)
            self.path = os.path.join(root, att.get('path', default='.'))
        self.parameter_names = self.getParameterNames()

    def on_start(self):
        sys.path.append(self.path)
        module = __import__(self.module)
        self.function = getattr(module, self.name)

    def evaluate(self, parameter_values):
        parameter2value = dict(zip(self.parameter_names, parameter_values))
        return self.function(**parameter2value)
