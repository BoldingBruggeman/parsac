from __future__ import print_function
import os.path
import importlib
import xml.etree.ElementTree

from . import shared

job_modules = ['idealized', 'function', 'program', 'gotm']

name2class = {}
for name in job_modules:
    try:
        mod = importlib.import_module('.%s' % name, __package__)
        name2class[name] = getattr(mod, 'Job')
    except ImportError as e:
        print('Import failure: %s\nSupport for jobs of type "%s" is disabled.' % (e, name))
        pass

def fromConfigurationFile(path, **kwargs):
    if not os.path.isfile(path):
        print('Configuration file "%s" not found.' % path)
        return None

    xml_tree = xml.etree.ElementTree.parse(path)

    job_id = os.path.splitext(os.path.basename(path))[0]

    element = xml_tree.find('model')
    if element is not None:
        with shared.XMLAttributes(element, 'the model element') as att:
            model_type = att.get('type')
    else:
        model_type = 'gotm'

    assert model_type in name2class, 'Unknown job type "%s" specified.' % model_type

    return name2class[model_type](job_id, xml_tree, os.path.dirname(path), **kwargs)
