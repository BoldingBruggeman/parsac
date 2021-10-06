import os
import datetime
import io
import shutil
import fnmatch

import yaml
import numpy
import netCDF4

from . import shared
from . import program

class BackupRestart(shared.Function):
    def initialize(self):
        restart = os.path.join(self.job.scenariodir, 'restart.nc')
        self.active = os.path.isfile(restart)
        if self.active:
            print('Backing up restart.nc')
            backup = os.path.join(self.job.scenariodir, 'restart.nc.bck')
            if os.path.isfile(backup):
                os.remove(backup)
            shutil.copy(restart, backup)

    def apply(self):
        if self.active:
            print('Copying in clean restart')
            target = os.path.join(self.job.scenariodir, 'restart.nc')
            if os.path.isfile(target):
                os.remove(target)
            shutil.copy(os.path.join(self.job.scenariodir, 'restart.nc.bck'), target)

class ChangeRestart(shared.Function):
    def __init__(self, job, att):
        shared.Function.__init__(self, job)
        self.variable = att.get('variable', str)
        self.expression = att.get('expression', str)
        self.mindepth = att.get('mindepth', float, -numpy.inf)
        self.maxdepth = att.get('maxdepth', float, numpy.inf)

    def apply(self):
        with netCDF4.Dataset(os.path.join(self.job.scenariodir, 'restart.nc'), 'r+') as nc:
            for name in fnmatch.filter(nc.variables.keys(), self.variable):
                z = nc.variables['z'][0, :, 0, 0]   # GOTM depth is NEGATIVE
                kstart, kstop = z.searchsorted((-self.maxdepth, -self.mindepth))
                ncvar = nc.variables[name]
                values = ncvar[0, kstart:kstop, 0, 0]
                m = self.getParameterMapping({'variable': values})
                newvalues = eval(self.expression, {}, m)
                print('Setting %s = %s (%i values, mean=%s)' % (name, self.expression, newvalues.size, numpy.mean(newvalues)))
                ncvar[0, kstart:kstop, 0, 0] = newvalues

class Job(program.Job):

    def __init__(self, job_id, xml_tree, root, **kwargs):
        self.start_time = None
        self.stop_time = None
        program.Job.__init__(self, job_id, xml_tree, root, **kwargs)
        self.functions.insert(0, BackupRestart(self))

    def getSimulationStart(self):
        if self.start_time is None:
            # Find start and stop of simulation.
            # These will be used to prune the observation table.

            # Check for existence of scenario directory
            # (we need it now already to find start/stop of simulation)
            if not os.path.isdir(self.scenariodir):
                raise Exception('GOTM scenario directory "%s" does not exist.' % self.scenariodir)

            # Parse GOTM configuration file with settings specifying the simulated period.
            gotmyaml_path = os.path.join(self.scenariodir, 'gotm.yaml')
            if os.path.isfile(gotmyaml_path):
                # Using yaml configuration
                with io.open(gotmyaml_path, 'rU') as f:
                    gotmyaml = yaml.safe_load(f)
                period = gotmyaml['time']
                self.start_time = period['start']
                self.stop_time = period['stop']
            else:
                # Using namelist configuration
                gotmrun_path = os.path.join(self.scenariodir, 'gotmrun.nml')
                nmls, _ = program.parseNamelistFile(gotmrun_path)
                assert 'time' in nmls, 'Cannot find namelist named "time" in "%s".' % gotmrun_path
                start_time = nmls['time']['start'][1:-1]
                stop_time = nmls['time']['stop'][1:-1]
                self.start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                self.stop_time = datetime.datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S')

        return self.start_time
