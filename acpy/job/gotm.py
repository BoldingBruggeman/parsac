import os
import datetime
import io

import yaml

from . import program

class Job(program.Job):

    def __init__(self, job_id, xml_tree, root, copyexe=False, tempdir=None, simulationdir=None):
        self.start_time = None
        self.stop_time = None
        program.Job.__init__(self, job_id, xml_tree, root, copyexe, tempdir, simulationdir)

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
                with io.open(gotmyaml_path, 'rU', encoding='utf-8') as f:
                    gotmyaml = yaml.load(f)
                period = gotmyaml['period']
                self.start_time = period['start']
                self.stop_time = period['stop']
            else:
                # Using namelist configuration
                gotmrun_path = os.path.join(self.scenariodir, 'gotmrun.nml')
                nmls, order = program.parseNamelistFile(gotmrun_path)
                assert 'time' in nmls, 'Cannot find namelist named "time" in "%s".' % gotmrun_path
                start_time = nmls['time']['start'][1:-1]
                stop_time = nmls['time']['stop'][1:-1]
                self.start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                self.stop_time = datetime.datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S')

        return self.start_time
