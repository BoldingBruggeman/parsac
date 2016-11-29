import os
import datetime

import program

class Job(program.Job):
    verbose = True

    def __init__(self, job_id, xml_tree, root, copyexe=False, tempdir=None, simulationdir=None):
        self.start = None
        program.Job.__init__(self, job_id, xml_tree, root, copyexe, tempdir, simulationdir)

    def getInfo(self):
        return {'parameters': self.externalparameters}

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

    def getSimulationStart(self):
        if self.start is None:
            # Check for existence of scenario directory
            # (we need it now already to find start/stop of simulation)
            if not os.path.isdir(self.scenariodir):
                raise Exception('GOTM scenario directory "%s" does not exist.' % self.scenariodir)

            # Parse file with namelists describing the main scenario settings.
            path = os.path.join(self.scenariodir, 'gotmrun.nml')
            nmls, order = program.parseNamelistFile(path)
            assert 'time' in nmls, 'Cannot find namelist named "time" in "%s".' % path

            # Find start and stop of simulation.
            # These will be used to prune the observation table.
            datematch = program.datetimere.match(nmls['time']['start'][1:-1])
            assert datematch is not None, 'Unable to parse start datetime in "%s".' % nmls['time']['start'][1:-1]
            self.start = datetime.datetime(*map(int, datematch.group(1, 2, 3, 4, 5, 6)))
            datematch = program.datetimere.match(nmls['time']['stop'][1:-1])
            assert datematch is not None, 'Unable to parse stop datetime in "%s".' % nmls['time']['stop'][1:-1]
            self.stop = datetime.datetime(*map(int, datematch.group(1, 2, 3, 4, 5, 6)))

        return self.start
