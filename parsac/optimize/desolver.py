from __future__ import print_function
import os
import time
import numpy
import atexit
import re

minppversion = '1.6.2'

try:
    import pp # http://www.parallelpython.com - can be single CPU, multi-core SMP, or cluster parallelization
except ImportError:
    pp = None

if pp is not None:
    ver = map(int, pp.version.split('.'))
    for v, tv in zip(ver, map(int, minppversion.split('.'))):
        if v > tv:
            break
        if v < tv:
            print('Old Parallel Python version %s. Minimum required = %s.' % (pp.version, minppversion))
            print('Parallel Python support for Differential Evolution is disabled.')
            pp = None

jobid = None
def processTrial(newjobid, newjob, trial):
    global jobid, job

    if 'jobid' not in globals() or jobid != newjobid:
        # This worker is being run for the first time with the specified job.
        # Store the job object, which will be reused for all consequentive runs
        # as long as the job identifier remaisn the same.
        # This job object is uninitialized, so the call to job.evaluateFitness
        # will force initialization for this worker only.
        if isinstance(newjob, (str, u''.__class__)):
            try:
                import cPickle as pickle
            except ImportError:
                import pickle
            with open(newjob, 'rb') as f:
                newjob = pickle.load(f)
        job, jobid = newjob, newjobid
        newjob.start()

    # Evaluate the fitness function (*more is better*)
    fitness = job.evaluateFitness(trial)

    return trial, fitness

class DESolver:

    def __init__(self, job, populationSize, maxGenerations, minInitialValue, maxInitialValue, F, CR,
                 initialpopulation = None, ncpus=None, ppservers=(), reporter = None, functions = (), modules =(),
                 reltol=0.01, abstol=1e-8, ftol=numpy.inf, verbose=True, strictbounds=True, socket_timeout=600, secret=None):
        # Store job (with fitness function) and reporter objects.
        self.job = job
        self.reporter = reporter

        # Check whether minimum and maximum vector have equal length.
        assert len(minInitialValue) == len(maxInitialValue), 'Lengths of minimum and maximum vectors do not match.'

        # Constrains on parameters, population size, generation count.
        self.minInitialValue = numpy.asarray(minInitialValue)
        self.maxInitialValue = numpy.asarray(maxInitialValue)
        self.parameterCount = len(minInitialValue)
        self.populationSize = populationSize
        self.maxGenerations = maxGenerations

        # Differential Evolution scale and cross-over probability parameters.
        self.scale = F
        self.crossOverProbability = CR

        # Whether to force parameter vectors to stay within specified bounds.
        self.strictbounds = strictbounds

        # Parallel Python settings
        if isinstance(ppservers, (str, u''.__class__)):
            match = re.match(r'(.*)\[(.*)\](.*)', ppservers)
            if match is not None:
                # Hostnames in PBS/SLURM notation, e.g., node[01-06]
                ppservers = []
                left, middle, right = match.groups()
                for item in middle.split(','):
                    if '-' in item:
                        start, stop = item.split('-')
                        for i in range(int(start), int(stop)+1):
                            ppservers.append('%s%s%s' % (left, str(i).zfill(len(start)), right))
                    else:
                        ppservers.append('%s%s%s' % (left, item, right))
                ppservers = tuple(ppservers)
            else:
                # Comma-separated hostnames
                ppservers = ppservers.split(',')
        elif ppservers is None:
            ppservers = ()

        if ncpus is None:
            ncpus = 'autodetect'
        else:
            if ncpus == 1 and not ppservers:
                print('Parallelization of Differential Evolution is disabled because number of cores is set to 1.')
            else:
                print('Local number of cores for Differential Evolution set to %i by user.' % ncpus)
        self.ncpus = ncpus
        self.ppservers = ppservers
        self.secret = secret
        self.socket_timeout = socket_timeout

        if pp is None:
            if self.ncpus != 1 or ppservers:
                print('Parallelization of Differential Evolution is disabled because Parallel Python is not available or the wrong version.')
            self.ncpus = 1
            self.ppservers = ()

        # Store initial population (if provided) and perform basic checks.
        self.initialpopulation = initialpopulation
        if self.initialpopulation is not None:
            assert self.initialpopulation.ndim == 2, 'Initial population must be a NumPy matrix.'
            assert self.initialpopulation.shape[1] == self.parameterCount, 'Initial population must be a matrix with %i columns (current column count = %i).' % (self.parameterCount, self.initialpopulation.shape[1])

        # Create random number generator.
        self.randomstate = numpy.random.RandomState(seed=None)

        # Function and module dependencies that need to be reported to Parallel Python.
        self.functions = functions
        self.modules = modules

        # Absolute and relative tolerances used to stop optimization
        self.reltol = reltol
        self.abstol = abstol
        self.ftol = ftol

        self.verbose = verbose

        #self.picked = dict([(i,0) for i in range(self.populationSize)])

    def Solve(self):

        job_server = None
        if self.ncpus != 1 or self.ppservers:
            # Create job server and give it time to conenct to nodes.
            if self.verbose:
                ('Starting Parallel Python server...')
            job_server = pp.Server(ncpus=self.ncpus, ppservers=self.ppservers, socket_timeout=self.socket_timeout, secret=self.secret)
            if self.ppservers:
                if self.verbose:
                    print('Giving Parallel Python 10 seconds to connect to: %s' % (', '.join(self.ppservers)))
                time.sleep(10)
                if self.verbose:
                    print('Running on:')
                    for node, ncpu in job_server.get_active_nodes().items():
                        print('   %s: %i cpus' % (node, ncpu))

            # Make sure the population size is a multiple of the number of workers
            nworkers = sum(job_server.get_active_nodes().values())
            if self.verbose:
                print('Total number of cpus: %i' % nworkers)
            if nworkers == 0:
                raise Exception('No cpus available; exiting.')
            jobsperworker = int(round(self.populationSize/float(nworkers)))
            if self.populationSize != jobsperworker*nworkers:
                if self.verbose:
                    print('Setting population size to %i (was %i) to ensure it is a multiple of number of workers (%i).' % (jobsperworker*nworkers, self.populationSize, nworkers))
                self.populationSize = jobsperworker*nworkers

        # Create initial population.
        if self.initialpopulation is not None:
            assert self.initialpopulation.shape[0] >= self.populationSize, 'Initial population must be a matrix with at least %i rows (current row count = %i).' % (self.populationSize, self.initialpopulation.shape[0])
            self.population = self.initialpopulation[-self.populationSize:, :]
        else:
            self.population = self.randomstate.uniform(self.minInitialValue, self.maxInitialValue, size=(self.populationSize, self.parameterCount))
        fitness = numpy.empty(self.population.shape[0])
        fitness[:] = -numpy.inf

        ibest = None

        # Create unique job id that will be used to check worker job ids against.
        # (this allows the worker to detect stale job objects)
        jobid = self.randomstate.rand()

        if job_server is not None:
            try:
                import cPickle as pickle
            except ImportError:
                import pickle
            jobpath = '%s.ppjob' % jobid
            with open(jobpath, 'wb') as f:
                pickle.dump(self.job, f, pickle.HIGHEST_PROTOCOL)
            self.job = os.path.abspath(jobpath)
            atexit.register(os.remove, self.job)

        ppcallback = None
        if self.reporter is not None:
            ppcallback = lambda arg: self.reporter.reportResult(*arg)

        # try/finally block is to ensure remote worker processes are killed
        try:
            igeneration = 1
            while 1:

                # Generate list with target,trial vector combinations to try.
                # If using Parallel Python, submit these trials to the workers.
                trials = []
                for itarget in range(self.population.shape[0]):
                    trial = self.generateNew(itarget, ibest, F=self.scale, CR=self.crossOverProbability, strictbounds=self.strictbounds)
                    if job_server is not None:
                        trial = job_server.submit(processTrial, (jobid, self.job, trial), self.functions, self.modules, callback=ppcallback)
                    trials.append(trial)

                # Process the individual target,trial combinations.
                for itarget, trial in enumerate(trials):
                    if job_server is None:
                        # No parallelization: "trial" is the parameter vector to be tested.
                        trial, trialfitness = processTrial(jobid, self.job, trial)
                        if self.reporter is not None:
                            self.reporter.reportResult(trial, trialfitness)
                    else:
                        # Parallelization: "trial" is the Parallel Python job processing the new parameter vector.
                        # This job returns the tested parameter vector, together with its fitness.
                        # Reporting of the result is already done by the callback routine provided to submit.
                        trial, trialfitness = trial()
                    if isinstance(trialfitness, tuple):
                        trialfitness = trialfitness[0]

                    # Determine whether trial vector is better than target vector.
                    # If so, replace target with trial.
                    if trialfitness >= fitness[itarget]:
                        self.population[itarget, :] = trial
                        fitness[itarget] = trialfitness
                        if ibest is None or trialfitness > fitness[ibest]:
                            ibest = itarget

                curminpar = self.population.min(axis=0)
                curmaxpar = self.population.max(axis=0)
                currange = curmaxpar-curminpar
                curcent = 0.5*(curmaxpar+curminpar)
                tol = numpy.maximum(self.abstol, abs(curcent)*self.reltol)
                frange = (fitness[ibest] - fitness).max()
                if self.verbose:
                    print('Finished generation %i' % igeneration)
                    print('  Range:     %s' % ', '.join(['%.2e' % v for v in currange]))
                    print('  Tolerance: %s' % ', '.join(['%.2e' % v for v in tol]))
                    print('  Fitness range: %s' % frange)
                if igeneration == self.maxGenerations or ((currange <= tol).all() and frange <= self.ftol):
                    break
                igeneration += 1
                #dup = False
                #for ipar in range(len(curminpar)):
                #    if len(set(self.population[:,ipar]))!=self.population.shape[0]:
                #        print('WARNING: duplicates detected in generation %i, parameter %i:' % (igeneration,ipar))
                #        print(self.population[:,ipar])
                #        dup = True
                #if dup: break
        finally:
            if job_server is not None:
                job_server.destroy()

        #print(self.picked)

        return self.population[ibest, :]

    def drawVectors(self, n, exclude=()):
        """Draws n vectors at random from the population, ensuring they
        do not overlap.
        """
        vectors = []
        excluded = list(exclude)
        for _ in range(n):
            ind = self.randomstate.randint(self.population.shape[0]-len(excluded))
            excluded.sort()
            for oldind in excluded:
                if ind >= oldind: ind += 1
            vectors.append(ind)
            excluded.append(ind)
        #for ind in vectors: assert ind not in exclude
        #assert len(frozenset(vectors))==len(vectors)
        #for ind in vectors: self.picked[ind] += 1
        return [self.population[ind, :] for ind in vectors]

    def generateNew(self, itarget, ibest, CR=0.9, strictbounds=True, ndiffvector=1, F=0.5, randomancestor=True):
        """Generates a new trial vector according to the Differential Evolution
        algorithm.

        Details in Storn, R. & Price, K. 1997. Differential evolution - A simple
        and efficient heuristic for global optimization over continuous spaces.
        Journal of Global Optimization 11:341-59.
        """
        if ibest is None: randomancestor = True
        #print('CR=%.1f,F=%.1f,strictbounds=%s,randomancestor=%s,ndiffvector=%i' % (CR,F,strictbounds,randomancestor,ndiffvector))

        # Draw random vectors
        if randomancestor:
            vectors = self.drawVectors(ndiffvector*2+1, exclude=(itarget,))
        else:
            vectors = self.drawVectors(ndiffvector*2, exclude=(itarget, ibest))

        # Determine base vector to mutate
        if randomancestor:
            # Randomly picked ancestor
            ref = vectors.pop()
        else:
            # Ancestor is current best
            ref = self.population[ibest, :]

        # Mutate base vector
        delta = numpy.zeros_like(ref)
        for _ in range(ndiffvector):
            r1 = vectors.pop()
            r2 = vectors.pop()
            delta += r1-r2
        #assert (delta!=0.).all(),'%s %s' % (r1,r2)
        mutant = ref + F*delta

        # Cross-over
        cross = self.randomstate.random_sample(delta.size) < CR
        cross[self.randomstate.randint(delta.size)] = True
        trial = numpy.where(cross, mutant, self.population[itarget, :])

        # Reflect parameter values if they have digressed beyond the specified boundaries.
        # This may need to be done multiple times, if the allowed range is small and the parameter deviation large.
        if strictbounds:
            while numpy.logical_or(trial < self.minInitialValue, trial > self.maxInitialValue).any():
                trial = self.minInitialValue + numpy.abs(self.minInitialValue-trial)
                trial = self.maxInitialValue - numpy.abs(self.maxInitialValue-trial)

        return trial

