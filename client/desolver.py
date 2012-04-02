import time
import numpy

try:
    import pp # http://www.parallelpython.com - can be single CPU, multi-core SMP, or cluster parallelization
except ImportError:
    pp = None
#pp = None

if pp is not None:
    # Override Parallel Pyton timeout
    import pptransport
    pptransport.TRANSPORT_SOCKET_TIMEOUT = 600

def processTrial(newjobid,newjob,trial):
    global jobid,job

    if 'jobid' not in globals() or jobid!=newjobid:
        # This worker is being run for the first time with the specified job.
        # Store the job object, which will be reused for all consequentive runs
        # as long as the job identifier remaisn the same.
        # This job object is uninitialized, so the call to job.evaluateFitness
        # will force initialization for this worker only.
        job,jobid = newjob,newjobid

    # Evaluate the cost function (*less is better*)
    score = -job.evaluateFitness(trial)

    return trial,score

class DESolver:

    def __init__(self, job, populationSize, maxGenerations, minInitialValue, maxInitialValue, diffScale, crossoverProb,
                 initialpopulation = None, ncpus=None, ppservers=(), reporter = None, functions = (), modules =()):
        # Store job (with fitness function) and reporter objects.
        self.job = job
        self.reporter = reporter

        # Check whether minimum and maximum vector have equal length.
        assert len(minInitialValue)==len(maxInitialValue),'Lengths of minimum and maximum vectors do not match.'

        # Constrains on parameters, population size, generation count.
        self.minInitialValue = numpy.asarray(minInitialValue)
        self.maxInitialValue = numpy.asarray(maxInitialValue)
        self.parameterCount = len(minInitialValue)
        self.populationSize = populationSize
        self.maxGenerations = maxGenerations

        # Differential Evolution scale and cross-over probability parameters.
        self.scale = diffScale
        self.crossOverProbability = crossoverProb

        # Whether to force parameetr vectors to stay withi specified bounds.
        self.strictbounds = True

        # Parallel Python settings
        if ncpus is None: ncpus = 'autodetect'
        self.ncpus = ncpus
        self.ppservers = ppservers

        # Store initial population (if provided) and perform basic checks.
        self.initialpopulation = initialpopulation
        if self.initialpopulation is not None:
            assert self.initialpopulation.ndim==2, 'Initial population must be a NumPy matrix.'
            assert self.initialpopulation.shape[1]==self.parameterCount, 'Initial population must be a matrix with %i columns (current column count = %i).' % (self.parameterCount,self.initialpopulation.shape[1])

        # Create random number generator.
        self.randomstate = numpy.random.RandomState(seed=None)

        self.functions = functions
        self.modules = modules

    def Solve(self):

        job_server = None
        if pp is not None:
            # Create job server and give it time to conenct to nodes.
            job_server = pp.Server(ncpus=self.ncpus,ppservers=self.ppservers)
            if self.ppservers:
                print 'Giving Parallel Python 5 seconds to connect to nodes...'
                time.sleep(5)
                
            # Make sure the population size is a multiple of the number of workers
            nworkers = sum(job_server.get_active_nodes().values())
            jobsperworker = int(numpy.round(self.populationSize/float(nworkers)))
            if self.populationSize!=jobsperworker*nworkers:
                print 'Setting population size to %i (was %i) to ensure it is a multiple of number of workers (%i).' % (jobsperworker*nworkers,self.populationSize,nworkers)
                self.populationSize = jobsperworker*nworkers

        # Create initial population.
        if self.initialpopulation is not None:
            assert self.initialpopulation.shape[0]>=self.populationSize, 'Initial population must be a matrix with at least %i rows (current row count = %i).' % (self.populationSize,self.initialpopulation.shape[0])
            self.population = self.initialpopulation[-self.populationSize:,:]
        else:
            self.population = self.randomstate.uniform(self.minInitialValue, self.maxInitialValue, size=(self.populationSize, self.parameterCount))
        cost = numpy.empty(self.population.shape[0])
        cost[:] = numpy.Inf

        ibest = None

        # Create unique job id that will be used to check worker job ids against.
        # (this allows the worker to detect stale job objects)
        jobid = self.randomstate.rand()
        
        # try/finally block is to ensure remote worker processes are killed
        try:

            for igeneration in range(self.maxGenerations):
                
                # Generate list with target,trial vector combinations to try.
                # If using Parallel Python, submit these trials to the workers.
                trials = []
                for itarget in range(self.population.shape[0]):
                    trial = self.generateNew(itarget,ibest,F=self.scale,CR=self.crossOverProbability,strictbounds=self.strictbounds)
                    if job_server is not None:
                        trial = job_server.submit(processTrial, (jobid,self.job,trial), self.functions, self.modules)
                    trials.append(trial)

                # Process the individual target,trial combinations.
                for itarget,trial in enumerate(trials):
                    if job_server is None:
                        trial,score = processTrial(jobid,self.job,trial)
                    else:
                        trial,score = trial()

                    if self.reporter is not None: self.reporter.reportResult(trial,-score)

                    # Determine whether trial vector is better than target vector.
                    # If so, replace target with trial.
                    if score<=cost[itarget]:
                        self.population[itarget,:] = trial
                        cost[itarget] = score
                        if ibest is None or score<cost[ibest]: ibest = itarget
        finally:
            if job_server is not None: job_server.destroy()

        return False

    def drawVectors(self,n,exclude=()):
        """Draws n vectors at random from the population, ensuring they
        do not overlap.
        """
        vectors = []
        excluded = list(exclude)
        for i in range(n):
            ind = self.randomstate.randint(self.population.shape[0]-len(excluded))
            excluded.sort()
            for oldind in excluded:
                if ind>=oldind: ind += 1
            vectors.append(self.population[ind,:])
            excluded.append(ind)
        return vectors

    def generateNew(self,itarget,ibest,CR=0.9,strictbounds=True,ndiffvector=1,F=0.5,randomancestor=True):
        """Generates a new trial vector according to the Differential Evolution
        algorithm.

        Details in Storn, R. & Price, K. 1997. Differential evolution - A simple
        and efficient heuristic for global optimization over continuous spaces.
        Journal of Global Optimization 11:341-59.
        """
        if ibest is None: randomancestor = True
        
        # Draw random vectors
        if randomancestor:
            vectors = self.drawVectors(ndiffvector*2+1,exclude=(itarget,))
        else:
            vectors = self.drawVectors(ndiffvector*2,  exclude=(itarget,ibest))

        # Determine target vector
        if randomancestor:
            # Randomly picked ancestor
            ref = vectors.pop()
        else:
            # Ancestor is current best
            ref = self.population[ibest,:]

        # Create difference vector
        delta = numpy.zeros_like(ref)
        for i in range(ndiffvector):
            r1 = vectors.pop()
            r2 = vectors.pop()
            delta += r1-r2
        mutant = ref + F*delta

        # Cross-over
        cross = self.randomstate.random_sample(delta.size)<=CR
        cross[self.randomstate.randint(delta.size)] = True
        trial = numpy.where(cross,mutant,self.population[itarget,:])

        # Reflect parameter values if they have digressed beyond the specified boundaries.
        # This may need to be done multiple times, if the allowed range is small and the parameter deviation large.
        if strictbounds:
            while numpy.logical_or(trial<self.minInitialValue,trial>self.maxInitialValue).any():
                trial = self.minInitialValue + numpy.abs(self.minInitialValue-trial)
                trial = self.maxInitialValue - numpy.abs(self.maxInitialValue-trial)

        return trial

