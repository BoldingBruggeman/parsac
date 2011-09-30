# Placed into the public domain by:
# James R. Phillips
# 2548 Vera Cruz Drive
# Birmingham, AL 35235 USA
# email: zunzun@zunzun.com

import numpy, random
try:
    import pp # http://www.parallelpython.com - can be single CPU, multi-core SMP, or cluster parallelization
except ImportError:
    pp = None
#pp = None

# NB thie function below runs only in remote worker
def GenerateTrialAndTestInWorker(in_candidate,ref_solver):
    global solver
    try:
        ready = bool(solver)
    except:
        ready = False
    if not ready:
        # This worker is being run fro the first time.
        # Create a copy of the central (virgin) job object.
        # This job object is uninitialized, so the call to solver.EnergyFunction
        # will force initializion for this thread only.
        solver = ref_solver
        solver.randomstate = numpy.random.RandomState(seed=None)
    else:
        # This worker has been used before. Simply update it with the latest population info.
        solver.population = ref_solver.population
        solver.bestEnergy = ref_solver.bestEnergy
        solver.generation = ref_solver.generation

    # Jorn added: multiply and mutate the selected candidate until obtaining an child that differs.
    # This may not happen immediately if the cross-over probability is low.
    while(1):
        # deStrategy is the name of the DE function to use for mutant generation.
        eval('solver.' + solver.deStrategy + '(in_candidate)')
        if (in_candidate!=solver.trialSolution).any(): break

    # Jorn added: reflect parameter values if they have digressed beyond the specified boundaries.
    # This may need to be done multiple times, if the allowed range is small and the parameter deviation large.
    if solver.checkbounds:
        while numpy.logical_or(solver.trialSolution<solver.minInitialValue,solver.trialSolution>solver.maxInitialValue).any():
            print 'Mirroring out-of-bound parameter values.'
            solver.trialSolution = solver.minInitialValue + numpy.abs(solver.minInitialValue-solver.trialSolution)
            solver.trialSolution = solver.maxInitialValue - numpy.abs(solver.maxInitialValue-solver.trialSolution)
        solver.trialSolution = tuple(solver.trialSolution)
            
    # Evaluate the current energy level (*less is better*)
    energy, atSolution = solver.EnergyFunction(solver.trialSolution)
    
    if solver.polishTheBestTrials == True and energy < solver.bestEnergy and solver.generation > 0: # not the first generation
        # try to polish these new coefficients a bit.
        solver.trialSolution = scipy.optimize.fmin(solver.externalEnergyFunction, solver.trialSolution, disp = 0) # don't print warning messages to stdout
        energy, atSolution = solver.EnergyFunction(solver.trialSolution) # recalc with polished coefficients

    return[in_candidate, solver.trialSolution, energy, atSolution]



class DESolver:

    def __init__(self, parameterCount, populationSize, maxGenerations, minInitialValue, maxInitialValue, deStrategy, diffScale, crossoverProb, cutoffEnergy, useClassRandomNumberMethods, polishTheBestTrials, initialpopulation = None):

        self.polishTheBestTrials = polishTheBestTrials # see the Solve method where this flag is used
        self.maxGenerations = maxGenerations
        self.parameterCount = parameterCount
        self.populationSize = populationSize
        self.cutoffEnergy   = cutoffEnergy
        self.minInitialValue = numpy.asarray(minInitialValue)
        self.maxInitialValue = numpy.asarray(maxInitialValue)
        self.deStrategy     = deStrategy # deStrategy is the name of the DE function to use
        self.useClassRandomNumberMethods = useClassRandomNumberMethods

        self.scale = diffScale
        self.crossOverProbability = crossoverProb

        # initial energies for comparison
        self.popEnergy = numpy.ones(self.populationSize) * 1.0E300

        self.bestSolution = numpy.zeros(self.parameterCount)
        self.bestEnergy = 1.0E300

        # Jorn added!
        self.randomstate = numpy.random.RandomState(seed=None)
        self.checkbounds = True
        self.initialpopulation = initialpopulation

        if self.initialpopulation is not None:
            assert self.initialpopulation.ndim==2, 'Initial population must be a NumPy matrix.'
            assert self.initialpopulation.shape[0]>=self.populationSize, 'Initial population must be a matrix with at least %i rows (current row count = %i).' % (self.populationSize,self.initialpopulation.shape[0])
            assert self.initialpopulation.shape[1]==self.parameterCount, 'Initial population must be a matrix with %i columns (current column count = %i).' % (self.parameterCount,self.initialpopulation.shape[1])
            self.initialpopulation = self.initialpopulation[-self.populationSize:,:]
            assert self.initialpopulation.shape[0]==self.populationSize, 'After checking, the initial population must be a matrix with exactly %i rows (current row count = %i).' % (self.populationSize,self.initialpopulation.shape[0])

    def Solve(self):

        breakLoop = False

        # a random initial population, returns numpy arrays directly
        # the population will be synchronized with the remote workers at the beginning of each generation

        # Changed by Jorn!
        #self.population = numpy.random.uniform(self.minInitialValue, self.maxInitialValue, size=(self.populationSize, self.parameterCount))
        if self.initialpopulation is not None:
            self.population = self.initialpopulation
        else:
            self.population = self.randomstate.uniform(self.minInitialValue, self.maxInitialValue, size=(self.populationSize, self.parameterCount))

        job_server = None
        if pp is not None: job_server = pp.Server(ncpus=4) # auto-detects number of SMP CPU cores (will detect 1 core on single-CPU systems)
        
        # try/finally block is to ensure remote worker processes are killed
        try:

            # now run DE
            for self.generation in range(self.maxGenerations):

                # no need to try another generation if we are done
                if breakLoop == True:
                    break # from generation loop
                
                # synchronize the populations for each worker
                if job_server is None:
                    jobs = range(self.populationSize)
                else:
                    jobs = []
                    for candidate in range(self.populationSize):
                        jobs.append(job_server.submit(GenerateTrialAndTestInWorker, (candidate,self), (), ('desolver','numpy.random', 'run','optimizer')))
                        
                # run this generation remotely
                for job in jobs:
                    if job_server is None:
                        candidate, trialSolution, trialEnergy, atSolution = GenerateTrialAndTestInWorker(job,self)
                    else:
                        candidate, trialSolution, trialEnergy, atSolution = job()
                    
                    # if we've reached a sufficient solution we can stop
                    if atSolution == True:
                        breakLoop = True
                        
                    if trialEnergy < self.popEnergy[candidate]:
                        # New low for this candidate
                        self.popEnergy[candidate] = trialEnergy
                        self.population[candidate] = numpy.copy(trialSolution)

                        # If at an all-time low, save to "best"
                        if trialEnergy < self.bestEnergy:
                            self.bestEnergy = self.popEnergy[candidate]
                            self.bestSolution = numpy.copy(self.population[candidate])

        finally:
            if job_server is not None: job_server.destroy()

        return atSolution


    def SetupClassRandomNumberMethods(self):
        assert False,'Base SetupClassRandomNumberMethods should never be called!'
        numpy.random.seed(3) # this yields same results each time Solve() is run
        self.nonStandardRandomCount = self.populationSize * self.parameterCount * 3
        if self.nonStandardRandomCount < 523: # set a minimum number of random numbers
            self.nonStandardRandomCount = 523
            
        self.ArrayOfRandomIntegersBetweenZeroAndParameterCount = numpy.random.random_integers(0, self.parameterCount-1, size=(self.nonStandardRandomCount))
        self.ArrayOfRandomRandomFloatBetweenZeroAndOne = numpy.random.uniform(size=(self.nonStandardRandomCount))
        self.ArrayOfRandomIntegersBetweenZeroAndPopulationSize = numpy.random.random_integers(0, self.populationSize-1, size=(self.nonStandardRandomCount))
        self.randCounter1 = 0
        self.randCounter2 = 0
        self.randCounter3 = 0


    def GetClassRandomIntegerBetweenZeroAndParameterCount(self):
        self.randCounter1 += 1
        if self.randCounter1 >= self.nonStandardRandomCount:
            self.randCounter1 = 0
        return self.ArrayOfRandomIntegersBetweenZeroAndParameterCount[self.randCounter1]

    def GetClassRandomFloatBetweenZeroAndOne(self):
        self.randCounter2 += 1
        if self.randCounter2 >= self.nonStandardRandomCount:
            self.randCounter2 = 0
        return self.ArrayOfRandomRandomFloatBetweenZeroAndOne[self.randCounter2]
        
    def GetClassRandomIntegerBetweenZeroAndPopulationSize(self):
        self.randCounter3 += 1
        if self.randCounter3 >= self.nonStandardRandomCount:
            self.randCounter3 = 0
        return self.ArrayOfRandomIntegersBetweenZeroAndPopulationSize[self.randCounter3]


    # this class might normally be subclassed and this method overridden, or the
    # externalEnergyFunction set and this method used directly
    def EnergyFunction(self, trial):
        try:
            energy = self.externalEnergyFunction(trial)
        except ArithmeticError:
            energy = 1.0E300 # high energies for arithmetic exceptions
        except FloatingPointError:
            energy = 1.0E300 # high energies for floating point exceptions

        # we will be "done" if the energy is less than or equal to the cutoff energy
        if energy <= self.cutoffEnergy:
            return energy, True
        else:
            return energy, False

    def Best1Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] = self.bestSolution[n] + self.scale * (self.population[r1][n] - self.population[r2][n])
            n = (n + 1) % self.parameterCount
            i += 1


    def Rand1Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,0,0)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] = self.population[r1][n] + self.scale * (self.population[r2][n] - self.population[r3][n])
            n = (n + 1) % self.parameterCount
            i += 1

    def Rand1Exp_jorn(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,0,0)

        self.trialSolution = numpy.copy(self.population[candidate])
        for i in range(self.parameterCount):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k <= self.crossOverProbability:
                self.trialSolution[i] = self.population[r1][i] + self.scale * (self.population[r2][i] - self.population[r3][i])

    def RandToBest1Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] += self.scale * (self.bestSolution[n] - self.trialSolution[n]) + self.scale * (self.population[r1][n] - self.population[r2][n])
            n = (n + 1) % self.parameterCount
            i += 1


    def Best2Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,1,0)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] = self.bestSolution[n] + self.scale * (self.population[r1][n] + self.population[r2][n] - self.population[r3][n] - self.population[r4][n])
            n = (n + 1) % self.parameterCount
            i += 1


    def Rand2Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,1,1)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] = self.population[r1][n] + self.scale * (self.population[r2][n] + self.population[r3][n] - self.population[r4][n] - self.population[r5][n])
            n = (n + 1) % self.parameterCount
            i += 1


    def Best1Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] = self.bestSolution[n] + self.scale * (self.population[r1][n] - self.population[r2][n])
            n = (n + 1) % self.parameterCount
            i += 1


    def Rand1Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,0,0)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] = self.population[r1][n] + self.scale * (self.population[r2][n] - self.population[r3][n])
            n = (n + 1) % self.parameterCount
            i += 1


    def RandToBest1Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] += self.scale * (self.bestSolution[n] - self.trialSolution[n]) + self.scale * (self.population[r1][n] - self.population[r2][n])
            n = (n + 1) % self.parameterCount
            i += 1


    def Best2Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,1,0)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] = self.bestSolution[n] + self.scale * (self.population[r1][n] + self.population[r2][n] - self.population[r3][n] - self.population[r4][n])
            n = (n + 1) % self.parameterCount
            i += 1


    def Rand2Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,1,1)
        if True == self.useClassRandomNumberMethods:
            n = self.GetClassRandomIntegerBetweenZeroAndParameterCount()
        else:
            n = random.randint(0, self.parameterCount-1)

        self.trialSolution = numpy.copy(self.population[candidate])
        i = 0
        while(1):
            if True == self.useClassRandomNumberMethods:
                k = self.GetClassRandomFloatBetweenZeroAndOne()
            else:
                k = random.uniform(0.0, 1.0)
            if k >= self.crossOverProbability or i == self.parameterCount:
                break
            self.trialSolution[n] = self.population[r1][n] + self.scale * (self.population[r2][n] + self.population[r3][n] - self.population[r4][n] - self.population[r5][n])
            n = (n + 1) % self.parameterCount
            i += 1


    def SelectSamples(self, candidate, r1, r2, r3, r4, r5):
        if r1:
            while(1):
                if True == self.useClassRandomNumberMethods:
                    r1 = self.GetClassRandomIntegerBetweenZeroAndPopulationSize()
                else:
                    r1 = random.randint(0, self.populationSize-1)
                if r1 != candidate:
                    break
        if r2:
            while(1):
                if True == self.useClassRandomNumberMethods:
                    r2 = self.GetClassRandomIntegerBetweenZeroAndPopulationSize()
                else:
                    r2 = random.randint(0, self.populationSize-1)
                if r2 != candidate and r2 != r1:
                    break
        if r3:
            while(1):
                if True == self.useClassRandomNumberMethods:
                    r3 = self.GetClassRandomIntegerBetweenZeroAndPopulationSize()
                else:
                    r3 = random.randint(0, self.populationSize-1)
                if r3 != candidate and r3 != r1 and r3 != r2:
                    break
        if r4:
            while(1):
                if True == self.useClassRandomNumberMethods:
                    r4 = self.GetClassRandomIntegerBetweenZeroAndPopulationSize()
                else:
                    r4 = random.randint(0, self.populationSize-1)
                if r4 != candidate and r4 != r1 and r4 != r2 and r4 != r3:
                    break
        if r5:
            while(1):
                if True == self.useClassRandomNumberMethods:
                    r5 = self.GetClassRandomIntegerBetweenZeroAndPopulationSize()
                else:
                    r5 = random.randint(0, self.populationSize-1)
                if r5 != candidate and r5 != r1 and r5 != r2 and r5 != r3 and r5 != r4:
                    break

        return r1, r2, r3, r4, r5
