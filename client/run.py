#!/usr/bin/python

# Import from standard Python library
import os.path,sys,optparse,datetime

# Import personal custom stuff
import optimizer

import desolver
class Solver(desolver.DESolver):
    def __init__(self,job,*args,**kwargs):
        desolver.DESolver.__init__(self,*args,**kwargs)
        self.job = job
        
    # Functions for random number generation
    def SetupClassRandomNumberMethods(self):
        pass
    def GetClassRandomIntegerBetweenZeroAndParameterCount(self):
        return self.randomstate.random_integers(0, self.parameterCount-1)
    def GetClassRandomFloatBetweenZeroAndOne(self):
        return self.randomstate.uniform()
    def GetClassRandomIntegerBetweenZeroAndPopulationSize(self):
        return self.randomstate.random_integers(0, self.populationSize-1)
        
    def externalEnergyFunction(self,trial):
        return -self.job.evaluateFitness(trial)

    def switchToLocal(self):
        self.job.bufferresults = True

    def getLocalResult(self):
        curqueue = self.job.resultqueue
        self.job.resultqueue = []
        return curqueue

    def processLocalResult(self,resultqueue):
        self.job.resultqueue += resultqueue
        self.job.flushResultQueue()

def getJob(jobid,allowedtransports=None):
    scenpath = os.path.join(os.path.dirname(__file__),'./scenarios/%i' % jobid)

    configpath = os.path.join(scenpath,'config.xml')
    if not os.path.isfile(configpath):
        print 'Configuration file "%s" not found.' % configpath
        return None

    print 'Reading configuration from %s...' % configpath
    return optimizer.Job.fromConfigurationFile(configpath,jobid,scenpath,allowedtransports=allowedtransports)

def main():
    parser = optparse.OptionParser()
    parser.add_option('-m', '--method',      type='choice', choices=('DE','fmin','galileo'), help='Optimization method: DE = Differential Evolution genetic algorithm, fmin = Nelder-Mead simplex, galileo = galileo genetic algorithm')
    parser.add_option('-t', '--transport',   type='choice', choices=('http','mysql'), help='Transport to use for server communication: http or mysql')
    parser.add_option('-i', '--interactive', action='store_true', help='Whether to allow for user interaction (input from stdin) when making decisions')
    parser.add_option('-n', '--ncpus',       type='int', help='Number of CPUs to use (only for Differential Evolution genetic algorithm).')
    parser.add_option('--ppservers',         type='string', help='Comma-separated list of names/IPs of Parallel Python servers to run on.')
    parser.set_defaults(method='DE',transport=None,interactive=False,ncpus=None,ppservers=None)
    (options, args) = parser.parse_args()
    if len(args)<1:
        print 'One argument must be provided: the (integer) job identifier.'
        sys.exit(2)
    jobid = int(args[0])

    allowedtransports = None
    if options.transport is not None:
        allowedtransports = (options.transport,)

    ppservers = ()
    if options.ppservers is not None:
        ppservers = tuple(options.ppservers.split(','))

    job = getJob(jobid,allowedtransports=allowedtransports)
        
    job.interactive = options.interactive

    repeat = True
    while repeat:
        repeat = (options.method!='fmin')   # repeating is only useful for stochastic algorithms - not for deterministic ones
        #job.reportRunStart()

        if options.method=='fmin':
            job.initialize()
            import scipy.optimize.optimize
            xopt = scipy.optimize.optimize.fmin(lambda x: -job.evaluateFitness(x),job.controller.createParameterSet())

            print 'Best parameter set:'
            vals = job.controller.untransformParameterValues(xopt)
            for parinfo,val in zip(job.controller.parameters,vals):
                print '  %s = %.6g' % (parinfo['name'],val)
        elif options.method=='galileo':
            import galileo
            
            popsize = 10*len(job.controller.parameters)
            maxgen = min(popsize,40)

            P = galileo.Population(popsize)
            P.evalFunc = job.evaluateFitness
            P.chromoMinValues,P.chromoMaxValues = job.controller.getParameterBounds()
            P.useInteger = 0
            P.crossoverRate = 1.0
            P.mutationRate = 0.05
            P.selectFunc = P.select_Roulette
            P.replacementSize = P.numChromosomes
            P.crossoverFunc = P.crossover_OnePoint
            P.mutateFunc = P.mutate_Default
            P.replaceFunc = P.replace_SteadyStateNoDuplicates
            P.prepPopulation()

            job.initialize()

            for itn in range(maxgen):
                #evaluate each chromosomes
                P.evaluate()
                #apply selection
                P.select()
                #apply crossover
                P.crossover()
                #apply mutation
                P.mutate()
                #apply replacement
                P.replace()

                print 'Generation %i done. Current best fitness = %.6g.' % (itn,P.maxFitness)
                print 'Best parameter set:'
                vals = job.controller.untransformParameterValues(P.bestFitIndividual.genes)
                for parinfo,val in zip(job.controller.parameters,vals):
                    print '  %s = %.6g' % (parinfo['name'],val)
        elif options.method=='DE':

            minpar,maxpar = job.controller.getParameterBounds()
            
            popsize = 10*len(job.controller.parameters)
            maxgen = 4000
            startpoppath = 'startpop.dat'

            startpop = None
            if os.path.isfile(startpoppath):
                # Retrieve cached copy of the observations
                print 'Reading initial population from file %s...' % startpoppath
                startpop = numpy.load(startpoppath)

            # parameterCount, populationSize, maxGenerations, minInitialValue, maxInitialValue, deStrategy, diffScale, crossoverProb, cutoffEnergy, useClassRandomNumberMethods, polishTheBestTrials
            solver = Solver(job, len(minpar), popsize, maxgen, minpar, maxpar, 'Rand1Exp_jorn', 0.5, 0.9, 0.01, True, False, initialpopulation=startpop, ncpus=options.ncpus, ppservers=ppservers)
            solver.Solve()

            #print 'Generation %i done. Current best fitness = %.6g.' % (itn,P.maxFitness)
            print 'Best parameter set:'
            vals = job.controller.untransformParameterValues(solver.bestSolution)
            for parinfo,val in zip(job.controller.parameters,vals):
                print '  %s = %.6g' % (parinfo['name'],val)

if __name__ == '__main__':
    main()
