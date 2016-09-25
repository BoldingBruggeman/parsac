#!/usr/bin/env python

# Import from standard Python library
import os.path,sys,optparse

# Import personal custom stuff
import optimize
import optimizer

def getJob(configpath,returnreporter=False,allowedtransports=None,tempdir=None,simulationdir=None):
    if not os.path.isfile(configpath):
        print 'Configuration file "%s" not found.' % configpath
        return None

    print 'Reading configuration from %s...' % configpath
    job = optimizer.Job.fromConfigurationFile(configpath,tempdir=tempdir,simulationdir=simulationdir)
    if returnreporter:
        f = open(configpath)
        xml = f.read()
        f.close()
        reporter = optimizer.Reporter.fromConfigurationFile(configpath,xml,allowedtransports=allowedtransports)
        return job,reporter
    return job

def main():
    parser = optparse.OptionParser()
    parser.add_option('-m', '--method',         type='choice', choices=('DE','fmin','galileo'), help='Optimization method: DE = Differential Evolution genetic algorithm, fmin = Nelder-Mead simplex, galileo = galileo genetic algorithm')
    parser.add_option('-t', '--transport',      type='choice', choices=('http','mysql'), help='Transport to use for server communication: http or mysql')
    parser.add_option('-r', '--reportfrequency',type='int',    help='Time between result reports (seconds).')
    parser.add_option('-i', '--interactive',    action='store_true', help='Whether to allow for user interaction (input from stdin) when making decisions')
    parser.add_option('-n', '--ncpus',          type='int',    help='Number of CPUs to use (only for Differential Evolution genetic algorithm).')
    parser.add_option('--tempdir',              type='string', help='Temporary directory for GOTM setups.')
    parser.add_option('--ppservers',            type='string', help='Comma-separated list of names/IPs of Parallel Python servers to run on (only for Differential Evolution genetic algorithm).')
    parser.set_defaults(method='DE',transport=None,interactive=False,ncpus=None,ppservers=None,reportfrequency=None,tempdir=None,scenarios='.')
    (options, args) = parser.parse_args()
    if len(args)<1:
        print 'One argument must be provided: path to job configuration file (xml).'
        sys.exit(2)
    jobid = os.path.splitext(os.path.basename(args[0]))[0]

    allowedtransports = None
    if options.transport is not None:
        allowedtransports = (options.transport,)

    ppservers = ()
    if options.ppservers is not None:
        ppservers = tuple(options.ppservers.split(','))

    job,reporter = getJob(args[0],returnreporter=True,allowedtransports=allowedtransports,tempdir=options.tempdir)

    # Configure result reporter
    reporter.interactive = options.interactive
    if options.reportfrequency is not None: reporter.timebetweenreports = options.reportfrequency

    opt = optimize.Optimizer(job, reportfunction=reporter.reportResult)

    repeat = True
    while repeat:
        repeat = (options.method!='fmin')   # repeating is only useful for stochastic algorithms - not for deterministic ones

        if options.method=='fmin':
            job.initialize()
            vals = opt.run(method=optimize.SIMPLEX,par_ini=job.controller.createParameterSet())
        elif options.method=='galileo':
            def fitnessFunction(x):
                fitness = job.evaluateFitness(x)
                reporter.reportResult(x,fitness)
                return fitness

            import galileo
            
            popsize = 10*len(job.controller.parameters)
            maxgen = min(popsize,40)

            P = galileo.Population(popsize)
            P.evalFunc = fitnessFunction
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
            vals = opt.run(method=optimize.DIFFERENTIALEVOLUTION,par_min=minpar,par_max=maxpar,popsize=popsize,maxgen=maxgen,F=0.5,CR=0.9,initialpopulation=startpop,ncpus=options.ncpus, ppservers=ppservers, modules=('run',))

            #print 'Generation %i done. Current best fitness = %.6g.' % (itn,P.maxFitness)

        print 'Best parameter set:'
        vals = job.controller.untransformParameterValues(vals)
        for parinfo,val in zip(job.controller.parameters,vals):
              print '  %s = %.6g' % (parinfo['name'],val)

if __name__ == '__main__':
    main()
