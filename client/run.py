#!/usr/bin/python

# Import from standard Python library
import os.path,sys,optparse,datetime

# Import personal custom stuff
import gotmcontroller,optimizer,transport

def getJob(jobid):
    scenpath = os.path.join(os.path.dirname(__file__),'./scenarios/%i' % jobid)

    configpath = os.path.join(scenpath,'config.xml')
    if os.path.isfile(configpath):
        print 'Reading run configuration from %s...' % configpath
        job = optimizer.Job.fromConfigurationFile(configpath,jobid,scenpath)
    else:
        print 'Configuration from %s not found - using old method...' % configpath

        obsdir = './obs'
        if sys.platform=='win32':
            gotmexe = '../../gotm-cur/Release/gotm-cur.exe'
            gotmexe = 'C:/Users/Jorn/Documents/gotm/compilers/vs2008/Release/gotm/gotm.exe'
        else:
            gotmexe = '../../gotm-cur/gotm/src/gotm_prod_IFORT'

        # If we have been frozen and distributed, paths change slightly.
        if hasattr(sys,'frozen'):
            scenpath = './scenario'
            gotmexe = os.path.basename(gotmexe)
            obsdir = 'obs'
        else:
            obsdir   = os.path.normpath(os.path.join(os.path.dirname(__file__),obsdir))
            scenpath = os.path.normpath(os.path.join(os.path.dirname(__file__),scenpath))
            gotmexe  = os.path.normpath(os.path.join(os.path.dirname(__file__),gotmexe))
            assert os.path.isdir(scenpath), 'Scenario path %s does not exist.' % scenpath

        mysql_transport = transport.MySQL(server='localhost',user='run',password='g0tm',database='optimize')
        #http_transport = transport.HTTP(server='www.bio.vu.nl:80',path='/thb/users/jbrugg/optimize/')
        job = optimizer.Job(jobid,
                            scenpath,
                            gotmexe=gotmexe,
        #                    transports=(mysql_transport,http_transport),
                            transports=(mysql_transport,),
                            copyexe=not hasattr(sys,'frozen'))

        parset = 3
        if parset==1:
            # Estimate k_min (background kinetic energy to parameterize internal waves)
            job.controller.addParameter('gotmturb.nml','turb_param','k_min',1e-7,1e-4,logscale=True)
            job.addObservation(os.path.join(obsdir,'pottemp.obs'),'temp',maxdepth=310)
            job.addObservation(os.path.join(obsdir,'salt.obs'),'salt',maxdepth=310)
            #job.addObservation(os.path.join(obsdir,'lctemp.obs'),'temp')
        elif parset==2:
            tf = gotmcontroller.SimpleTransform((('bio_DEB.nml','parameters','j_V_Am_H'),('bio_DEB.nml','parameters','y_D_V')),
                                                ('net_j_V_Am_H','eff_H'),
                                                lambda net_j,eff: (net_j/eff,1/eff),
                                                bounds={'net_j_V_Am_H':(0.,4.),'eff_H':(0.,1.)})
            job.controller.addParameterTransform(tf)

            job.controller.addParameter('bio_DEB.nml','parameters','j_V_Am_A',0,4.)
            job.controller.addParameter('bio_DEB.nml','parameters','K_L_ref', 0,20.)
            job.controller.addParameter('bio_DEB.nml','parameters','K_N',     0,2.)
            #job.controller.addParameter('bio_DEB.nml','parameters','j_V_Am_H',0,4.)
            job.controller.addParameter('bio_DEB.nml','parameters','K_D_ref', 0,1.)
            job.controller.addParameter('bio_DEB.nml','parameters','k',       0,.1)
            job.controller.addParameter('bio_DEB.nml','parameters','h',       0,.1)
            #job.controller.addParameter('bio_DEB.nml','parameters','y_D_V',   1,4.)
            job.controller.addParameter('bio_DEB.nml','moments_bivariate','ini_mean1',0,1.)
            job.controller.addParameter('bio_DEB.nml','moments_bivariate','ini_mean2',0,1.)
            job.controller.addParameter('bio_DEB.nml','moments_bivariate','ini_var1',0,.3)
            job.controller.addParameter('bio_DEB.nml','moments_bivariate','ini_var2',0,.3)
            job.controller.addParameter('bio_DEB.nml','parameters','T_arr',   0,15000)
            job.controller.addParameter('bio_DEB.nml','parameters','v_sinking_D_top', 0,10.)
            job.controller.addParameter('bio_DEB.nml','parameters','v_sinking_D_slope',0,.1)
            job.controller.addParameter('bio_DEB.nml','parameters','immigration_rate',1e-6,1e-2,logscale=True)
            job.controller.addDummyParameter('dummy',0.,1.)
            job.addObservation(os.path.join(obsdir,'din.obs'    ),'X_N',                      maxdepth=310,spinupyears=2)
            job.addObservation(os.path.join(obsdir,'chl.obs'    ),'mean_1*X_V',               maxdepth=310,spinupyears=2,relativefit=True,min_scale_factor=100,max_scale_factor=40000)
            job.addObservation(os.path.join(obsdir,'pon.obs'    ),'X_V*(1+mean_1+mean_2)+X_D',maxdepth=310,spinupyears=2)
            job.addObservation(os.path.join(obsdir,'ponflux.obs'),'pom_flux',                 maxdepth=310,spinupyears=2)
            job.addObservation(os.path.join(obsdir,'pp.obs'     ),'prod_aut',                 maxdepth=310,spinupyears=2,relativefit=True,min_scale_factor=0.01,max_scale_factor=100.)
        elif parset==3:
            #job.controller.addParameter('fabm.nml','phytosize','gam', -0.5,0.) # use growth rate insetad of size
            job.controller.addParameter('fabm.nml','phytosize','delt',-2.,2.) # now exponent relative to growth rate exponent
            job.controller.addParameter('fabm.nml','phytosize','j_NM',0.,0.1)
            job.controller.addParameter('fabm.nml','phytosize','K_L', 0.,20.)
            #job.controller.addParameter('fabm.nml','phytosize','j_NAm', 0.5,2.) # use growth rate insetad of size
            job.controller.addParameter('fabm.nml','phytosize','K_N', 0.,.5)
            job.controller.addParameter('fabm.nml','phytosize','g_m', 0.,3.)
            job.controller.addParameter('fabm.nml','phytosize','bet', 0.,2.) # now predator half sat for growth rate 1/ln growth rate 0
            job.controller.addParameter('fabm.nml','phytosize','y_PZ', 1.,2.)
            job.controller.addParameter('fabm.nml','phytosize','h2', 0.,.2)
            job.controller.addParameter('fabm.nml','phytosize','h_D', 0.,.2)
            #job.controller.addParameter('fabm.nml','phytosize','var_om_imm', 0.,0.)
            job.controller.addParameter('fabm.nml','phytosize','var_om_imm', 0.,15.)
            job.controller.addParameter('fabm.nml','phytosize','var_relax', 0.,.5)
            job.controller.addParameter('fabm.nml','phytosize','w_det', -10.,0.)
            job.controller.addParameter('fabm.nml','phytosize','T_A_P', 0.,20000.)
            job.controller.addParameter('fabm.nml','phytosize','T_A_Z', 0.,20000.)
            job.controller.addDummyParameter('dummy',0.,1.)
            job.addObservation(os.path.join(obsdir,'din.obs'    ),'phytosize_n',                        maxdepth=310,spinupyears=2)
            job.addObservation(os.path.join(obsdir,'chl.obs'    ),'phytosize_m_v',                        maxdepth=310,spinupyears=2,relativefit=True,min_scale_factor=100,max_scale_factor=40000)
            job.addObservation(os.path.join(obsdir,'pon.obs'    ),'phytosize_m_v+phytosize_d+phytosize_z',maxdepth=310,spinupyears=2)
            job.addObservation(os.path.join(obsdir,'ponflux.obs'),'phytosize_ponflux',                  maxdepth=310,spinupyears=2)
            job.addObservation(os.path.join(obsdir,'pp.obs'     ),'phytosize_pp',                       maxdepth=310,spinupyears=2,relativefit=True,min_scale_factor=0.01,max_scale_factor=100.)
        elif parset==4:
            job.controller.addParameter('fabm.nml','jbruggeman_si_fractionation','r_max',    0.5, 2.)
            job.controller.addParameter('fabm.nml','jbruggeman_si_fractionation','K_Si',     0.5, 4.)
            job.controller.addParameter('fabm.nml','jbruggeman_si_fractionation','K_L',      0., 20.)
            job.controller.addParameter('fabm.nml','jbruggeman_si_fractionation','w_top',   -5.,  0.)
            job.controller.addParameter('fabm.nml','jbruggeman_si_fractionation','w_slope', -1.,  0.)
            job.controller.addParameter('fabm.nml','jbruggeman_si_fractionation','specific_area', 3.,30.)
            job.controller.addDummyParameter('dummy',0.,1.)
            job.addObservation(os.path.join(obsdir,'si.obs'    ),  'jbruggeman_si_fractionation_si28+jbruggeman_si_fractionation_si30',spinupyears=1)
            job.addObservation(os.path.join(obsdir,'massflux.obs'),'jbruggeman_si_fractionation_si_flux',spinupyears=1,fixed_scale_factor=60./(.12*.88))
        #job.excludeObservationPeriod(datetime.datetime(1993,1,1),datetime.datetime(1994,1,1))
    return job

def main():
    parser = optparse.OptionParser()
    parser.add_option('-m', '--method',      type='choice', choices=('DE','fmin','galileo'), help='Optimization method: DE = Differential Evolution genetic algorithm, fmin = Nelder-Mead simplex, galileo = galileo genetic algorithm')
    parser.add_option('-t', '--transport',   type='choice', choices=('http','mysql'), help='Transport to use for server communication: http or mysql')
    parser.add_option('-i', '--interactive', action='store_true', help='Whether to allow for user interaction (input from stdin) when making decisions')
    parser.set_defaults(method='DE',transport=None,interactive=not hasattr(sys,'frozen'))
    (options, args) = parser.parse_args()
    if len(args)<1:
        print 'One argument must be provided: the (integer) job identifier.'
        sys.exit(2)
    jobid = int(args[0])
    job = getJob(jobid)

    if options.transport=='http':
        job.transports = (http_transport,)
    elif options.transport=='mysql':
        job.transports = (mysql_transport,)
        
    job.interactive = options.interactive

    repeat = (options.method!='fmin')

    while repeat:
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
            import DESolver_nopp as DESolver
            import numpy.random

            class Solver(DESolver.DESolver):
                def __init__(self,job,*args,**kwargs):
                    DESolver.DESolver.__init__(self,*args,**kwargs)
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
            solver = Solver(job, len(minpar), popsize, maxgen, minpar, maxpar, 'Rand1Exp_jorn', 0.5, 0.9, 0.01, True, False, initialpopulation=startpop)
            solver.Solve()

            #print 'Generation %i done. Current best fitness = %.6g.' % (itn,P.maxFitness)
            print 'Best parameter set:'
            vals = job.controller.untransformParameterValues(solver.bestSolution)
            for parinfo,val in zip(job.controller.parameters,vals):
                print '  %s = %.6g' % (parinfo['name'],val)

if __name__ == '__main__':
    main()
