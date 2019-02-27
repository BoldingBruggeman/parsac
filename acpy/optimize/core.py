from __future__ import print_function
import numpy

from . import desolver
try:
    from . import bfgs
except ImportError:
    bfgs = None

try:
    import scipy.optimize
    import scipy.stats
except ImportError:
    scipy = None

#desolver.pp = None

SIMPLEX = 1
DIFFERENTIALEVOLUTION = 2
BFGS = 3

class OptimizationProblem:
    """Base class for optimization problems."""
    def __init__(self):
        pass

    def evaluateFitness(self, parameters):
        raise NotImplementedError('Classes deriving from OptimizationProblem must implement evaluateFitness.')

    def evaluateAndUnpack(self, parameters):
        result = self.evaluateFitness(parameters)
        return result if not isinstance(result, tuple) else result[0]

    def start(self):
        pass

class TransformedProblem(OptimizationProblem):
    """Filter that transforms one or more parameters."""
    def __init__(self, problem, transforms=()):
        self.problem = problem
        if transforms is None:
            transforms = ()
        self.transform_names = transforms
        self.createTransformFunctions()

    def __getstate__(self):
        return {'problem': self.problem, 'transform_names': self.transform_names}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.createTransformFunctions()

    def createTransformFunctions(self):
        self.transforms = []
        self.inverse_transforms = []
        self.transformed = False
        for transform in self.transform_names:
            if transform and not isinstance(transform, (str, u''.__class__)):
                # old convention: True means log10 transform
                transform = 'log10'
            if isinstance(transform, (str, u''.__class__)):
                if transform == 'log10':
                    transform = lambda x: numpy.log10(x)
                    inverse_transform = lambda x: 10.**x
                    self.transformed = True
                elif transform == 'log':
                    transform = lambda x: numpy.log(x)
                    inverse_transform = lambda x: numpy.exp(x)
                    self.transformed = True
                elif transform == 'logit':
                    transform = lambda x: numpy.log(x/(1.-x))
                    inverse_transform = lambda x: numpy.exp(x)/(1.+numpy.exp(x))
                    self.transformed = True
                else:
                    assert False, 'Unknown transform %s selected' % transform
            else:
                transform = lambda x: x
                inverse_transform = lambda x: x
            self.transforms.append(transform)
            self.inverse_transforms.append(inverse_transform)

    def transform(self, parameters):
        if not self.transformed:
            return parameters
        assert len(parameters) == len(self.transforms)
        values = []
        for value, transform in zip(parameters, self.transforms):
            if value is not None:
                value = transform(value)
            values.append(value)
        return numpy.array(values)

    def untransform(self, parameters):
        if not self.transformed:
            return parameters
        assert len(parameters) == len(self.transforms)
        values = []
        for value, transform in zip(parameters, self.inverse_transforms):
            if value is not None:
                value = transform(value)
            values.append(value)
        return numpy.array(values)

    def evaluateFitness(self, parameters):
        return self.problem.evaluateFitness(self.untransform(parameters))

    def start(self):
        self.problem.start()

class ReducedProblem(OptimizationProblem):
    """Filter that sets one parameter to a constant value."""
    def __init__(self, problem, ipar, value):
        self.problem = problem
        self.ipar = ipar
        self.value = value

    def reduce(self, parameters):
        p = list(parameters)
        del p[self.ipar]
        return p

    def expand(self, parameters):
        p = list(parameters)
        p.insert(self.ipar, self.value)
        return p

    def evaluateFitness(self,parameters):
        return self.problem.evaluateFitness(self.expand(parameters))

    def start(self):
        self.problem.start()

class ReportingProblem(OptimizationProblem):
    """Filter that sends all results to a reporting function."""
    def __init__(self, problem, reportfunction):
        self.problem = problem
        self.reportfunction = reportfunction

    def evaluateFitness(self, parameters):
        fitness = self.problem.evaluateFitness(parameters)
        if self.reportfunction is not None:
            self.reportfunction(parameters, fitness)
        return fitness

    def start(self):
        self.problem.start()

class Optimizer:
    def __init__(self, problem, reportfunction=None):
        assert isinstance(problem, OptimizationProblem)
        self.problem = problem
        self.reportfunction = reportfunction

    def run(self, method=SIMPLEX, par_ini=None, par_min=None, par_max=None, transform=None,
            maxiter=1000, maxfun=1000, verbose=True,
            modules=(), reltol=0.01, abstol=1e-8, ftol=numpy.inf, popsize=None, maxgen=500, F=0.5, CR=0.9, initialpopulation=None, parallelize=True, ppservers=(), secret=None, ncpus=None, max_runtime=None):
        if isinstance(method, (list, tuple)):
            for curmethod in method:
                par_ini = self.run(curmethod, par_ini, par_min, par_max, transform=transform, modules=modules, maxiter=maxiter, maxfun=maxfun, verbose=verbose, reltol=reltol, abstol=abstol, ftol=ftol, popsize=popsize, maxgen=maxgen, CR=CR, F=F, initialpopulation=initialpopulation, parallelize=parallelize, ppservers=ppservers, secret=secret, ncpus=ncpus, max_runtime=max_runtime)
            return par_ini

        problem = self.problem
        if method != DIFFERENTIALEVOLUTION:
            problem.start()
            problem = ReportingProblem(problem, self.reportfunction)

        problem = TransformedProblem(problem, transforms=transform)
        if par_min is not None:
            par_min = problem.transform(par_min)
        if par_max is not None:
            par_max = problem.transform(par_max)
        if par_ini is not None:
            par_ini = problem.transform(par_ini)

        if method == SIMPLEX:
            assert par_ini is not None, 'Simplex method requires an initial estimate.'
            p_final = scipy.optimize.fmin(lambda p: -problem.evaluateAndUnpack(p), par_ini, maxiter=maxiter, maxfun=maxfun, disp=verbose)
        elif method == BFGS:
            assert par_ini is not None, 'BFGS method requires an initial estimate.'
            p_final = bfgs.fmin_bfgs(lambda p: -problem.evaluateAndUnpack(p), par_ini, disp=False)
            #p_final = scipy.optimize.fmin_bfgs(lambda p: -problem.evaluateAndUnpack(p),par_ini,disp=False)
        else:
            assert par_min is not None, 'Differential Evolution method requires lower parameter bounds.'
            assert par_max is not None, 'Differential Evolution method requires upper parameter bounds.'
            if not parallelize:
                ncpus = 1

            nfreepar = (numpy.asarray(par_min) != numpy.asarray(par_max)).sum()
            if popsize is None:
                popsize = nfreepar*10

            class Reporter:
                def __init__(self, reportfunction, transform):
                    self.transform = transform
                    self.reportfunction = reportfunction
                def reportResult(self, pars, result):
                    if self.reportfunction is not None:
                        self.reportfunction(self.transform(pars), result)

            solver = desolver.DESolver(problem, popsize, maxgen,
                                       par_min, par_max, F=F, CR=CR, initialpopulation=initialpopulation,
                                       ncpus=ncpus, ppservers=ppservers,
                                       reporter=Reporter(self.reportfunction, problem.untransform), verbose=verbose,
                                       reltol=reltol, abstol=abstol, ftol=ftol, socket_timeout=max_runtime, secret=secret)
            p_final = solver.Solve()

        return problem.untransform(p_final)

    def calculateP(self, best, ipar, altvalue):
        self.problem.start()
        bestfitness = self.problem.evaluateAndUnpack(best)
        problem = ReportingProblem(self.problem, self.reportfunction)
        problem = ReducedProblem(problem, ipar, altvalue)
        est = scipy.optimize.fmin(lambda p: -problem.evaluateAndUnpack(p), problem.reduce(best), disp=False)
        altfitness = problem.evaluateAndUnpack(est)

        chi2_crit = 2.*(bestfitness-altfitness)
        chi2_p = 1. - scipy.stats.chi2.cdf(chi2_crit, 1)
        return chi2_p

    def profile(self, start, maxstep=None, targetdelta=None, profile=None, nzoom=4, transform=None):
        self.problem.start()
        if profile is None:
            profile = numpy.ones((len(start),), dtype=bool)
        if targetdelta is None:
            targetdelta = 0.5*scipy.stats.chi2.ppf(0.95, 1)
        assert targetdelta > 0, 'Target fitness difference (targetdelta) must be positive.'

        problem = self.problem
        problem = ReportingProblem(problem, self.reportfunction)
        problem = TransformedProblem(problem, transform=transform)

        start = problem.transform(start)
        startfitness = problem.evaluateAndUnpack(start)
        target = startfitness-targetdelta

        def run(ipar, x, step, target):
            while 1:
                x[ipar] += step
                def ev(p):
                    p = list(p)
                    p.insert(ipar, x[ipar])
                    fitness = problem.evaluateAndUnpack(p)
                    return -fitness
                x_ini = list(x)
                del x_ini[ipar]
                if not numpy.isfinite(ev(x_ini)):
                    return x
                est = scipy.optimize.fmin(ev, x_ini, disp=False)
                if -ev(est) < target:
                    return x

        left, right = [], []
        for ipar in range(len(start)):
            if not profile[ipar]:
                left.append(None)
                right.append(None)
                continue

            print('Profiling parameter %i...' % ipar)

            if maxstep is None:
                # Assume parabolic shape of likelihood, and estimate 2nd derivative by finite differences.
                pert = list(start)
                pertstep = max(abs(start[ipar]*1e-6), 1e-6)
                pert[ipar] = start[ipar]+pertstep
                pertfitness_h = problem.evaluateAndUnpack(pert)
                pert[ipar] = start[ipar]-pertstep
                pertfitness_l = problem.evaluateAndUnpack(pert)
                ddfddpar = (pertfitness_h+pertfitness_l-2*startfitness)/pertstep/pertstep
                curmaxstep = numpy.sqrt(targetdelta/abs(ddfddpar))/5.
                print('   auto-selected initial step size of %s' % curmaxstep)
            else:
                if maxstep[ipar] == 0:
                    print('Skipping profiling of parameter %i because min and max are equal (%s).' % (ipar, x_min_ori[ipar]))
                    left.append(start[ipar])
                    right.append(start[ipar])
                    continue
                curmaxstep = maxstep[ipar]

            x = list(start)
            for izoom in range(nzoom):
                curstep = -curmaxstep*0.1**izoom
                x = run(ipar, x, curstep, target)
                x[ipar] -= curstep
            left.append(x[ipar]+curstep)

            x = list(start)
            for izoom in range(nzoom):
                curstep = curmaxstep*0.1**izoom
                x = run(ipar, x, curstep, target)
                x[ipar] -= curstep
            right.append(x[ipar]+curstep)

        return problem.untransform(left), problem.untransform(right)
