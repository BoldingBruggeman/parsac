import numpy
import desolver
desolver.pp = None

def testbed1_3(x):
    xinrange = numpy.logical_and(x>=-5.12,x<=5.12)
    if xinrange.all(): return 30. + numpy.floor(x).sum()
    include = numpy.logical_and(numpy.logical_not(xinrange),x<0.)
    return numpy.prod(30.*numpy.sign(-x[include]-5.12))

testbed1 = [(lambda x:(x*x).sum(),                   [-5.12]*3, [5.12]*3, 1e-6,3,10,0.9,0.1),   # NP changed from 5 to 10!
            (lambda x:(x[0]**2-x[1])**2+(1.-x[0])**2,[-2.048]*2,[2.048]*2,1e-6,2,10,0.9,0.9),
            (testbed1_3,                             [-5.12]*5, [5.12]*5, 1e-6,5,10,0.9,0.0)]

class Job:
    def __init__(self,costfunction):
        self.costfunction = costfunction
        self.nfe = 0
    def evaluateFitness(self,x):
        self.nfe += 1
        cost = self.costfunction(x)
        return -cost

nrepeat = 20
for iproblem,(cost,minbounds,maxbounds,VTR,npar,NP,F,CR) in enumerate(testbed1):
    job = Job(cost)
    solver = desolver.DESolver(job,NP,10000,minbounds,maxbounds,
                               F=F,CR=CR,strictbounds=False,verbose=False,
                               reltol=0.,abstol=0.,minfitnesstoreach=-VTR)
    print 'Problem %i' % iproblem
    nfes = []
    for ireplica in range(nrepeat):
        job.nfe = 0
        sol = solver.Solve()
        nfe = job.nfe
        print '   solution %i: %s, FV = %s, NFE=%i' % (ireplica,sol,-job.evaluateFitness(sol),nfe)
        nfes.append(job.nfe)
    print '   mean NFE = %i' % (float(sum(nfes))/len(nfes))
