import numpy

def normal(yobs,sdyobs,ymod):
    lnls = -0.5*numpy.log(2.*numpy.pi) - numpy.log(sdyobs) - 0.5*((yobs-ymod)/sdyobs)**2
    #lnls = -((yobs-ymod)/sdyobs)**2
    return lnls.sum()

def normal_nosd(yobs,ymod):
    var = ((yobs-ymod)**2).mean()   # biased MLE estimator!
    lnls = -0.5*numpy.log(2.*numpy.pi*var) - 0.5*(yobs-ymod)**2/var
    return lnls.sum()
