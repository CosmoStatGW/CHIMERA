cimport cython
from libc.math cimport exp, pow, sqrt, log1p, log
import numpy as np
cimport numpy as np

from scipy.integrate import cumtrapz

cdef inline double logaddexp(double x, double y) nogil:
    cdef double tmp = x - y
    if tmp > 0:
        return x + log1p(exp(-tmp))
    elif tmp <= 0:
        return y + log1p(exp(tmp))
    else:
        return x + y

cdef inline double logdiffexp(double x, double y) nogil:
    return x + log1p(-exp(y-x))
    # cdef double tmp = x - y
    # if tmp > 0:
    #     return x + log1p(-exp(-tmp))
    # elif tmp <= 0:
    #     return y + log1p(-exp(tmp))
    # else:
    #     return x + y



#
#
#
###############################################################
#################  TPL - Truncated PowerLaw  ##################
###############################################################
# NOT USED _ Marginal distribution p(m1), not normalised
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _logpdfm1_TPL(double[::1] m1, 
                    double ml, double mh, double alpha):
    
    cdef Py_ssize_t size     = m1.shape[0]
    res                      = np.zeros(size)
    cdef double[:] res_view  = res  # memoryview

    for i in range(size):
        if (ml < m1[i] < mh):
            res_view[i] = -alpha*log(m1[i])
        else:
            res_view[i] = np.NINF

    return res

# NOT USED _ Conditional distribution p(m2 | m1)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _logpdfm2_TPL(double[::1] m2, 
                    double ml, double beta):

    cdef Py_ssize_t size     = m2.shape[0]
    res                      = np.zeros(size)
    cdef double[:] res_view  = res  # memoryview

    for i in range(size):
        if ml < m2[i]:
            res_view[i] = beta*log(m2[i]) 
        else:
            res_view[i] = np.NINF

    return res

# Inverse log integral of `p(m1,m2)dm2` (log C(m1) in the LVC notation)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _logC_TPL(double[::1] m,                # Slow!!!
                double ml, double beta):
    
    cdef Py_ssize_t size                  = m.shape[0]
    cdef np.ndarray[double, ndim = 1] res = np.zeros(size)
    cdef double[:] res_view               = res  # memoryview
    cdef double c, ld, lmb, lmib                

    lmb = (1+beta)*log(ml)

    if beta > -1:
        c  = log1p(beta)
        for i in range(size):
            lmib        = (1+beta)*log(m[i])
            res_view[i] = c - logdiffexp(lmib, lmb)  
    elif beta < -1:
        c = log(-1-beta)
        for i in range(size):
            lmib        = (1+beta)*log(m[i])
            res_view[i] = c - logdiffexp(lmb, lmib)

    return res


# log integral of  `p(m1,m2)dm1dm2` (total normalization of the mf)
cpdef double _logN_TPL(double ml, double mh, double alpha):
    
    if (alpha < 1) & (alpha!=0):
        return -log1p(-alpha) + logdiffexp( (1-alpha)*log(mh), (1-alpha)*log(ml) )

    elif alpha > 1:
        return -log(alpha-1) + logdiffexp( (1-alpha)*log(ml), (1-alpha)*log(mh) )


########################################
# p( m1, m2 | TPL ), normalized to one
########################################
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef logpdf_TPL(double[::1] m1, double[::1] m2, 
                 double ml, double mh, double alpha, double beta):

    cdef int i
    cdef Py_ssize_t size                       = m1.shape[0]
    cdef np.ndarray[double, ndim = 1] res      = np.zeros(size)
    # cdef np.ndarray[double, ndim = 1] logpdfm1 = _logpdfm1_TPL(m1, ml, mh, alpha)
    # cdef np.ndarray[double, ndim = 1] logpdfm2 = _logpdfm2_TPL(m2, ml, beta)
    cdef np.ndarray[double, ndim = 1] logC     = _logC_TPL(m1, ml, beta)
    cdef double logN                           = _logN_TPL(ml, mh, alpha)
    cdef double[:] res_view                    = res 

    # Compute logpdf if (mh>m1>m2>ml, m1!=nan, m2!=nan) else assign np.NINF
    for i in range(size): 
        if mh>m1[i]>m2[i]>ml:
            res_view[i] = (-alpha*log(m1[i]) + 
                           beta*log(m2[i]) +
                           logC[i] - logN)
        else: 
            res_view[i] = np.NINF

    return res
    
#
#
#
###############################################################
###################  BPL - Broken PowerLaw  ###################
###############################################################

cdef double _mbreak(double ml, double mh, double b):
    return ml + b*(mh-ml)

cdef double _logSigmoidLike(double mi, double ml, double dm):
    return -logaddexp(0, dm/(mi-ml) + dm/(mi-ml-dm))

# Marginal distribution p(m1), not normalised
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _logpdfm1_BPL(double[::1] m1, 
              double ml, double mh, double alphal, double alphah, double dm, double b):
    
    cdef double mbr          = _mbreak(ml, mh, b)
    cdef Py_ssize_t size     = m1.shape[0]
    cdef double logS         = 0
    cdef double alpha        = 0
    cdef double corr         = 0
    res                      = np.zeros(size)
    cdef double[:] res_view  = res  # memoryview

    for i in range(size):
        if (ml < m1[i] < mh):
            logS        = _logSigmoidLike(m1[i], ml, dm) if (0<m1[i]-ml<dm) else 0
            alpha       = alphal if m1[i]<mbr else alphah
            corr        = 0 if m1[i]<mbr else log(mbr)*(-alphal+alphah)   #np.log(mBreak)*(-alpha1+alpha2) if m>mbreak ????
            res_view[i] = -alpha*log(m1[i]) + logS + corr 
        else:
            res_view[i] = np.NINF

    return res

    
# Conditional distribution p(m2 | m1)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _logpdfm2_BPL(double[::1] m2, 
              double ml, double beta, double dm):

    cdef double logS         = 0
    cdef Py_ssize_t size     = m2.shape[0]
    res                      = np.zeros(size)
    cdef double[:] res_view  = res  # memoryview

    for i in range(size):
        if ml < m2[i]:
            logS        = _logSigmoidLike(m2[i], ml, dm) if (0<m2[i]-ml<dm) else 0
            res_view[i] = beta*log(m2[i]) + logS
        else:
            res_view[i] = np.NINF

    return res
        

# OCIO Python Function
# Inverse log integral of `p(m1,m2)dm2` (log C(m1) in the LVC notation)
@cython.boundscheck(False)
@cython.wraparound(False)
def _logC_BPL(double[::1] m, 
          double ml, double beta, double dm):
    
    cdef Py_ssize_t size                  = m.shape[0]
    cdef np.ndarray[double, ndim = 1] xx  = np.zeros(400)
    cdef np.ndarray[double, ndim = 1] p2  = np.zeros(400)
    cdef np.ndarray[double, ndim = 1] cdf = np.zeros(400)
    cdef np.ndarray[double, ndim = 1] res = np.zeros(size)

    xx = np.concatenate([np.linspace(ml, ml + 1.1*dm, 200),
                         np.linspace(ml + 1.1*dm + 1e-01, np.nanmax(m), 200)])
    p2  = np.exp(_logpdfm2_BPL(xx, ml, beta, dm))
    cdf = cumtrapz(p2, xx)

    return -np.log( np.interp(m, xx[1:], cdf) )


# OCIO Python Function
# log integral of  `p(m1,m2)dm1dm2` (total normalization of the mf)
@cython.boundscheck(False)
@cython.wraparound(False)
def _logN_BPL(double ml, double mh, double alphal, double alphah, double dm, double b):
    
    cdef double mbr                       = _mbreak(ml, mh, b)
    cdef np.ndarray[double, ndim = 1] xx  = np.zeros(400)
    cdef np.ndarray[double, ndim = 1] p1  = np.zeros(400)

    xx = np.concatenate([np.linspace(1., ml+ 1.1*dm, 200),
                         np.linspace(ml + 1.1*dm + 1e-01, 0.9*mbr, 100),
                         np.linspace(0.9*mbr + 1e-01, 1.1*mbr, 50),
                         np.linspace(1.1*mbr + 1e-01, 1.1*mh, 50)])

    p1 = np.exp(_logpdfm1_BPL(xx, ml, mh, alphal, alphah, dm, b))

    return np.log(np.trapz(p1,xx))


########################################
# p( m1, m2 | BPL ), normalized to one
########################################
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef logpdf_BPL(double[::1] m1, double[::1] m2, 
                 double ml, double mh, double alphal, double alphah, double beta, double dm, double b):
    
    assert tuple(m1.shape) == tuple(m2.shape)

    cdef Py_ssize_t size                       = m1.shape[0]
    cdef np.ndarray[double, ndim = 1] logpdfm1 = _logpdfm1_BPL(m1, ml, mh, alphal, alphah, dm, b)
    cdef np.ndarray[double, ndim = 1] logpdfm2 = _logpdfm2_BPL(m2, ml, beta, dm)
    cdef np.ndarray[double, ndim = 1] logC     = _logC_BPL(m1, ml, beta, dm)
    cdef double logN                           = _logN_BPL(ml, mh, alphal, alphah, dm, b)

    res                     = np.zeros(size)
    cdef double[:] res_view = res 

    # Compute logpdf if (mh>m1>m2>ml, m1!=nan, m2!=nan) else assign np.NINF
    for i in range(size): 
        if mh>m1[i]>m2[i]>ml:
            res_view[i] = logpdfm1[i] + logpdfm2[i] + logC[i] - logN
        else: 
            res_view[i] = np.NINF

    return res


#
#
#
###############################################################
##################  PLP - PowerLaw + Peak  ####################
###############################################################
# Marginal distribution p(m1), not normalised
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _logpdfm1_PLP(double[::1] m1, 
                    double ml, double mh, double alpha, double dm, double lambdaPeak, double mu, double sigma):
    
    cdef double trunc_comp, gauss_comp, logS
    cdef double m_max        = max(mh, mu+10*sigma)
    cdef Py_ssize_t size     = m1.shape[0]
    cdef double logN_TPL     = _logN_TPL(ml,mh,alpha)
    res                      = np.zeros(size, order='C')
    cdef double[:] res_view  = res  # memoryview

    for i in range(size):
        if (ml < m1[i] < m_max):
            trunc_comp  = exp(-alpha*log(m1[i])-logN_TPL)
            gauss_comp  = exp(-pow(m1[i]-mu, 2)/(2*pow(sigma,2))) / (sqrt(6.2831853)*sigma)
            logS        = _logSigmoidLike(m1[i], ml, dm) if (0<m1[i]-ml<dm) else 0

            res_view[i] = log((1-lambdaPeak)*trunc_comp + lambdaPeak*gauss_comp) + logS 
        else:
            res_view[i] = np.NINF

    return res


# Conditional distribution p(m2 | m1)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _logpdfm2_PLP(double[::1] m2, 
                    double ml, double beta, double dm):

    cdef double logS         = 0
    cdef Py_ssize_t size     = m2.shape[0]
    res                      = np.zeros(size, order='C')
    cdef double[:] res_view  = res  # memoryview

    for i in range(size):
        if ml < m2[i]:
            logS        = _logSigmoidLike(m2[i], ml, dm) if (0<m2[i]-ml<dm) else 0
            res_view[i] = beta*log(m2[i]) + logS
        else:
            res_view[i] = np.NINF

    return res


# OCIO Python Function
# Inverse log integral of `p(m1,m2)dm2` (log C(m1) in the LVC notation)
@cython.boundscheck(False)
@cython.wraparound(False)
def _logC_PLP(double[::1] m, 
          double ml, double beta, double dm):
    
    cdef Py_ssize_t size                  = m.shape[0]
    cdef np.ndarray[double, ndim = 1] xx  = np.zeros(400, order='C')
    cdef np.ndarray[double, ndim = 1] p2  = np.zeros(400, order='C')
    cdef np.ndarray[double, ndim = 1] cdf = np.zeros(400, order='C')
    cdef np.ndarray[double, ndim = 1] res = np.zeros(size, order='C')

    xx = np.concatenate([np.linspace(ml, ml + 1.1*dm, 200),
                         np.linspace(ml + 1.1*dm + 1e-01, np.nanmax(m), 200)])
    p2  = np.exp(_logpdfm2_BPL(xx, ml, beta, dm))
    cdf = cumtrapz(p2, xx)

    return -np.log( np.interp(m, xx[1:], cdf) )

# OCIO Python Function
# log integral of  `p(m1,m2)dm1dm2` (total normalization of the mf)
@cython.boundscheck(False)
@cython.wraparound(False)
def _logN_PLP(double ml, double mh, double alpha, double dm, double lambdaPeak, double mu, double sigma):
    
    cdef double m_max        = max(mh, mu+10*sigma)
    xx  = np.zeros(500, order='C')
    p1  = np.zeros(500, order='C')
    cdef double[:] xx_view  = xx  # memoryview
    cdef double[:] p1_view  = p1  # memoryview

    if lambdaPeak!=0:
        xx = np.concatenate([np.linspace(1., ml + 1.1*dm, 200),                  # lower edge
                            np.linspace(ml + 1.1*dm + 1e-01, mu-5*sigma, 100),  # before gaussian peak
                            np.linspace(mu-5*sigma + 1e-01, mu+5*sigma, 100),   # around gaussian peak
                            np.linspace(mu+5*sigma + 1e-01, 1.5*m_max, 100)])   # after gaussian peak
    else:
        xx = np.linspace(ml, mh, 500) 

    p1 = np.exp(_logpdfm1_PLP(xx, ml, mh, alpha, dm, lambdaPeak, mu, sigma))

    return np.log(np.trapz(p1,xx))



########################################
# p( m1, m2 | BPL ), normalized to one
########################################
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef logpdf_PLP(double[::1] m1, double[::1] m2, 
                 double ml, double mh, double alpha, double beta, double dm, double lambdaPeak, double mu,  double sigma):
    
    assert tuple(m1.shape) == tuple(m2.shape)

    cdef Py_ssize_t size                       = m1.shape[0]
    cdef np.ndarray[double, ndim = 1] logpdfm1 = _logpdfm1_PLP(m1, ml, mh, alpha, dm, lambdaPeak, mu, sigma)
    cdef np.ndarray[double, ndim = 1] logpdfm2 = _logpdfm2_PLP(m2, ml, beta, dm)
    cdef np.ndarray[double, ndim = 1] logC     = _logC_PLP(m1, ml, beta, dm)
    cdef double logN                           = _logN_PLP(ml, mh, alpha, dm, lambdaPeak, mu, sigma)
    cdef double m_max                          = max(mh, mu+10*sigma)

    res                     = np.zeros(size, order='C')
    cdef double[:] res_view = res 

    # Compute logpdf if (mh>m1>m2>ml, m1!=nan, m2!=nan) else assign np.NINF
    for i in range(size): 
        if m_max>m1[i]>m2[i]>ml:
            
            res_view[i] = logpdfm1[i] + logpdfm2[i] + logC[i] - logN
        else: 
            res_view[i] = np.NINF

    return res
#
#
#
def get_base_PLP():
    return {'ml':5., 'mh':112., 'alpha': 3.78 , 'beta':0.81, 'deltam':4.8, 'lambdaPeak':0.03, 'muMass': 32., 'sigmaMass':3.88 }














#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

# def compute(int[:, ::1] array_1):
#     # get the maximum dimensions of the array
#     cdef Py_ssize_t x_max = array_1.shape[0]
#     cdef Py_ssize_t y_max = array_1.shape[1]
    
#     #create a memoryview
#     cdef int[:, :] view2d = array_1

#     # access the memoryview by way of our constrained indexes
#     for x in range(x_max):
#         for y in range(y_max):
#             view2d[x,y] = something()



# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef cumtrapz(double[::1] Y, double[::1] X):

#     cdef int i
#     cdef Py_ssize_t N        = Y.shape[0]-1
#     res                      = np.zeros(N)
#     cdef double[:] res_view  = res 

#     for i in range(1,N):
#         res_view[i] = res_view[i-1] + 0.5*(X[i] - X[i-1])*(Y[i] + Y[i-1])
    
#     return res


