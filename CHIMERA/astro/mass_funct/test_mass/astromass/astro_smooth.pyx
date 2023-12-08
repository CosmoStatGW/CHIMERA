from libc.math cimport log, exp

cpdef double _logaddexp(double x, double y) nogil:
    cdef double max_val = max(x, y)
    return max_val + log(exp(x - max_val) + exp(y - max_val))

cpdef double logpdf_truncated_power_law(double x, double alpha, double mmin, double mmax):
    cdef double norm_const = (1 - alpha) / (pow(mmax, 1 - alpha) - pow(mmin, 1 - alpha))
    return log(norm_const) - alpha * log(x)

cpdef double logpdf_gaussian(double x, double mu, double sigma):
    return -0.5 * log(2 * 3.141592653589793 * sigma * sigma) - (x - mu) * (x - mu) / (2 * sigma * sigma)

cpdef double _logSmoothing(double m, double ml, double delta_m):
    if m <= ml:
        return float('-inf')
    elif m >= ml + delta_m:
        return 0.0
    else:
        return -_logaddexp(0.0, delta_m / (m - ml) + delta_m / (m - ml - delta_m))

cpdef double logpdfm1_my_single(double m1, double lambda_peak, double alpha, double delta_m, double ml, double mh, double mu_m, double sigma_m):
    cdef double P, G, result

    P = exp(logpdf_truncated_power_law(m1, alpha, ml, mh))
    G = exp(logpdf_gaussian(m1, mu_m, sigma_m))
    result = log((1 - lambda_peak) * P + lambda_peak * G) + _logSmoothing(m1, ml, delta_m)

    return result