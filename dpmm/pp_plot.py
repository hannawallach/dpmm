from numpy import arange, array, empty_like, searchsorted, sort
from pylab import plot, show


def cdf(data):
    """
    Returns the empirical CDF (a function) for the specified data.

    Arguments:

    data -- data from which to compute the CDF
    """

    tmp = empty_like(data)
    tmp[:] = data
    tmp.sort()

    def f(x):
        return searchsorted(tmp, x, 'right') / float(len(tmp))

    return f


def pp_plot(a, b):
    """
    Generates a P-P plot.
    """

    x = sort(a)

    if len(x) > 10000:
        step = len(x) / 5000
        x = x[::step]

    plot(cdf(a)(x), cdf(b)(x), alpha=0.5)
    plot([0, 1], [0, 1], ':', c='k', lw=2, alpha=0.5)

    show()


def test(num_samples=100000):

    from numpy.random import normal

    a = normal(20.0, 5.0, num_samples)
    b = normal(20.0, 5.0, num_samples)

    pp_plot(a, b)


if __name__ == '__main__':
    test()
