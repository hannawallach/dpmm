from numpy import argsort, bincount, ones, where, zeros
from numpy.random import poisson, seed
from numpy.random.mtrand import dirichlet

from kale.math_utils import sample


def generate_data(V, D, l, alpha, beta):
    """
    Generates a synthetic corpus of documents from a Dirichlet process
    mixture model with multinomial mixture components (topics). The
    mixture components are drawn from a symmetric Dirichlet prior.

    Arguments:

    V -- vocabulary size
    D -- number of documents
    l -- average document length
    alpha -- concentration parameter for the Dirichlet process
    beta -- concentration parameter for the symmetric Dirichlet prior
    """

    T = D # maximum number of topics

    phi_TV = zeros((T, V))
    z_D = zeros(D, dtype=int)
    N_DV = zeros((D, V), dtype=int)

    for d in xrange(D):

        # draw a topic assignment for this document

        dist = bincount(z_D).astype(float)
        dist[0] = alpha
        [t] = sample(dist)
        t = len(dist) if t == 0 else t
        z_D[d] = t

        # if it's a new topic, draw the parameters for that topic

        if t == len(dist):
            phi_TV[t - 1, :] = dirichlet(beta * ones(V) / V)

        # draw the tokens from the topic

        for v in sample(phi_TV[t - 1, :], num_samples=poisson(l)):
            N_DV[d, v] += 1

    z_D = z_D - 1

    return phi_TV, z_D, N_DV
