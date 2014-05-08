from numpy import bincount, log, log2, ones, seterr, unique, zeros
from numpy.random.mtrand import dirichlet

from kale.math_utils import log_sample, vi


def iteration(V, D, N_DV, N_D, alpha, beta, M, phi_TV, z_D, inv_z_T, active_topics, inactive_topics, N_TV, N_T, D_T):
    """
    Performs a single iteration of Radford Neal's Algorithm 8.
    """

    for t in active_topics:
        phi_TV[t, :] = dirichlet(N_TV[t, :] + beta / V)

    for d in xrange(D):

        old_t = z_D[d]

        if inv_z_T is not None:
            inv_z_T[old_t].remove(d)

        N_TV[old_t, :] -= N_DV[d, :]
        N_T[old_t] -= N_D[d]
        D_T[old_t] -= 1

        seterr(divide='ignore')
        log_dist = log(D_T)
        seterr(divide='warn')

        idx = -1 * ones(M, dtype=int)
        idx[0] = old_t if D_T[old_t] == 0 else inactive_topics.pop()
        for m in xrange(1, M):
            idx[m] = inactive_topics.pop()
        active_topics |= set(idx)
        log_dist[idx] = log(alpha) - log(M)

        if idx[0] == old_t:
            phi_TV[idx[1:], :] = dirichlet(beta * ones(V) / V, M - 1)
        else:
            phi_TV[idx, :] = dirichlet(beta * ones(V) / V, M)

        for t in active_topics:
            log_dist[t] += (N_DV[d, :] * log(phi_TV[t, :])).sum()

        [t] = log_sample(log_dist)

        z_D[d] = t

        if inv_z_T is not None:
            inv_z_T[t].add(d)

        N_TV[t, :] += N_DV[d, :]
        N_T[t] += N_D[d]
        D_T[t] += 1

        idx = set(idx)
        idx.discard(t)
        active_topics -= idx
        inactive_topics |= idx


def inference(N_DV, alpha, beta, z_D, num_itns, true_z_D=None):
    """
    Algorithm 8.
    """

    M = 10 # number of auxiliary samples

    D, V = N_DV.shape

    T = D + M - 1 # maximum number of topics

    N_D = N_DV.sum(1) # document lengths

    phi_TV = zeros((T, V)) # topic parameters

    active_topics = set(unique(z_D))
    inactive_topics = set(xrange(T)) - active_topics

    N_TV = zeros((T, V), dtype=int)
    N_T = zeros(T, dtype=int)

    for d in xrange(D):
        N_TV[z_D[d], :] += N_DV[d, :]
        N_T[z_D[d]] += N_D[d]

    D_T = bincount(z_D, minlength=T)

    for itn in xrange(num_itns):

        iteration(V, D, N_DV, N_D, alpha, beta, M, phi_TV, z_D, None, active_topics, inactive_topics, N_TV, N_T, D_T)

        if true_z_D is not None:

            v = vi(true_z_D, z_D)

            print 'Itn. %d' % (itn + 1)
            print '%d topics' % len(active_topics)
            print 'VI: %f bits (%f bits max.)' % (v, log2(D))

            if v < 1e-6:
                break

    return phi_TV, z_D
