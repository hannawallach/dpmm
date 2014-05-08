from collections import defaultdict
from numpy import bincount, empty, log, log2, unique, zeros
from numpy.random import choice, uniform
from scipy.special import gammaln

from algorithm_3 import iteration as algorithm_3_iteration
from kale.math_utils import log_sample, log_sum_exp, vi


def iteration(V, D, N_DV, N_D, alpha, beta, z_D, inv_z_T, active_topics, inactive_topics, N_TV, N_T, D_T, num_inner_itns):
    """
    Performs a single iteration of Metropolis-Hastings (split-merge).
    """

    N_s_V = empty(V, dtype=int)
    N_t_V = empty(V, dtype=int)

    log_dist = empty(2)

    d, e = choice(D, 2, replace=False) # choose 2 documents

    if z_D[d] == z_D[e]:
        s = inactive_topics.pop()
        active_topics.add(s)
    else:
        s = z_D[d]

    inv_z_s = set([d])
    N_s_V[:] = N_DV[d, :]
    N_s = N_D[d]
    D_s = 1

    t = z_D[e]
    inv_z_t = set([e])
    N_t_V[:] = N_DV[e, :]
    N_t = N_D[e]
    D_t = 1

    if z_D[d] == z_D[e]:
        idx = inv_z_T[t] - set([d, e])
    else:
        idx = (inv_z_T[s] | inv_z_T[t]) - set([d, e])

    for f in idx:
        if uniform() < 0.5:
            inv_z_s.add(f)
            N_s_V += N_DV[f, :]
            N_s += N_D[f]
            D_s += 1
        else:
            inv_z_t.add(f)
            N_t_V += N_DV[f, :]
            N_t += N_D[f]
            D_t += 1

    acc = 0.0

    for inner_itn in xrange(num_inner_itns):
        for f in idx:

            # (fake) restricted Gibbs sampling scan

            if f in inv_z_s:
                inv_z_s.remove(f)
                N_s_V -= N_DV[f, :]
                N_s -= N_D[f]
                D_s -= 1
            else:
                inv_z_t.remove(f)
                N_t_V -= N_DV[f, :]
                N_t -= N_D[f]
                D_t -= 1

            log_dist[0] = log(D_s)
            log_dist[0] += gammaln(N_s + beta)
            log_dist[0] -= gammaln(N_D[f] + N_s + beta)
            tmp = N_s_V + beta / V
            log_dist[0] += gammaln(N_DV[f, :] + tmp).sum()
            log_dist[0] -= gammaln(tmp).sum()

            log_dist[1] = log(D_t)
            log_dist[1] += gammaln(N_t + beta)
            log_dist[1] -= gammaln(N_D[f] + N_t + beta)
            tmp = N_t_V + beta / V
            log_dist[1] += gammaln(N_DV[f, :] + tmp).sum()
            log_dist[1] -= gammaln(tmp).sum()

            log_dist -= log_sum_exp(log_dist)

            if inner_itn == num_inner_itns - 1 and z_D[d] != z_D[e]:
                u = 0 if z_D[f] == s else 1
            else:
                [u] = log_sample(log_dist)

            if u == 0:
                inv_z_s.add(f)
                N_s_V += N_DV[f, :]
                N_s += N_D[f]
                D_s += 1
            else:
                inv_z_t.add(f)
                N_t_V += N_DV[f, :]
                N_t += N_D[f]
                D_t += 1

            if inner_itn == num_inner_itns - 1:
                acc += log_dist[u]

    if z_D[d] == z_D[e]:

        acc *= -1.0

        acc += log(alpha)
        acc += gammaln(D_s) + gammaln(D_t) - gammaln(D_T[t])

        acc += gammaln(beta) + gammaln(N_T[t] + beta)
        acc -= gammaln(N_s + beta) + gammaln(N_t + beta)
        tmp = beta / V
        acc += gammaln(N_s_V + tmp).sum() + gammaln(N_t_V + tmp).sum()
        acc -= V * gammaln(tmp) + gammaln(N_TV[t, :] + tmp).sum()

        if log(uniform()) < min(0.0, acc):
            z_D[list(inv_z_s)] = s
            z_D[list(inv_z_t)] = t
            inv_z_T[s] = inv_z_s
            inv_z_T[t] = inv_z_t
            N_TV[s, :] = N_s_V
            N_TV[t, :] = N_t_V
            N_T[s] = N_s
            N_T[t] = N_t
            D_T[s] = D_s
            D_T[t] = D_t
        else:
            active_topics.remove(s)
            inactive_topics.add(s)

    else:

        for f in inv_z_T[s]:
            inv_z_t.add(f)
            N_t_V += N_DV[f, :]
            N_t += N_D[f]
            D_t += 1

        acc -= log(alpha)
        acc += gammaln(D_t) - gammaln(D_T[s]) - gammaln(D_T[t])

        acc += gammaln(N_T[s] + beta) + gammaln(N_T[t] + beta)
        acc -= gammaln(beta) + gammaln(N_t + beta)
        tmp = beta / V
        acc += V * gammaln(tmp) + gammaln(N_t_V + tmp).sum()
        acc -= (gammaln(N_TV[s, :] + tmp).sum() +
                gammaln(N_TV[t, :] + tmp).sum())

        if log(uniform()) < min(0.0, acc):
            active_topics.remove(s)
            inactive_topics.add(s)
            z_D[list(inv_z_t)] = t
            inv_z_T[s].clear()
            inv_z_T[t] = inv_z_t
            N_TV[s, :] = zeros(V, dtype=int)
            N_TV[t, :] = N_t_V
            N_T[s] = 0
            N_T[t] = N_t
            D_T[s] = 0
            D_T[t] = D_t


def inference(N_DV, alpha, beta, z_D, num_itns, true_z_D=None):
    """
    Conjugate split-merge.
    """

    D, V = N_DV.shape

    T = D # maximum number of topics

    N_D = N_DV.sum(1) # document lengths

    inv_z_T = defaultdict(set)
    for d in xrange(D):
        inv_z_T[z_D[d]].add(d) # inverse mapping from topics to documents

    active_topics = set(unique(z_D))
    inactive_topics = set(xrange(T)) - active_topics

    N_TV = zeros((T, V), dtype=int)
    N_T = zeros(T, dtype=int)

    for d in xrange(D):
        N_TV[z_D[d], :] += N_DV[d, :]
        N_T[z_D[d]] += N_D[d]

    D_T = bincount(z_D, minlength=T)

    for itn in xrange(num_itns):

        for _ in xrange(3):
            iteration(V, D, N_DV, N_D, alpha, beta, z_D, inv_z_T, active_topics, inactive_topics, N_TV, N_T, D_T, 6)

        algorithm_3_iteration(V, D, N_DV, N_D, alpha, beta, z_D, inv_z_T, active_topics, inactive_topics, N_TV, N_T, D_T)

        if true_z_D is not None:

            v = vi(true_z_D, z_D)

            print 'Itn. %d' % (itn + 1)
            print '%d topics' % len(active_topics)
            print 'VI: %f bits (%f bits max.)' % (v, log2(D))

            if v < 1e-6:
                break

    return z_D
