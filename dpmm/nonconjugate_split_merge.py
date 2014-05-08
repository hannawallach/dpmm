from collections import defaultdict
from numpy import bincount, empty, log, log2, unique, zeros
from numpy.random import choice, uniform
from numpy.random.mtrand import dirichlet
from scipy.special import gammaln

from algorithm_8 import iteration as algorithm_8_iteration
from kale.math_utils import log_sample, log_sum_exp, vi


def iteration(V, D, N_DV, N_D, alpha, beta, phi_TV, z_D, inv_z_T, active_topics, inactive_topics, N_TV, N_T, D_T, num_inner_itns):
    """
    Performs a single iteration of Metropolis-Hastings (split-merge).
    """

    phi_s_V = empty(V)
    phi_t_V = empty(V)
    phi_merge_t_V = empty(V)

    N_s_V = empty(V, dtype=int)
    N_t_V = empty(V, dtype=int)
    N_merge_t_V = empty(V, dtype=int)

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

    inv_z_merge_t = set([d, e])
    N_merge_t_V[:] = N_DV[d, :] + N_DV[e, :]
    N_merge_t = N_D[d] + N_D[e]
    D_merge_t = 2

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

        inv_z_merge_t.add(f)
        N_merge_t_V += N_DV[f, :]
        N_merge_t += N_D[f]
        D_merge_t += 1

    if z_D[d] == z_D[e]:
        phi_merge_t_V[:] = phi_TV[t, :]
    else:
        phi_merge_t_V = dirichlet(N_merge_t_V + beta / V)

    acc = 0.0

    for inner_itn in xrange(num_inner_itns):

        # sample new parameters for topics s and t ... but if it's the
        # last iteration and we're doing a merge, then just set the
        # parameters back to phi_TV[s, :] and phi_TV[t, :]

        if inner_itn == num_inner_itns - 1 and z_D[d] != z_D[e]:
            phi_s_V[:] = phi_TV[s, :]
            phi_t_V[:] = phi_TV[t, :]
        else:
            phi_s_V = dirichlet(N_s_V + beta / V)
            phi_t_V = dirichlet(N_t_V + beta / V)

        if inner_itn == num_inner_itns - 1:

            acc += gammaln(N_s + beta)
            acc -= gammaln(N_s_V + beta / V).sum()
            acc += ((N_s_V + beta / V - 1) * log(phi_s_V)).sum()

            acc += gammaln(N_t + beta)
            acc -= gammaln(N_t_V + beta / V).sum()
            acc += ((N_t_V + beta / V - 1) * log(phi_t_V)).sum()

            acc -= gammaln(N_merge_t + beta)
            acc += gammaln(N_merge_t_V + beta / V).sum()
            acc -= ((N_merge_t_V + beta / V - 1) *
                    log(phi_merge_t_V)).sum()

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
            log_dist[0] += (N_DV[f, :] * log(phi_s_V)).sum()

            log_dist[1] = log(D_t)
            log_dist[1] += (N_DV[f, :] * log(phi_t_V)).sum()

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
        tmp = beta / V
        acc += gammaln(beta) - V * gammaln(tmp)
        acc += (tmp - 1) * (log(phi_s_V).sum() + log(phi_t_V).sum())
        acc -= (tmp - 1) * log(phi_TV[t, :]).sum()

        acc += (N_s_V * log(phi_s_V)).sum() + (N_t_V * log(phi_t_V)).sum()
        acc -= (N_TV[t, :] * log(phi_TV[t, :])).sum()

        if log(uniform()) < min(0.0, acc):
            phi_TV[s, :] = phi_s_V
            phi_TV[t, :] = phi_t_V
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

        acc -= log(alpha)
        acc += gammaln(D_merge_t) - gammaln(D_T[s]) - gammaln(D_T[t])
        tmp = beta / V
        acc += V * gammaln(tmp) - gammaln(beta)
        acc += (tmp - 1) * log(phi_merge_t_V).sum()
        acc -= (tmp - 1) * (log(phi_TV[s, :]).sum() + log(phi_TV[t, :]).sum())

        acc += (N_merge_t_V * log(phi_merge_t_V)).sum()
        acc -= ((N_TV[s, :] * log(phi_TV[s, :])).sum() +
                (N_TV[t, :] * log(phi_TV[t, :])).sum())

        if log(uniform()) < min(0.0, acc):
            phi_TV[s, :] = zeros(V)
            phi_TV[t, :] = phi_merge_t_V
            active_topics.remove(s)
            inactive_topics.add(s)
            z_D[list(inv_z_merge_t)] = t
            inv_z_T[s].clear()
            inv_z_T[t] = inv_z_merge_t
            N_TV[s, :] = zeros(V, dtype=int)
            N_TV[t, :] = N_merge_t_V
            N_T[s] = 0
            N_T[t] = N_merge_t
            D_T[s] = 0
            D_T[t] = D_merge_t


def inference(N_DV, alpha, beta, z_D, num_itns, true_z_D=None):
    """
    Nonconjugate split-merge.
    """

    M = 10 # number of auxiliary samples

    D, V = N_DV.shape

    T = D + M - 1 # maximum number of topics

    N_D = N_DV.sum(1) # document lengths

    phi_TV = zeros((T, V)) # topic parameters

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

    # intialize topic parameters (necessary for Metropolis-Hastings only)

    for t in active_topics:
        phi_TV[t, :] = dirichlet(N_TV[t, :] + beta / V)

    for itn in xrange(num_itns):

        for _ in xrange(3):
            iteration(V, D, N_DV, N_D, alpha, beta, phi_TV, z_D, inv_z_T, active_topics, inactive_topics, N_TV, N_T, D_T, 6)

        algorithm_8_iteration(V, D, N_DV, N_D, alpha, beta, M, phi_TV, z_D, inv_z_T, active_topics, inactive_topics, N_TV, N_T, D_T)

        if true_z_D is not None:

            v = vi(true_z_D, z_D)

            print 'Itn. %d' % (itn + 1)
            print '%d topics' % len(active_topics)
            print 'VI: %f bits (%f bits max.)' % (v, log2(D))

            if v < 1e-6:
                break

    return phi_TV, z_D
