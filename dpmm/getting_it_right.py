from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from numpy import array, empty_like, unique, zeros
from numpy.random import poisson, seed

from generative_process import generate_data
from kale.iterview import iterview
from kale.math_utils import sample
from pp_plot import pp_plot


def getting_it_right(algorithm, V, D, l, alpha, beta, num_itns, s):
    """
    Runs Geweke's "getting it right" test.
    """

    seed(s)

    # generate forward samples via the generative process

    print 'Generating forward samples...'

    forward_samples = []

    for _ in iterview(xrange(num_itns)):
        forward_samples.append(generate_data(V, D, l, alpha, beta)[1:])

    # generate reverse samples via the inference algorithm

    print 'Generating reverse samples...'

    reverse_samples = []

    phi_TV, z_D, _ = generate_data(V, D, l, alpha, beta)

    for _ in iterview(xrange(num_itns)):

        N_DV = zeros((D, V), dtype=int)

        if (algorithm.__name__ == 'algorithm_8' or
            algorithm.__name__ == 'nonconjugate_split_merge'):
            for d in xrange(D):
                for v in sample(phi_TV[z_D[d], :], num_samples=poisson(l)):
                    N_DV[d, v] += 1

            phi_TV, z_D = algorithm.inference(N_DV, alpha, beta, z_D, 1)

        else:

            T = D # maximum number of topics

            N_TV = zeros((T, V), dtype=int)
            N_T = zeros(T, dtype=int)

            for d in xrange(D):
                t = z_D[d]
                for _ in xrange(poisson(l)):
                    [v] = sample((N_TV[t, :] + beta / V) / (N_T[t] + beta))
                    N_DV[d, v] += 1
                    N_TV[t, v] += 1
                    N_T[t] += 1

            z_D = algorithm.inference(N_DV, alpha, beta, z_D, 1)

        z_D_copy = empty_like(z_D)
        z_D_copy[:] = z_D

        reverse_samples.append((z_D_copy, N_DV))

    print 'Computing test statistics...'

    # test statistics: number of topics, maximum topic size, mean
    # topic size, standard deviation of topic sizes

    # compute test statistics for forward samples

    forward_num_topics = []
    forward_max_topic_size = []
    forward_mean_topic_size = []
    forward_std_topic_size = []

    for z_D, _ in forward_samples:
        forward_num_topics.append(len(unique(z_D)))
        topic_sizes = []
        for t in unique(z_D):
            topic_sizes.append((z_D[:] == t).sum())
        topic_sizes = array(topic_sizes)
        forward_max_topic_size.append(topic_sizes.max())
        forward_mean_topic_size.append(topic_sizes.mean())
        forward_std_topic_size.append(topic_sizes.std())

    # compute test statistics for reverse samples

    reverse_num_topics = []
    reverse_max_topic_size = []
    reverse_mean_topic_size = []
    reverse_std_topic_size = []

    for z_D, _ in reverse_samples:
        reverse_num_topics.append(len(unique(z_D)))
        topic_sizes = []
        for t in unique(z_D):
            topic_sizes.append((z_D[:] == t).sum())
        topic_sizes = array(topic_sizes)
        reverse_max_topic_size.append(topic_sizes.max())
        reverse_mean_topic_size.append(topic_sizes.mean())
        reverse_std_topic_size.append(topic_sizes.std())

    # generate P-P plots

    pp_plot(array(forward_num_topics), array(reverse_num_topics))
    pp_plot(array(forward_max_topic_size), array(reverse_max_topic_size))
    pp_plot(array(forward_mean_topic_size), array(reverse_mean_topic_size))
    pp_plot(array(forward_std_topic_size), array(reverse_std_topic_size))


def main():

    import algorithm_3
    import algorithm_8
    import conjugate_split_merge
    import nonconjugate_split_merge

    functions = {
        'algorithm_3': algorithm_3,
        'algorithm_8': algorithm_8,
        'conjugate_split_merge': conjugate_split_merge,
        'nonconjugate_split_merge': nonconjugate_split_merge
        }

    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('algorithm', metavar='<inference-algorithm>',
                   choices=['algorithm_3',
                            'algorithm_8',
                            'conjugate_split_merge',
                            'nonconjugate_split_merge'],
                   help='inference algorithm to test')
    p.add_argument('-V', type=int, metavar='<vocab-size>', default=3,
                   help='vocabulary size')
    p.add_argument('-D', type=int, metavar='<num-docs>', default=10,
                   help='number of documents')
    p.add_argument('-l', type=int, metavar='<avg-doc-length>', default=10,
                   help='average document length')
    p.add_argument('--alpha', type=float, metavar='<alpha>', default=1.0,
                   help='concentration parameter for the DP')
    p.add_argument('--beta', type=float, metavar='<beta>', default=3.0,
                   help='concentration parameter for the Dirichlet prior')
    p.add_argument('--num-itns', type=int, metavar='<num-itns>', default=50000,
                   help='number of iterations')
    p.add_argument('--seed', type=int, metavar='<seed>',
                   help='seed for the random number generator')

    args = p.parse_args()

    getting_it_right(functions[args.algorithm],
                     args.V,
                     args.D,
                     args.l,
                     args.alpha,
                     args.beta,
                     args.num_itns,
                     args.seed)


if __name__ == '__main__':
    main()
