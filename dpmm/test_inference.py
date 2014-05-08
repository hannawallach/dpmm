from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from numpy import argsort, bincount, set_printoptions, where, zeros
from numpy.random import seed

from generative_process import generate_data


def test_inference(algorithm, V, D, l, alpha, beta, num_itns, s):
    """
    Generates data via the generative process and then infers the
    parameters of the generative process using that data.
    """

    seed(s)

    print 'Generating data...'

    phi_TV, z_D, N_DV = generate_data(V, D, l, alpha, beta)

    set_printoptions(precision=4, suppress=True)

    for t in argsort(bincount(z_D))[::-1]:
        idx, = where(z_D[:] == t)
        print len(idx), phi_TV[t, :]

    print 'Running inference...'

    # initialize every document to the same topic

    algorithm.inference(N_DV, alpha, beta, zeros(D, dtype=int), num_itns, z_D)


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
    p.add_argument('-V', type=int, metavar='<vocab-size>', default=5,
                   help='vocabulary size')
    p.add_argument('-D', type=int, metavar='<num-docs>', default=1000,
                   help='number of documents')
    p.add_argument('-l', type=int, metavar='<avg-doc-length>', default=1000,
                   help='average document length')
    p.add_argument('--alpha', type=float, metavar='<alpha>', default=1.0,
                   help='concentration parameter for the DP')
    p.add_argument('--beta', type=float, metavar='<beta>', default=0.5,
                   help='concentration parameter for the Dirichlet prior')
    p.add_argument('--num-itns', type=int, metavar='<num-itns>', default=250,
                   help='number of iterations')
    p.add_argument('--seed', type=int, metavar='<seed>',
                   help='seed for the random number generator')

    args = p.parse_args()

    test_inference(functions[args.algorithm],
                   args.V,
                   args.D,
                   args.l,
                   args.alpha,
                   args.beta,
                   args.num_itns,
                   args.seed)


if __name__ == '__main__':
    main()
