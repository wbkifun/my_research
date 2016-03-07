from __future__ import division
import numpy as np
import numpy.polynomial.legendre as lgd


def get_c_leg_list(max_order):
    c_leg_list = [ [0]*i for i in xrange(1, max_order+2) ]
    for c_leg in c_leg_list: c_leg[-1] = 1

    return c_leg_list


def weights_roots(order, rules='Gauss-Lobatto'):
    c_leg = [1] if order == 0 else [i//order for i in xrange(order+1)]
    c_dleg = lgd.legder(c_leg)

    if rules == 'Gauss-Lobatto':
        xs = np.array( [-1] + list( lgd.legroots(c_dleg) ) + [1] )
        ws = 2 / ( order * (order + 1) * (lgd.legval(xs, c_leg)**2) )

    elif rules == 'Gauss-Legendre':
        xs = lgd.legroots(c_leg)
        ws = 2 / ( (1 - xs**2) * (lgd.legval(xs, c_dleg)**2) )

    return xs, ws


def integrate(func, order, rules='Gauss-Lobatto', a=-1, b=1):
    xs, ws = weights_roots(order, rules)

    if a == -1 and b == 1:
        val = sum( ws*func(xs) )
    else:
        val = (b-a)*0.5*sum( ws*func( (b-a)*0.5*xs + (b+a)*0.5 ) )

    return val



if __name__ == '__main__':
    order = 8
    xs, ws = weights_roots(order, 'Gauss-Lobatto')

    print "Order  : ", order
    print "c_leg_list: "
    for c_leg in get_c_leg_list(max_order=order):
        print c_leg
    print "Roots  : ", xs
    print "Weights: ", ws
    print ""

    # Integrating the function
    func = lambda x: np.exp(x)
    a, b = -1, 1
    exact = np.exp(b) - np.exp(a)
    gq_lo = integrate(func, order, 'Gauss-Lobatto')
    gq_le = integrate(func, order, 'Gauss-Legendre')

    print "exact: ", exact
    print "gq_lo: ", gq_lo
    print "gq_le: ", gq_le
    print "diff (Lobatto) : ", exact - gq_lo
    print "diff (Legendre): ", exact - gq_le
