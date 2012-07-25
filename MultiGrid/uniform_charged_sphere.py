from __future__ import division
from scipy.constants import epsilon_0 as ep0
from scipy.sparse import spdiags
from numpy import pi 
import numpy as np
import matplotlib.pyplot as plt


def exact(r, q, x):
    coeff = q / (4 * pi * ep0)
    y = np.zeros_like(x)
    
    i0 = (x > r).argmax()
    y[:i0] = coeff * x[:i0] / (r**3)
    y[i0:] = coeff / (x[i0:]**2)
    
    return y
    
    
def scipy_sparse_matrix(r, cd, x, n):
    A = spdiags([-np.ones(n), 2*np.ones(n), -np.ones(n)], [-1, 0, 1], n, n)
    f = np.zeros_like(x)
    y = np.zeros_like(x)
    
    dx = x[1] - x[0]
    i0 = (x > r).argmax()
    f[:i0] = cd * dx**2 / ep0
    
    from numpy.linalg import solve
    potential = solve(A.todense(), f)
    y[:-1] = - np.diff(potential)
    
    return y



if __name__ == '__main__':
    radius = 0.5
    charge_density = 1e-9
    x0, x1 = 0, 2
    
    n = 100
    x = np.linspace(x0, x1, num=n, endpoint=False)    
    
    charge = (4 / 3) * pi * radius**3 * charge_density
    e_exact = exact(radius, charge, x)
    
    e_scipy = scipy_sparse_matrix(radius, charge_density, x, n)
    
    # plot    
    plt.plot(x, e_exact, '.-k', label='Exact')
    plt.plot(x, e_scipy, '.-b', label='Scipy')
    plt.title('Uniformly charged sphere')
    plt.xlabel('x')
    plt.ylabel('E')
    plt.legend()
    plt.show()