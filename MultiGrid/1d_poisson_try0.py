# Solve a 1D Poisson equation
# Written by Ki-Hwan Kim
# 2012.1.12

from __future__ import division
from scipy.sparse import spdiags, eye, tril
import numpy as np
import matplotlib.pyplot as plt


def scipy_sparse_matrix(A, b, u):
    #from numpy.linalg import solve
    #u[1:-1] = solve(A.todense(), b)
    
    from scipy.sparse.linalg import spsolve
    u[1:-1] = spsolve(A.tocsr(), b)
    
    return u


'''
def gauss_seidel(A, b, u, itermax=50):
    mb = np.mat(b)
    mu = np.mat(u)
    invP = np.linalg.inv(tril(A).todense())
    G = eye(*A.shape) - invP * A
    cG = np.inner(invP, b)
    
    for i in xrange(itermax):
        mu = np.inner(G, mu) + cG
    u[:] = mu
        
    return u
'''



#====================================================
# main
#----------------------------------------------------
# setup
omega = 5.4
u0, u1 = 0, 1
func = lambda x: omega**2 * np.sin(omega * x)

n = 100
x = np.linspace(0, 1, n)    

#----------------------------------------------------
# make A, b, u
A = spdiags([-np.ones(n-2), 2*np.ones(n-2), -np.ones(n-2)], [-1, 0, 1], n-2, n-2)

dx = x[1] - x[0]
b = func(x[1:-1]) * dx**2
b[0] += u0; b[-1] += u1

u = np.zeros_like(x)
u[0] = u0
u[-1] = u1
    
#----------------------------------------------------
# solutions
u_exact = np.sin(omega * x) - (np.sin(omega) - 1) * x
u_scipy = scipy_sparse_matrix(A, b, u.copy())
#u_gs = gauss_seidel(A, b, u.copy())

#----------------------------------------------------
# plot    
plt.plot(x, u_exact, '.-k', label='Exact')
plt.plot(x, u_scipy, '.-b', label='Scipy')
#plt.plot(x, u_gs, '.-r', label='Gauss-Seidel')
plt.title('1-D Poisson')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()
