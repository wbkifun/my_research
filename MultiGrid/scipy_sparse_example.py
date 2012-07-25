from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

n = 10
A = lil_matrix((n, n))
A[0,:] = rand(n)
A[1,:] = A[0,:]
A.setdiag(rand(n))

b = rand(n)
A = A.tocsr()
x = spsolve(A, b)

x_ = solve(A.todense(), b)

if norm(x-x_) < 1e-9:
    print 'OK!'
else:
    print 'Failed!'