import math
import scipy, pylab
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import tril
from pylab import *

from scipy.linalg import norm,inv
from scipy.sparse import spdiags,SparseEfficiencyWarning,csc_matrix
from scipy.sparse.linalg.dsolve import spsolve,use_solver,splu,spilu

levels = 7  # size of problem
nu1 = 2     # number of presmoothing iterations
nu2 = 2     # number of postsmoothing iterations
gamma = 1   # number of coarse grid iterations (1=V-cycle, 2=W-cycle)

n = math.pow(2, levels+2)-1 # number of grid points
h = 1/(n+1)
x = scipy.linspace(h, 1-h, num=int(n)+1)

temp = 0.0

for i in range(1,int(n+1)):
        temp += h
        x[i] = temp

ff = range(int(n)+1)
ff[1:int(n+1)] = math.pow(np.pi,2) * (np.sin(np.pi*x[1:int(n+1)]) + pow(4,2) * np.sin(np.pi * 4* x[1:int(n+1)]) + math.pow(9,2) * np.sin(np.pi * 9* x[1:int(n+1)]))

e = np.ones(n);
A = spdiags( [-e, 2*e, -e], [-1,0,1], n, n).todense()
b = range(int(n)+1)

for i in range(1,int(n+1)):
        b[i] = ff[i] * math.pow(h,2)
#print b
u = range(int(n)+1)

def twogrid(FA,Fb,Fnu1,Fnu2,Fgamma,Fuu):
        nn = len(Fb)
        G = [[0.0] * (nn+1) for _ in [0.0]* (nn+1)]
#    ltemp = 0
        G = scipy.sparse.eye(nn,nn) - (inv(tril(FA))*FA)
#    G = scipy.sparse.eye(nn,nn) 
        print G
#   cG = inv(tril(FA))*Fb
        return Fuu

twogrid(A,b,nu1,nu2,gamma,u)


################# error message ################# 
#kuma@g105:~/Multigrid$ python MG_V2.py 
#Traceback (most recent call last):
#  File "MG_V2.py", line 50, in <module>
#    twogrid(A,b,nu1,nu2,gamma,u)
#  File "MG_V2.py", line 44, in twogrid
#    G = scipy.sparse.eye(nn,nn) - (inv(tril(FA))*FA)
#  File "/usr/lib/python2.7/dist-packages/scipy/sparse/base.py", line 224, in __sub__
#    return self.tocsr().__sub__(other)
#  File "/usr/lib/python2.7/dist-packages/scipy/sparse/compressed.py", line 202, in __sub__
#    return self.todense() - other
#ValueError: shape mismatch: objects cannot be broadcast to a single shape
