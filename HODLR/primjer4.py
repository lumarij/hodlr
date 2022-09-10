import scipy.sparse
#from hodlr import *
from scipy.linalg import hilbert
from scipy.sparse import lil_matrix
from check_tolerance import *
from lu import *

n = 1024
a = scipy.sparse.random(n,n,density = 0.000025)

b = HODLR(a,"sparse",5,0)
print(b.hodlr_rank())
print(check_tolerance(a.toarray(), b.full()))
b = HODLR(a,"sparse",5,1)
print(b.hodlr_rank())
print(check_tolerance(a.toarray(), b.full()))
b = HODLR(a,"sparse",5,3)
print(b.hodlr_rank())
print(check_tolerance(a.toarray(), b.full()))
b = HODLR(a,"sparse",5,5)
print(b.hodlr_rank())
print(check_tolerance(a.toarray(), b.full()))
b = HODLR(a,"sparse",5,10)
print(b.hodlr_rank())
print(check_tolerance(a.toarray(), b.full()))
#arrow = lil_matrix((n, n))
#def construct_arrowhead(n):
    #for i in range(n):
        #for j in range(n):
            #if i == 0:
                #arrow[i , j] = 1
            #if j == 0:
                #arrow[i , j] = 1
            #if i == j:
                #arrow[i, j] = 1
#construct_arrowhead(n)
#b = HODLR(arrow, "sparse")
#print(b.hodlr_rank())
#print(check_tolerance(arrow.toarray(),b.full()))


