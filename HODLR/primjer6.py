import math
from scipy.sparse import lil_matrix
from check_tolerance import check_tolerance
from hodlr import *
from compress import *
from inv import *
#primjer 4.1.6
def construct_exp(n, k):
    a = lil_matrix((n,n)) #ne bi je inače spremali u ovom formatu, ali ovo je da možemo iskoristiti randomized
    for i in range(n):
        for j in range(n):
            a[i, j] = 0
            for m in range(1, k+1):
                a[i,j] = math.exp(m*abs(i-j)/n) +a[i, j]
            a[i,j] += pow(10,-12)
    return a
n = 512
k = 5

a = construct_exp(n, k)
b = HODLR(a, "sparse-randomized", 5, 0)
print(b.hodlr_rank())

b = HODLR(a, "sparse-randomized", k, 2)
print(b.hodlr_rank())
check_tolerance(a.toarray(),b.full())
b = HODLR(a, "sparse-randomized", k, 5)
print(b.hodlr_rank())
check_tolerance(a.toarray(),b.full())

a = a.toarray()
b = HODLR(a)
print(b.hodlr_rank())
check_tolerance(a, b.full())
b = np.linalg.inv(a)
bh = HODLR(b)
print(bh.hodlr_rank())
check_tolerance(b, bh.full())


k = 10
a = construct_exp(n, k)
b = HODLR(a, "sparse-randomized", k, 0)
print(b.hodlr_rank())
b = HODLR(a, "sparse-randomized", k, 2)
print(b.hodlr_rank())
check_tolerance(a.toarray(),b.full())
b = HODLR(a, "sparse-randomized", k, 5)
print(b.hodlr_rank())
check_tolerance(a.toarray(),b.full())

a = a.toarray()
b = HODLR(a)
print(b.hodlr_rank())
check_tolerance(a, b.full())
b = np.linalg.inv(a)
bh = HODLR(b)
print(bh.hodlr_rank())
check_tolerance(b, bh.full())


