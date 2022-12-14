from HODLR.hodlr import *
from compress import *
from scipy.linalg import hilbert
import math

#primjer 4.1.2
n = 1024
a = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        a[i, j] = math.exp(abs(i-j)/n)
print(rank_svd(a))
print(HODLR(a).hodlr_rank())
