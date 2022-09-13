from random import randint
import scipy.sparse
from hodlr import *
from scipy.linalg import hilbert
from scipy.sparse import lil_matrix
from check_tolerance import *
from inv import hodlr_inv
from lu import *

#primjer 4.1.5

def random_banded_matrix(n, bandl, bandu):
   if bandl >= n or bandu >= n:
      print("Incorrect bandwidth!")
      return
   a = lil_matrix((n, n))
   for i in range(n):
      for j in range(n):
         if i >= j:
            if (i - j) <= bandl:
               a[i,j] = randint(1,9)
         else:
            if (j - i) <= bandu:
               a[i,j] = randint(1,9)
   return a
n = 1024
a = random_banded_matrix(n,1,1)
b = HODLR(a,"sparse-randomized",1,0)
print(b.hodlr_rank())
check_tolerance(a.toarray(), b.full())

ia = np.linalg.inv(a.toarray())
ib = HODLR(ia)
print(ib.hodlr_rank())
check_tolerance(np.linalg.inv(a.toarray()), ib.full())


a = random_banded_matrix(n,5,5)
b = HODLR(a,"sparse-randomized",5,0)
print(b.hodlr_rank())
check_tolerance(a.toarray(), b.full())

ia = np.linalg.inv(a.toarray())
ib = HODLR(ia)
print(ib.hodlr_rank())
check_tolerance(np.linalg.inv(a.toarray()), ib.full())


a = random_banded_matrix(n,10,10)
b = HODLR(a,"sparse-randomized",10,0)
print(b.hodlr_rank())
check_tolerance(a.toarray(), b.full())

ia = np.linalg.inv(a.toarray())
ib = HODLR(ia)
print(ib.hodlr_rank())
check_tolerance(np.linalg.inv(a.toarray()), ib.full())

n = 2048
a = random_banded_matrix(n,1,1)
b = HODLR(a,"sparse-randomized",1,0)
print(b.hodlr_rank())
check_tolerance(a.toarray(), b.full())

ia = np.linalg.inv(a.toarray())
ib = HODLR(ia)
print(ib.hodlr_rank())
check_tolerance(np.linalg.inv(a.toarray()), ib.full())


a = random_banded_matrix(n,5,5)
b = HODLR(a,"sparse-randomized",5,0)
print(b.hodlr_rank())
check_tolerance(a.toarray(), b.full())

ia = np.linalg.inv(a.toarray())
ib = HODLR(ia)
print(ib.hodlr_rank())
check_tolerance(np.linalg.inv(a.toarray()), ib.full())


a = random_banded_matrix(n,10,10)
b = HODLR(a,"sparse-randomized",10,0)
print(b.hodlr_rank())
check_tolerance(a.toarray(), b.full())

ia = np.linalg.inv(a.toarray())
ib = HODLR(ia)
print(ib.hodlr_rank())
check_tolerance(np.linalg.inv(a.toarray()), ib.full())






