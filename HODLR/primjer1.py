from scipy.linalg import hilbert
from hodlr import *
from options import *
from compress import *

print(5)
print(HODLR(hilbert(5)).hodlr_rank())

print(HODLR(hilbert(5)).storage_complexity())
print(10)
print(HODLR(hilbert(10)).hodlr_rank())
print(HODLR(hilbert(10)).storage_complexity())
print(15)
print(HODLR(hilbert(15)).hodlr_rank())
print(HODLR(hilbert(15)).storage_complexity())
print(20)
print(HODLR(hilbert(20)).hodlr_rank())
print(HODLR(hilbert(20)).storage_complexity())
print(50)
print(HODLR(hilbert(50)).hodlr_rank())
print(HODLR(hilbert(50)).storage_complexity())
print(100)
print(rank_svd(hilbert(100)))
print(HODLR(hilbert(100)).hodlr_rank())
print(HODLR(hilbert(100)).storage_complexity())
print(500)
print(HODLR(hilbert(500)).hodlr_rank())
print(HODLR(hilbert(500)).storage_complexity())
print(1000)
print(HODLR(hilbert(1000)).hodlr_rank())
print(HODLR(hilbert(1000)).storage_complexity())
print(2000)
print(HODLR(hilbert(2000)).hodlr_rank())
print(HODLR(hilbert(2000)).storage_complexity())
print(5000)
print(HODLR(hilbert(5000)).hodlr_rank())
print(HODLR(hilbert(5000)).storage_complexity())
print(HODLR(hilbert(7000)).hodlr_rank())

