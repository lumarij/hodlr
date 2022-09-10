from math import exp, log, pi, sin, sqrt
import math
from compress import *
from hodlr import *


n = int(input("Unesite veličinu matrice:"))
tmp = n//2
quadric = np.zeros((tmp,tmp))
multiquadric = np.zeros((tmp,tmp))
inverse_quadric = np.zeros((tmp,tmp))
inverse_multiquadric = np.zeros((tmp,tmp))
exponential = np.zeros((tmp,tmp))
gaussian = np.zeros((tmp,tmp))
logarithm = np.zeros((tmp,tmp))
for i in range(tmp):
    ti = (tmp+i)*2*pi/n
    for j in range(tmp):
        tj = j*2*pi/n
        r = 2*abs(sin((ti-tj)/2))
        quadric[i, j] = 1+ math.pow(r,2)
        multiquadric[i, j] = sqrt(quadric[i, j])
        inverse_quadric[i, j] = 1/quadric[i, j]
        inverse_multiquadric[i, j] = 1/multiquadric[i, j]
        exponential[i, j] = exp(-r)
        gaussian[i,j] = exp(-pow(r, 2))
        logarithm[i, j] = log(1+r)

print("quadric")
print(rank_svd(quadric))
print("multiquadric")
print(rank_svd(multiquadric))
print("inverse_quadric")
print(rank_svd(inverse_quadric))
print("inverse_multiquadric")
print(rank_svd(inverse_multiquadric))
print("exponential")
print(rank_svd(exponential))
print("gaussian")
print(rank_svd(gaussian))
print("logarithm")
print(rank_svd(logarithm))