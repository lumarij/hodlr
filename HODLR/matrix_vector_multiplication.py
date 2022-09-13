from hodlr import *
from compress import *

#množenje HODLR matrice i vektora
def matrix_vector_multiplication(H,v):
   if H.is_leafnode():
      return np.matmul(H.F, v)
   else:
      v1 = v[:H.sz//2]
      v2 = v[H.sz//2:]
      mult1 = np.matmul(H.V12, v2)
      mult1 = np.matmul(H.U12, mult1)
      mult2 = np.matmul(H.V21, v1)
      mult2 = np.matmul(H.U21, mult2)
      return np.concatenate([np.add(matrix_vector_multiplication(H.A11, v1), mult1),np.add(matrix_vector_multiplication(H.A22, v2), mult2) ])