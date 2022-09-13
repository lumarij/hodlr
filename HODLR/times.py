from hodlr import *
from compress import *
import copy

#množenje dviju HODLR matrica

def hodlr_times(H1, H2):
   H = hodlr_matrix_multiplication(H1, H2)
   H.U12, H.V12 = compress_factors(H.U12, np.transpose(H.V12));
   H.U21, H.V21 = compress_factors(H.U21, np.transpose(H.V21));
   return H


def hodlr_matrix_multiplication(H1, H2):
   H = copy.deepcopy(H1)

   if H1.is_leafnode():
      H.F = np.matmul(H1.F, H2.F)

   else:
      #blok 11
      H.A11 = hodlr_matrix_multiplication(H1.A11, H2.A11)
      U = np.matmul(H1.V12, H2.U21)
      U = np.matmul(H1.U12, U)
      H.A11 = hodlr_rank_update(H.A11, U, H2.V21)
      
      #blok 22
      H.A22 = hodlr_matrix_multiplication(H1.A22, H2.A22)
      U = np.matmul(H1.V21,H2.U12)
      U = np.matmul(H1.U21, U)
      H.A22 = hodlr_rank_update(H.A22, U, H2.V12)

      #blok 12
      H.U12 = np.block([hodlr_times_dense(H1.A11, H2.U12 ), H1.U12] )
      H.V12 = np.block([[H2.V12], [dense_times_hodlr(H1.V12, H2.A22)]])
      

      #blok21
      H.U21 = np.block([hodlr_times_dense(H1.A22, H2.U21 ), H1.U21] )
      H.V21 = np.block([[H2.V21], [dense_times_hodlr(H1.V21, H2.A11)]])
    
   return H
#množenje HODLR matrice i gusto popunjene matrice
def hodlr_times_dense(H, v):
   if H.is_leafnode():
      return np.matmul(H.F, v)
   else:
      m = H.A11.sz
      v1 = v[:m,:]
      v2 = v[m:,:]
      w1 = hodlr_times_dense(H.A11, v1)
      w2 = hodlr_times_dense(H.A22, v2)
      w1 = np.add(w1, np.matmul(H.U12, np.matmul(H.V12,v2)))
      w2 = np.add(w2, np.matmul(H.U21, np.matmul(H.V21,v1)))
      return np.block([[w1], [w2]])

#množenje gusto popunjene matrice i HODLR matrice
def dense_times_hodlr(v, H):
   if H.is_leafnode():
      return np.matmul(v, H.F)
   else:
      m = H.A11.sz
      v1 = v[:, :m]
      v2 = v[:, m:]
      w1 = dense_times_hodlr(v1, H.A11)
      w2 = dense_times_hodlr(v2, H.A22)
      w1 = np.add(w1, np.matmul(np.matmul(v2, H.U21), H.V21))
      w2 = np.add(np.matmul(np.matmul(v1, H.U12), H.V12), w2)
      return np.block([w1,w2])

#H+UV, gdje je H HODLR matrica a U i V gusto popunjene matrice 
def hodlr_rank_update(H, U, V):
   if H.is_leafnode():
      H.F = np.add(H.F, np.matmul(U, V))
      return H
   else:
      m = H.A11.sz
      H.U12, H.V12 = compress_factors(np.block([H.U12, U[:m, :]]), np.transpose(np.block([[H.V12], [V[: ,m:]]])))
      H.U21, H.V21 = compress_factors(np.block([H.U21, U[m:, :]]), np.transpose(np.block([[H.V21], [V[: ,:m]]])))
      H.A11 = hodlr_rank_update(H.A11, U[:m, : ],V[:, :m])
      H.A22 = hodlr_rank_update(H.A22, U[m:, :], V[:, m:])
      return H
