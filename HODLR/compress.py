import numpy as np
from hodlr import *
threshold = pow(10, -12)
from scipy.sparse import csr_matrix

def compress_matrix_svd(A):
   u, s, v = np.linalg.svd(A, full_matrices = True)
   smatrix = np.diagflat(s)
   k = 0
   for i in s:
      if(abs(i) > s[0] * threshold):
         k += 1 #numericki rang
   u = np.matmul(u[:,:k], smatrix[:k, :k])
   v = v[:k,:] #python vec vrati transponirano pa uzimam k redaka ne k stupaca
   return u, v
   
def rank_svd(A):
   u, s, v = np.linalg.svd(A, full_matrices = True)
   smatrix = np.diagflat(s)
   k = 0
   for i in s:
      if(abs(i) > s[0] * threshold):
         k += 1 #numericki rang
   return k

def compress_factors(Uold, Vold):
   if len(Uold) == 0 or len(Uold[0]) == 0:
      U = Uold
      V = Vold
      return U, np.transpose(V)
   else:
      qu, ru = np.linalg.qr(Uold, 'reduced')
      qv, rv = np.linalg.qr(Vold, 'reduced')
      u, s ,v = np.linalg.svd(np.matmul(ru,np.transpose(rv))) 
      nrm = s[0]
      k = 0
      for i in s:
         if(abs(i) > nrm * threshold):
            k += 1
      smatrix = np.diagflat(s)
      U = np.matmul(qu, u[:, :k])
      U = np.matmul(U, smatrix[:k, :k])
      v = np.transpose(v)
      V = np.matmul(qv, v[:, :k])
      V = np.transpose(V)
      return U, V


def lanczos_method(A, k): 
   m = A.shape[0]
   n = A.shape[1]
   u = np.zeros((k, m))
   v = np.zeros((k, n))
   alfa = np.zeros(k)
   beta = np.zeros(k-1)
   u1 = np.random.rand(m)
   norm = np.linalg.norm(u1)
   u[0] = u1/norm
   v[0] = csr_matrix.transpose(A)@u[0] 
   alfa[0] = np.linalg.norm(v[0])
   v[0] = v[0]/alfa[0]
   for i in range(k-1):
      u1 = np.subtract((A@v[i]), alfa[i]*u[i]) 
      beta[i] = np.linalg.norm(u)
      tmp = beta[i]
      u[i+1] = u1/tmp
      v1 = np.subtract(csr_matrix.transpose(A)@u[i+1], beta[i]*v[i]) 
      alfa[i+1] = np.linalg.norm(v1)
      v[i+1] = v1/alfa[i+1]
   B1 = np.diagflat(alfa)
   B2 = np.diagflat(beta, -1) 
   B = np.add(B1, B2)
   ub, sb, vb = np.linalg.svd(B)
   rk = 0
   for i in sb:
      if(abs(i) > sb[0] * threshold):
         rk += 1
   U = np.matmul(np.transpose(u),ub[:, :rk] )
   S = np.diagflat(sb)
   S = S[:rk, :rk]
   V = np.matmul(np.transpose(v), np.transpose(vb)[: ,:rk])
   return np.matmul(U, S), np.transpose(V)

def randomized_algorithm(A, k, p): 
   m = A.shape[0]
   n = A.shape[1]
   O = np.random.normal(size = (n, p+k)) 
   Y = A@O 
   q, r = np.linalg.qr(Y, 'reduced')
   Z = np.transpose(q)@ A
   u, s, v = np.linalg.svd(Z)
   rk = 0
   for i in s:
      if(abs(i) > s[0] * threshold):
         rk += 1
   U = np.matmul(q, u[:, :rk])
   
   S = np.diagflat(s)
   S = S[:rk, :rk]
   V = v[:rk, :]
   U = np.matmul(U,S)
   return U, V



   
  




  



      

