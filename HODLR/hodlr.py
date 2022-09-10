
#from scipy.linalg import bandwidth
import numpy as np
from options import *
from compress import compress_matrix_svd
from compress import randomized_algorithm
from compress import lanczos_method
from scipy import sparse

block_min = 1
threshold = pow(10, -12)

class HODLR:
   
   def __init__(self, A, type=None,k = None, p= None):
      if type == "sparse":
         self.sz = A.shape[0]
         if self.sz > 64:
            tmp = self.sz//2
            self.F = []
            self.U12, self.V12 = randomized_algorithm(A.tocsr()[:tmp, tmp:], k, p)
            self.U21, self.V21 = randomized_algorithm(A.tocsr()[tmp:, :tmp], k, p)
            self.A11 = HODLR(A.tocsr()[:tmp,:tmp], "sparse", k, p)
            self.A22 = HODLR(A.tocsr()[tmp:,tmp:], "sparse",k, p)
         elif self.sz > block_min:
            tmp = self.sz//2
            self.F = []
            self.U12, self.V12 = compress_matrix_svd(A.tocsr()[:tmp, tmp:].todense())
            self.U21, self.V21 = compress_matrix_svd(A.tocsr()[tmp:,:tmp].todense())
            self.A11 = HODLR(A.tocsr()[:tmp,:tmp], "sparse")
            self.A22 = HODLR(A.tocsr()[tmp:,tmp:], "sparse")
         else:
            self.F = A.todense()
            self.A11 = []
            self.A22 = []
            self.U12 = []
            self.U21 = []
            self.V12 = []
            self.V21 = []
      else:  
         self.sz  = len(A) #broj stupaca = broj redaka jer zasad radim s kvadratnom matricom 
         if self.sz > block_min:
            self.F = []
            tmp = self.sz//2
            if type == "tridiagonal": 
               self.U12 = np.block([[np.zeros((tmp - 1, 1))], [A[tmp-1, tmp]]])
               self.V12 = np.block([[1, np.zeros(self.sz-tmp-1)]])
               self.U21 = np.block([[A[tmp, tmp-1]], [np.zeros((self.sz-tmp-1, 1))]])
               self.V21 = np.block([[np.zeros(tmp - 1), 1]])
            elif type == "diagonal" or type == "zeros": #zasad ovako vidit cu ako ima sto pametnije za pohranu da ne spremam nule u F
               self.U12 = np.zeros((tmp,0))
               self.V12 = np.zeros((0, self.sz-tmp))
               self.U21 = np.zeros((self.sz-tmp, 0))
               self.V21 = np.zeros((0, tmp))
            elif type == "ones":
               self.U12 = np.ones((tmp,1))
               self.V12 = np.ones((1, self.sz-tmp))
               self.U21 = np.ones((self.sz-tmp, 1))
               self.V21 = np.ones((1, tmp))
            elif type == "sparse":
               if(tmp > 256):
                  self.U12, self.V12 = randomized_algorithm(A.tocsr()[:tmp, tmp:], 5,5)
                  self.U21, self.V21 = randomized_algorithm(A.tocsr()[tmp:, :tmp], 5,5)
               else:
                  self.U12, self.V12 = compress_matrix_svd(A.tocsr()[:tmp, tmp:].todense())
                  self.U21, self.V21 = compress_matrix_svd(A.tocsr()[tmp:,:tmp].todense())
            # if type == "banded":
               #bandl , bandu = bandwidth(A)
               #if max(bandl, bandu) <= min(self.sz - tmp, tmp):
                  #self.U12 = np.block([[np.zeros((tmp - bandu,bandu))],[A[tmp - bandu:tmp, tmp:tmp+ bandu]]])
                  #self.V12 = np.block([np.eye(bandu), np.zeros((bandu, self.sz - tmp - bandu))]) 
                  #self.U21 = np.block([[A[tmp:tmp + bandl, tmp - bandl:tmp]],[np.zeros((self.sz - tmp - bandl, bandl))]])
                  #self.V21 = np.block([np.zeros((bandl, tmp - bandl)), np.eye(bandl)]) 
               #else:
                  #self.U12, self.V12 = compress_matrix_svd(A[:tmp, tmp:])
                  #self.U21, self.V21 = compress_matrix_svd(A[tmp:,:tmp])   
            else:
               self.U12, self.V12 = compress_matrix_svd(A[:tmp, tmp:])
               self.U21, self.V21 = compress_matrix_svd(A[tmp:,:tmp])
               self.A11 = HODLR(A[:tmp,:tmp], type)
               self.A22 = HODLR(A[tmp:,tmp:], type)
         else:
            self.F = A
            self.A11 = []
            self.A22 = []
            self.U12 = []
            self.U21 = []
            self.V12 = []
            self.V21 = []
    
   
      
   
   def is_leafnode(self):
      return False if self.A11 else True
   
   def full(self):
   #HODLR matricu pretvara u njen originalni "gusti" oblik
      if self.is_leafnode():
         return self.F
      else:
         a11 = self.A11.full()
         a12 = np.matmul(self.U12, self.V12)
         a21 = np.matmul(self.U21, self.V21)
         a22 = self.A22.full()
         sz = self.A11.sz
         A = np.zeros((self.A11.sz + self.A22.sz,self.A11.sz + self.A22.sz ))
         A[:sz, :sz] = a11
         A[:sz, sz:] = a12
         A[sz:, :sz] = a21
         A[sz:, sz:] = a22
         return A
      
   def hodlr_rank(self):
      if self.is_leafnode():
         return 0
      else:
         return max(self.A11.hodlr_rank(), self.A22.hodlr_rank(), len(self.V12), len(self.V21))
   
   def hodlr_depth(self):
      if self.is_leafnode():
         return 1
      else:
         return 1 + self.A11.hodlr_depth()
   
   def storage_complexity(self):
      if self.is_leafnode():
         return len(self.F)*len(self.F)
      else:
         if(len(self.V12)) == 0:
               return self.A11.storage_complexity() + self.A22.storage_complexity() + len(self.U12)*len(self.U12[0])+ \
                      len(self.U21)*len(self.U21[0])
         else:
            return self.A11.storage_complexity() + self.A22.storage_complexity() + len(self.U12)*len(self.U12[0])+ \
                        len(self.V12)* len(self.V12[0]) + len(self.U21)*len(self.U21[0])+ len(self.V21)* len(self.V21[0])
      