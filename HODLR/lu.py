from hodlr import *
from times import *
from scipy.linalg import lu
import copy
#približna lu faktorizacija HODLR matrice
def hodlr_lu(H):
   HL = copy.deepcopy(H)
   HU = copy.deepcopy(H)
   if H.is_leafnode():
      HL.F, HU.F = lu(H.F, permute_l=True)
      return HL, HU
   else:
      n = H.sz
      m1 = H.A11.sz
      HL.U12 = np.zeros((m1, 0))
      HL.V12 = np.zeros((0,n-m1))
      HU.U21 = np.zeros((n-m1,0))
      HU.V21 = np.zeros((0, m1))
      HL.A11, HU.A11 = hodlr_lu(H.A11)
      HU.U12 = solve_lower_triangular(HL.A11, H.U12)
      HL.V21 = solve_upper_triangular(HU.A11, H.V21)
      U = copy.deepcopy(H.A22)
      V1 = np.matmul(HL.V21, HU.U12)
      V1 = np.matmul((-1)*HL.U21, V1)
      V2 = HU.V12
      HL.A22, HU.A22 = hodlr_lu(hodlr_rank_update(U,V1,V2))
      return HL,HU

def solve_lower_triangular(H1, x):
   if H1.is_leafnode():
      y = np.linalg.solve(H1.F, x)
      return y
   else:
      n = H1.A11.sz
      x2 = solve_lower_triangular(H1.A11, x[:n, :])
      x1 = solve_lower_triangular(H1.A22, x[n:,:] - np.matmul(H1.U21,np.matmul(H1.V21,x2)))
      y = np.block([[x2], [x1]])
      return y

def solve_upper_triangular(H1, x):
   if H1.is_leafnode():
      y = np.matmul(x, np.linalg.inv(H1.F)) 
      return y
   else:
      n = H1.A11.sz
      y1 = solve_upper_triangular(H1.A11, x[:,:n])
      y2 = solve_upper_triangular(H1.A22, x[:,n:] - np.matmul(np.matmul(y1, H1.U12),H1.V12))
      y = np.block([y1,y2])
      return y   

