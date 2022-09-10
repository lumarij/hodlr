from hodlr import *
from lu import *

def solve_upper_triangular2(H,b):
   if H.is_leafnode():
      return np.linalg.solve(H.F, b)
   else:
      n = H.A11.sz
      x2 = solve_upper_triangular2(H.A22, b[n:,:])
      x1 = solve_upper_triangular2(H.A11,b[:n,:] - np.matmul(H.U12,np.matmul(H.V12,x2)))
      return np.block([[x1],[x2]])

def solve_lower_triangular2(H,b):
   if H.is_leafnode():
      return np.linalg.solve(H.F, b)
   else:
      n = H.A11.sz
      x1 = solve_lower_triangular2(H.A11, b[:n,:])
      x2 = solve_lower_triangular2(H.A22, b[n:,:] - np.matmul(H.U21,np.matmul(H.V21,x1)))
      return np.block([[x1],[x2]])



