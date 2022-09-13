from hodlr import *
from lu import *

#rješavanje linearnih sustava Ax = b, gdje je A HODLR matrica
def solve_linear(H, b):
   l, u = hodlr_lu(H)
   y = solve_lower_triangular2(l, b)
   return solve_upper_triangular2(u,y)

def solve_lower_triangular2(H, b):
   if H.is_leafnode():
      return np.linalg.solve(H.F, b)
   else:
      n = H.A11.sz
      x1 = solve_lower_triangular2(H.A11, b[:n])
      x2 = solve_lower_triangular2(H.A22, b[n:] - np.matmul(H.U21,np.matmul(H.V21,x1)))
      return np.concatenate([x1, x2])

def solve_upper_triangular2(H, b):
   if H.is_leafnode():
      return np.linalg.solve(H.F, b)
   else:
      n = H.A11.sz
      x2 = solve_upper_triangular2(H.A22, b[n:])
      x1 = solve_upper_triangular2(H.A11,b[:n] - np.matmul(H.U12,np.matmul(H.V12,x2)))
      return np.concatenate([x1,x2])





