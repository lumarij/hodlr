from hodlr import *

def check_tolerance(A, B):
   diff = np.subtract(A, B)
   u1, s1, v1 = np.linalg.svd(A)
   u, s, v = np.linalg.svd(diff)
   print(s[0])
   print(threshold*s1[0]*len(A))
   if s[0] < threshold*s1[0]*len(A):
      return True
   else:
      return False