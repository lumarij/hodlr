from hodlr import *
import copy
#transponiranje HODLR matrice
def transpose(H):
   HT = copy.deepcopy(H)
   if H.is_leafnode():
      HT.F = np.transpose(H.F)
   else:
      HT.U12 = np.transpose(H.V21)
      HT.V12 = np.transpose(H.U21)
      HT.U21 = np.transpose(H.V12)
      HT.V21 = np.transpose(H.U12)
      HT.A11 = transpose(H.A11)
      HT.A22 = transpose(H.A22)
   return HT