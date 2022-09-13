from hodlr import *
import copy

#mijenjanje predznaka svim elementima HODLR matrice
def change_sign(H):
   mH = copy.deepcopy(H)
   if H.is_leafnode():
      mH.F = (-1)*H.F
   else:
      mH.A11 = change_sign(H.A11)
      mH.A22 = change_sign(H.A22)
      mH.U21 = (-1)*H.U21
      mH.U12 = (-1)*H.U12
   return mH

      

