from hodlr import *
from lu import *
from times import *
from transpose import *
from change_sign import *
from check_tolerance import *
#za hilbert ne radi dobro
# i za donjetrokutaste

def hodlr_inv(H):
   IH = copy.deepcopy(H)
   if H.is_leafnode():
      IH.F = np.linalg.inv(H.F)
   elif len(H.U12[0]) == 0:
      IH = inv_lower_triangular(H)
   elif len(H.U21[0]) == 0:
      IH = inv_upper_triangular(H)
   else:
      HL, HU = hodlr_lu(H)
      #IH = np.linalg.solve(HU.full(), inv_lower_triangular(HL).full())
      #IH = HODLR(IH)
      IH = hodlr_times(inv_upper_triangular(HU), inv_lower_triangular(HL))
   return IH

def inv_upper_triangular(H):
   IH = copy.deepcopy(H)
   IH.A11 = hodlr_inv(H.A11)
   IH.A22 = hodlr_inv(H.A22)
   IH.U12 = hodlr_times_dense(change_sign(IH.A11), H.U12)
   IH.V12 = np.transpose(hodlr_times_dense(transpose(IH.A22), np.transpose(H.V12)))
   return IH

def inv_lower_triangular(H):
   IH = copy.deepcopy(H)
   IH.A11 = hodlr_inv(H.A11)
   IH.A22 = hodlr_inv(H.A22)
   IH.U21 = hodlr_times_dense(change_sign(IH.A22), H.U21)
   IH.V21 = np.transpose(hodlr_times_dense(transpose(IH.A11), np.transpose(H.V21)))
   return IH

