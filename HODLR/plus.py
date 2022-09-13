from hodlr import *
from compress import *
import copy

#zbrajanje dviju HODLR matrica
def hodlr_plus(H1, H2):
   H = copy.deepcopy(H1) 
   if H1.is_leafnode():
      H.F = np.add(H1.F, H2.F)
      return H
   else: 
      H.A11 = hodlr_plus(H1.A11, H2.A11)
      H.A22 = hodlr_plus(H1.A22, H2.A22)
      H.U12 = np.block([H1.U12, H2.U12])
      H.U21 = np.block([H1.U21, H2.U21])
      H.V12 = np.block([[H1.V12], [H2.V12]])
      H.V21 = np.block([[H1.V21],[H2.V21]])
      H.U12, H.V12 = compress_factors(H.U12,np.transpose(H.V12))
      H.U21, H.V21 = compress_factors(H.U21, np.transpose(H.V21))
      return H