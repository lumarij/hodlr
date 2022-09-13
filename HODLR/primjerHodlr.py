from hodlr import *
#primjer 3.1.3

d = np.array([2, -1, 1, 2, 2, 1, 1, 1])
d1 = np.array([2, 1, 2, 1, 2, 2, 2])
d2 = np.array([1, 1, 1, 1, 1, 1, 1])
a1 = np.diagflat(d)
a2 = np.diagflat(d1, -1)
a3 = np.diagflat(d2, 1)
a = np.add(np.add(a1,a2),a3)
b = np.linalg.inv(a)
h = HODLR(b)
print("Razina 1- blok A12", h.U12, h.V12 )
print("Razina 1- blok A21", h.U21, h.V21 )
print("Razina 2 - blok A11.A12", h.A11.U12, h.A11.V12)
print("Razina 2 - blok A11.A21", h.A11.U21, h.A11.V21)
print("Razina 2 - blok A11.A12", h.A22.U12, h.A22.V12)
print("Razina 2 - blok A11.A21", h.A22.U21, h.A22.V21)

print("Razina 3 - blok A11.A11.A12", h.A11.A11.U12, h.A11.A11.V12)
print("Razina 3 - blok A11.A11.A21", h.A11.A11.U21, h.A11.A11.V21)
print("Razina 3- blok A11.A22.A12", h.A11.A22.U12, h.A11.A22.V12)
print("Razina 3 - blok A11.A11.A21", h.A11.A22.U21, h.A11.A22.V21)

print("Razina 3 - blok A22.A11.A12", h.A22.A11.U12, h.A22.A11.V12)
print("Razina 3 - blok A22.A11.A21", h.A22.A11.U21, h.A22.A11.V21)
print("Razina 3- blok A22.A22.A12", h.A22.A22.U12, h.A22.A22.V12)
print("Razina 3 - blok A22.A22.A21", h.A22.A22.U21, h.A22.A22.V21)