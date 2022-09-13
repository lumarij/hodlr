from hodlr import *


block_min = 1
threshold = pow(10, -12)

def change_block_min(nmin):
   global block_min
   block_min = nmin
def change_treshold(tres):
   global threshold
   threshold = tres