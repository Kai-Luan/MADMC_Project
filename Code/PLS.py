import numpy as np
import matplotlib.pyplot as plt
from utils import *
from read_file import *
from indicators import *

## Loading data
numInstance=0
n= 200
p = 6
w=np.zeros(n,dtype=int) # poids des objets
v=np.zeros((n,p),dtype=int) # utilités des objets
filename = f"./data/2KP{n}-TA-{numInstance}.dat"
print(v.shape)
# W: budget
W=readFile(filename,w,v)

## Phase I: generate non-dominated solution with PLS
params = (n,p,v,w,W)
m = 1
verbose = True
NBMAX = 20

YND = PLS(m,params,NBMAX,verbose= True)