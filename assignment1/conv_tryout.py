from EDF_Percpetron import *
import numpy as np


x = np.array([1,2,2,1,0,1,1,2,3,2,1,0,0,3,1,2,0,2]).reshape(-1,3,3,1)

a = Parameter(np.array([-0.5,1,-2,0.25,0,-1,2,0.123,2]).reshape(3,3,1,1))

b = Parameter(np.array([1]))

i = Input()

c = Conv([b,a],i)

i.value = x

c.forward()

c.value
