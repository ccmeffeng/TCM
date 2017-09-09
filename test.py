import numpy as np

x = np.array([[1,2,3,4,10],[2,4,5,6,12]])
for i in x:
    print(i[0: -1])
