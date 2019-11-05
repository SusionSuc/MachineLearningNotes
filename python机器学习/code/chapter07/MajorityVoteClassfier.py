import numpy as np

# 加权大多数票概念

bincount = np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])
print(bincount)
print(np.argmax(bincount))
