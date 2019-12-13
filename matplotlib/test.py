import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 50)
y = 2 * x + 1

plt.xticks(np.linspace(0, 1, 2), ["tick1","tick2"])

plt.figure(num=1, figsize=(10, 5))
plt.xlim(-1, 1)
plt.ylim(0, 5)
plt.xlabel("XXXX")
plt.plot(x, y, color='blue', linewidth=1, linestyle='--',label='line1')
plt.legend(loc='upper right')
plt.show()
