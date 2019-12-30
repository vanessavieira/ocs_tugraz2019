#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%

# generate 2D grid
y,x = np.mgrid[-5:5:0.1, -5:5:0.1]

#%%
f1 = (2 * x ** 3) - (6 * y ** 2) + (3 * (x ** 2) * y)
f2 = ((x - 2 * y) ** 4) + (64 * x * y)
f3 = (2 * x ** 2) + (3 * y ** 2) - (2 * x * y) + (2 * x) - (3 * y)
f4 = np.log(1 + (1 / 2) * (x ** 2 + y ** 2))

# compute objective function


#%%
plt.figure(1)

# 30 contour lines of f
plt.contour(x, y, f4, 20)

# plot global maximum
#plt.plot(1.0/np.sqrt(2),1.0/np.sqrt(2),"*", color="red", markersize=15, label=r'$x_1^*$')

# plot global minimum
#plt.plot(-1.0/np.sqrt(2),-1.0/np.sqrt(2),"*", color="blue", markersize=15, label=r'$x_2^*$')

# add legend
plt.legend(loc="lower left")

plt.show()


#%%

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, f4, cmap=plt.cm.jet)

plt.show()