import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

###################################################################
#Meshgrid creates 20 evenly spaced points of x
#Replicates x 20 times for each y -> x_repl
#Replicates each y 20 times and does it for each y -> y_repl
###################################################################

points = np.linspace(-10,10,20)
x_repl, y_repl = np.meshgrid(points, points)

fig = plt.figure()
ax = fig.gca(projection='3d') 
ax.plot_trisurf(x_repl.flatten(), y_repl.flatten(), y_repl.flatten())
plt.show()
