from Parabola_Minimiser import Gradient_Method_2D
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3,3, num = 500)
y = np.linspace(-3,3, num = 500)

vals = []
for i in range(len(x)):
    yvals = []
    for j in range(len(y)):
        valnew = x[i]**2 + y[j]**2
        yvals.append(valnew)
    vals.append(yvals)

vals = np.array(vals)

X, Y = np.meshgrid(x, y)

fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, vals, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

def diff_x_f(x):
    return 2*x[0]

def diff_y_f(x):
    return 2*x[1]

minimum = Gradient_Method_2D(-2,-2,diff_x_f,diff_y_f,0.1)

