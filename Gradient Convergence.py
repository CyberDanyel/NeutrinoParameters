import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from Parabola_Minimiser import Parabola_Minimiser, Univariate_Minimisation, Univariate_Minimisation_cs,Univariate_Minimisation_plane, Gradient_Method, Newton_Method,Gradient_Method_cs,Gradient_Method_Numerical,Gradient_Method_Numerical_cs
from Equations import NLL_Finder, NLLs_Finder, find_lambdas, NLL_Finder_cs, NLLs_Finder_cs

#%% For Left
#%% Calculation
thetas_l = np.linspace(0.70,0.73, num = 100, endpoint = True)
diffs_l = np.linspace(2.09e-3,2.15e-3, num = 100, endpoint = True)
#diffs are in rows, thetas are in columns
NLLs_l = NLLs_Finder(thetas_l,diffs_l)
#%% For Right
#%% Calculation
thetas_r = np.linspace(0.84,0.87, num = 100, endpoint = True)
diffs_r = np.linspace(2.09e-3,2.15e-3, num = 100, endpoint = True)
#diffs are in rows, thetas are in columns
NLLs_r = NLLs_Finder(thetas_r,diffs_r)
#%% Plotting
fig, (ax1,ax2) = plt.subplots(2,figsize = (5,6))
plt.tight_layout(pad = 4)
X, Y = np.meshgrid(thetas_l, diffs_l)
cp = ax1.contourf(X, Y, NLLs_l, 30)
ax1.set_title('Left Minimum ($θ_{23}^{l}$) Contour Plot', fontsize = 11)
ax1.set_xlabel('$θ_{23}$')
ax1.set_ylabel('$Δm^{2}_{23}$')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
l = Gradient_Method(2.1e-3,0.71,[1e-9,1e-4],'orange',ax1)
l2 = Gradient_Method(2.11e-3,0.72,[1e-9,1e-4],'green',ax1)
l3 = Gradient_Method(2.14e-3,0.715,[1e-9,1e-4],'blue',ax1)
l4 = Gradient_Method(2.13e-3,0.705,[1e-9,1e-4],'brown',ax1)
ax1.plot(l[0][1],l[0][0],'X', label = 'Minimum')
ax1.legend(loc = 'upper right', fontsize = 7)
#plt.plot(0.71,2.1e-3,'o')

X, Y = np.meshgrid(thetas_r, diffs_r)
cp = ax2.contourf(X, Y, NLLs_r, 30)
ax2.set_title('Right Minimum ($θ_{23}^{r}$) Contour Plot', fontsize = 11)
ax2.set_xlabel('$θ_{23}$')
ax2.set_ylabel('$Δm^{2}_{23}$')
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
r = Gradient_Method(2.1e-3,0.86,[1e-9,1e-4],'orange',ax2)
r2 = Gradient_Method(2.11e-3,0.85,[1e-9,1e-4],'green',ax2)
r3 = Gradient_Method(2.14e-3,0.855,[1e-9,1e-4],'blue',ax2)
r4 = Gradient_Method(2.13e-3,0.865,[1e-9,1e-4],'brown',ax2)
ax2.plot(r[0][1],r[0][0],'X', label = 'Minimum')
ax2.legend(loc = 'upper left', fontsize = 7)
fig.suptitle('Gradient Method', fontsize = 15, fontweight = 'bold')
plt.savefig('Gradients', dpi = 800)
plt.show()

print('For left minimum, theta = ' + str(l[0][1])+ '(+' + str(l[1][1][1]) +',-' + str(l[1][1][0]) +'), delm = ' + str(l[0][0]) + '(+' + str(l[1][0][1]) +',-' +str(l[1][0][0]) + ')')
print('For right minimum, theta = ' + str(r[0][1])+ '(+' + str(r[1][1][1]) +',-' + str(r[1][1][0]) +'), delm = ' + str(r[0][0]) + '(+' + str(r[1][0][1]) +',-' +str(r[1][0][0]) + ')')
