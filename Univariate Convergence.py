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
l = Univariate_Minimisation(0.705, [0.705, 0.715, 0.725], [2.08e-3, 2.1e-3, 2.175e-3],'orange',ax1)
l1 = Univariate_Minimisation(0.72, [0.705, 0.715, 0.725], [2.08e-3, 2.1e-3, 2.175e-3],'blue',ax1)
delm_minimisee = Parabola_Minimiser([2.08e-3, 2.1e-3, 2.175e-3])
delm_minimised = delm_minimisee.minimise_delm(0.705)
#x0 = [delm_minimised,0.705]
#ax1.plot(x0[1],x0[0],'o')
ax1.plot(l[0][1],l[0][0],'X', label = 'Minimum')
ax1.legend(loc = 'upper right', fontsize = 7)
X, Y = np.meshgrid(thetas_r, diffs_r)
cp = ax2.contourf(X, Y, NLLs_r, 30)
ax2.set_title('Right Minimum ($θ_{23}^{r}$) Contour Plot', fontsize = 11)
ax2.set_xlabel('$θ_{23}$')
ax2.set_ylabel('$Δm^{2}_{23}$')
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
r = Univariate_Minimisation(0.865, [0.84, 0.86, 0.89], [2.05e-3, 2.1e-3, 2.175e-3],'orange',ax2)
r1 = Univariate_Minimisation(0.850, [0.84, 0.86, 0.89], [2.05e-3, 2.1e-3, 2.175e-3],'blue',ax2)
delm_minimisee = Parabola_Minimiser([2.06e-3, 2.1e-3, 2.175e-3])
delm_minimised = delm_minimisee.minimise_delm(0.865)
#x0 = [delm_minimised,0.865]
#ax2.plot(x0[1],x0[0],'o')
ax2.plot(r[0][1],r[0][0],'X', label = 'Minimum')
ax2.legend(loc = 'upper left', fontsize = 7)
fig.suptitle('Univariate Minimisation', fontsize = 15, fontweight = 'bold')
plt.savefig('Univariates', dpi = 800)
plt.show()

print('For left minimum, theta = ' + str(l[0][1])+ '(+' + str(l[1][1][1]) +',-' + str(l[1][1][0]) +'), delm = ' + str(l[0][0]) + '(+' + str(l[1][0][1]) +',-' +str(l[1][0][0]) + ')')
print('For right minimum, theta = ' + str(r[0][1])+ '(+' + str(r[1][1][1]) +',-' + str(r[1][1][0]) +'), delm = ' + str(r[0][0]) + '(+' + str(r[1][0][1]) +',-' +str(r[1][0][0]) + ')')

