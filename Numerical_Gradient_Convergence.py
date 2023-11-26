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
fig, (ax1,ax2) = plt.subplots(2)
plt.tight_layout(pad = 4)
X, Y = np.meshgrid(thetas_l, diffs_l)
cp = ax1.contourf(X, Y, NLLs_l, 30)
ax1.set_title('Filled Contours Plot Left')
ax1.set_xlabel('thetas')
ax1.set_ylabel('diffs')
l = Gradient_Method_Numerical(2.1e-3,0.71,[1e-9,1e-4],'orange',1e-12,ax1)
l2 = Gradient_Method_Numerical(2.11e-3,0.72,[1e-9,1e-4],'green',1e-12,ax1)
l3 = Gradient_Method_Numerical(2.14e-3,0.715,[1e-9,1e-4],'purple',1e-12,ax1)
l4 = Gradient_Method_Numerical(2.13e-3,0.705,[1e-9,1e-4],'brown',1e-12,ax1)
ax1.plot(l[0][1],l[0][0],'X',color='white')
#plt.plot(0.71,2.1e-3,'o')

X, Y = np.meshgrid(thetas_r, diffs_r)
cp = ax2.contourf(X, Y, NLLs_r, 30)
ax2.set_title('Filled Contours Plot Right')
ax2.set_xlabel('thetas')
ax2.set_ylabel('diffs')
r = Gradient_Method_Numerical(2.1e-3,0.86,[1e-9,1e-4],'orange',1e-12,ax2)
r2 = Gradient_Method_Numerical(2.11e-3,0.85,[1e-9,1e-4],'green',1e-12,ax2)
r3 = Gradient_Method_Numerical(2.14e-3,0.855,[1e-9,1e-4],'purple',1e-12,ax2)
r4 = Gradient_Method_Numerical(2.13e-3,0.865,[1e-9,1e-4],'brown',1e-12,ax2)
ax2.plot(r[0][1],r[0][0],'X',color='white')
plt.savefig('Gradients')
plt.show()

print('For left minimum, theta = ' + str(l[0][1])+ '(+' + str(l[1][1][1]) +',-' + str(l[1][1][0]) +'), delm = ' + str(l[0][0]) + '(+' + str(l[1][0][1]) +',-' +str(l[1][0][0]) + ')')
print('For right minimum, theta = ' + str(r[0][1])+ '(+' + str(r[1][1][1]) +',-' + str(r[1][1][0]) +'), delm = ' + str(r[0][0]) + '(+' + str(r[1][0][1]) +',-' +str(r[1][0][0]) + ')')
