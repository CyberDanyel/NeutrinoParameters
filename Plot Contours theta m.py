import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from Parabola_Minimiser import Parabola_Minimiser, Univariate_Minimisation, Univariate_Minimisation_cs,Univariate_Minimisation_plane, Gradient_Method, Newton_Method,Gradient_Method_cs,Gradient_Method_Numerical,Gradient_Method_Numerical_cs
from Equations import NLL_Finder, NLLs_Finder, find_lambdas, NLL_Finder_cs, NLLs_Finder_cs

thetas1 = np.linspace(0+0.65,(np.pi/2)-0.65,num = 200,endpoint = True)
diffs1 = np.linspace(1.9e-3,2.3e-3, num = 200, endpoint = True)
#diffs are in rows, thetas are in columns
NLLs = NLLs_Finder(thetas1,diffs1)

#%%

levels1 = np.arange(start = -70, stop = -68,step = 0.1)
levels2 = np.arange(start = -68, stop = 2,step = 2)

levels = np.concatenate((levels1,levels2))
X, Y = np.meshgrid(thetas1/np.pi, diffs1)
fig,ax=plt.subplots()
cp = ax.contourf(X, Y, NLLs,levels,cmap = 'viridis')
cbar = fig.colorbar(cp) # Add a colorbar to a plot
cbar.ax.tick_params(labelsize=12) 
ax.set_title('NLL as a Function of $θ_{23}$ and $Δm^{2}_{23}$', fontsize = 13)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
ax.tick_params(direction='in', top=True, right=True, which='both', labelsize = 12)
ax.set_xticks([0.21,0.23,0.25,0.27,0.29])
ax.set_yticks([1.95e-3,2e-3,2.05e-3,2.1e-3,2.15e-3,2.2e-3,2.25e-3])
ax.set_xlabel('$θ_{23}$/π', fontsize = 13)
ax.set_ylabel('$Δm_{23}^{2}$', fontsize = 13)
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.savefig('contours', dpi = 800)
plt.show()
