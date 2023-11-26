import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from Parabola_Minimiser import Gradient_Method_Numerical_plane,Parabola_Minimiser, Univariate_Minimisation, Univariate_Minimisation_cs,Univariate_Minimisation_plane, Gradient_Method, Newton_Method,Gradient_Method_cs,Gradient_Method_Numerical,Gradient_Method_Numerical_cs
from Equations import find_energy_midpoints, NLL_Finder, NLLs_Finder, find_lambdas, NLL_Finder_cs, NLLs_Finder_cs
from matplotlib.ticker import AutoMinorLocator

energy_midpoints = find_energy_midpoints()
#%%
c = np.linspace(0.5,2,num=50)
NLLs = []
for i in range(len(c)):
    print(i+1)
    minim = Gradient_Method_Numerical_plane(0.002,1,c[i],[1e-9,1e-5],'orange',1e-12)
    #minim = Gradient_Method_plane(0.0021,1,c[i],[1e-9,1e-5],'orange',ax)
    #minim = Univariate_Minimisation_plane(0.90, [0.80, 0.86, 1], [2.06e-3, 2.1e-3, 2.175e-3],c[i])
    NLLs.append(NLL_Finder_cs(minim[0][1], minim[0][0],c[i],295,energy_midpoints))
    
#%%   
minim_left = Gradient_Method_Numerical_plane(0.002,0.5,1.08,[1e-9,1e-5],'orange',1e-12)
minim_right = Gradient_Method_Numerical_plane(0.002,1,1.08,[1e-9,1e-5],'orange',1e-12)
print(minim_left)
print(minim_right)
#%%
fig, ax = plt.subplots()

ax.plot(c,NLLs)
ax.set_title('Minimum NLL against α',fontsize = 15)
ax.set_xlabel('α',fontsize = 13)
ax.set_ylabel('Minimum NLL',fontsize = 13)
ax.grid()
ax.set_xlim([0.5,2])
ax.set_ylim([-320,-200])
ax.set_xticks([0.5,0.75,1,1.25,1.5,1.75,2])
ax.tick_params(direction='in', top=True, right=True, which='both', labelsize = 10)
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.savefig('NLL against alpha', dpi = 800)