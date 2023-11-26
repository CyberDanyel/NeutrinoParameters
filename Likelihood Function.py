import numpy as np
from Equations import NLLs_Finder, NLL_Finder,find_energy_midpoints
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from Parabola_Minimiser import Parabola_Minimiser

energy_midpoints = find_energy_midpoints()

number = 2000

thetas = np.linspace(0+0.65,(np.pi/2)-0.65,num = number,endpoint = True)
NLLs = NLLs_Finder(thetas, [2.1e-3])

minimisee_l = Parabola_Minimiser([0.71,0.715,0.72])
minimisee_r = Parabola_Minimiser([0.84,0.85,0.86])
min_l = minimisee_l.minimise_theta(2.1e-3)
min_r = minimisee_r.minimise_theta(2.1e-3)
errors_l = minimisee_l.secant_theta(2.1e-3)
errors_r = minimisee_r.secant_theta(2.1e-3)
parabolic_error_l = minimisee_l.parabolic_error()
parabolic_error_r = minimisee_r.parabolic_error()

fig, ax = plt.subplots()
ax.plot(thetas/np.pi,NLLs[0,:],linewidth = 2.5)
ax.plot(min_l/np.pi,NLL_Finder(min_l, 2.1e-3, 295, energy_midpoints),'X', color = '#176D44', label = '$θ_{23}^{l}$')
ax.plot(min_r/np.pi,NLL_Finder(min_r, 2.1e-3, 295, energy_midpoints),'X', color = '#960D17', label = '$θ_{23}^{r}$')
ax.legend(loc = 'upper center', fontsize = 13)
ax.set_title('NLL as a Function of $θ_{23}$ ($Δm^{2}_{23}$ = $2.1x10^{-3}$)', fontsize = 16)
ax.set_xlabel('$θ_{23}/π$', fontsize = 14)
ax.set_ylabel('NLL', fontsize = 14)
ax.set_xlim([thetas[0]/np.pi,thetas[number-1]/np.pi])
ax.set_ylim([-68,-58])
ax.grid()
ax.tick_params(direction='out', top=False, right=False, which='both', labelsize = 13)
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.savefig('NLL against theta', dpi = 800)
plt.show()


'''
thetass = np.linspace(0,np.pi/2,num = number,endpoint = True)
NLLss = NLLs_Finder(thetass, [2.1e-3])
plt.figure()
plt.plot(thetass,NLLss[0,:])
plt.show()
'''