import numpy as np
from Equations import Oscillation_Probability, find_energy_midpoints
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

mixing_angle = np.pi/4
delm = 2.4e-3
L = 295
energy_midpoints = find_energy_midpoints()
P = Oscillation_Probability(mixing_angle, delm, L, energy_midpoints)
#%% Plot P
fig, ax = plt.subplots()
ax.plot(energy_midpoints,P)
ax.set_title('P($ν_{µ}$-->$ν_{u}$) where $θ_{23}=π/4$, $Δm^{2}_{23}$ = $2.4*10^{-3}$, L = 295')
ax.set_xlabel('Energy (GeV)')
ax.set_ylabel('P($ν_{µ}$-->$ν_{µ})$')
ax.set_xlim([0,10])
ax.grid()
ax.tick_params(direction='out', top=False, right=False, which='both')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.show()

