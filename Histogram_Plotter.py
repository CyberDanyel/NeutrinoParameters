import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from Equations import find_energy_midpoints,find_lambdas
#%%
events = np.array(pd.read_csv('Data.txt'))
events = events[:,0]
flux_pred = np.array(pd.read_csv('Unoscillated_Flux_Prediction.txt'))
flux_pred = flux_pred[:,0]
mixing_angle = np.pi/4
mixing_angle_left = 0.713350700427252
mixing_angle_right = 0.8576384528762316
delm = 2.1e-3
L = 295
energy_edges = np.linspace(0, 10, num = 201, endpoint = True)
energy_midpoints = find_energy_midpoints()

lambdas = find_lambdas(mixing_angle,delm,L,energy_midpoints)
#%%
xticks = np.arange(0,11,step = 1)
def Plot_Histograms():
    fig, (ax1,ax2,ax3) = plt.subplots(3, figsize = (8,10))
    plt.tight_layout(pad = 5)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid()
    ax1.hist(energy_edges[:-1], energy_edges, weights = events)
    ax1.set_title('Experimental Data', fontsize = 15)
    ax1.set_xlabel('Energy (GeV)', fontsize = 13)
    ax1.set_ylabel('Events', fontsize = 13)
    ax1.set_xlim([0,10])
    ax1.set_ylim([0,25])
    ax1.set_xticks(xticks)
    ax1.tick_params(direction='out', top=False, right=False, which='both', labelsize = 13)
    ax1.minorticks_on()
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.set_axisbelow(True)
    ax2.yaxis.grid()
    ax2.hist(energy_edges[:-1], energy_edges, weights = flux_pred)
    ax2.set_title('Unoscillated Event Rate Prediction', fontsize = 15)
    ax2.set_xlabel('Energy (GeV)', fontsize = 14)
    ax2.set_ylabel('Events', fontsize = 14)
    ax2.set_xlim([0,10])
    ax2.set_ylim([0,155])
    ax2.set_xticks(xticks)
    ax2.tick_params(direction='out', top=False, right=False, which='both', labelsize = 13)
    ax2.minorticks_on()
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.set_axisbelow(True)
    ax3.yaxis.grid()
    ax3.hist(energy_edges[:-1], energy_edges, weights = lambdas)
    ax3.set_title('Oscillated Event Rate Prediction ($θ_{23}=π/4$, $Δm^{2}_{23}$ = $2.1x10^{-3}$)', fontsize = 15)
    ax3.set_xlabel('Energy (GeV)', fontsize = 14)
    ax3.set_ylabel('Events', fontsize = 14)
    ax3.set_xlim([0,10])
    ax3.set_ylim([0,40])
    ax3.set_xticks(xticks)
    ax3.tick_params(direction='out', top=False, right=False, which='both', labelsize = 13)
    ax3.minorticks_on()
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.show()
    plt.savefig('Histograms', dpi = 800)
    
Plot_Histograms()