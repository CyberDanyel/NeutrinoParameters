import matplotlib.pyplot as plt
import numpy as np
from Equations import NLL_Finder,find_energy_midpoints

energy_midpoints = find_energy_midpoints()

def quadratic(x):
    return x**2 + x + 1

def parabola_finder(domain,x0,x1,x2,y0,y1,y2):
    x = [x0,x1,x2]
    y = [y0,y1,y2]
    sum_value = 0
    for i in range(0,3):
        mult_value = 1
        for j in range(0,3):
            if j == i:
                pass
            else:
                mult_value = mult_value * ((domain-x[j])/(x[i]-x[j]))
        sum_value = sum_value + mult_value * y[i]
    return sum_value

x_vals = np.linspace(0,4,num=100000)

y_vals = parabola_finder(x_vals, 0.64, 0.6, 0.66, 52.748692257179876, 15.696160076908013, 75.42488356458956)

plt.plot(0.64,52.748692257179876,'x')
plt.plot(0.6,15.696160076908013,'x')
plt.plot(0.66,75.42488356458956,'x')

plt.plot(x_vals,y_vals)
plt.plot(x_vals, x_vals**2)
def parabolic_minimiser(function,x0,x1,x2):
    y0 = function(x0)
    y1 = function(x1)
    y2 = function(x2)
