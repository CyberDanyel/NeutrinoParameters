from Equations import NLL_Finder, NLL_Finder_cs, find_energy_midpoints

energy_midpoints = find_energy_midpoints()

def Secant_Root_Finder_for_y_is_x_squared(x):
    def f(x):
        return x**2 - 1
    while True:
        x_new = x[1] - (f(x[1]))*((x[1]-x[0])/(f(x[1])-f(x[0])))
        if abs(((x_new-x[1])/x[1])) < 1e-13:
            return x_new
        else:
            x[0] = x[1]
            x[1] = x_new
        
def Secant_Root_Finder_theta(x,minimum,delm):
    def f(x):
        return NLL_Finder(x, delm, 295, energy_midpoints) - minimum - 0.5
    while True:
        x_new = x[1] - (f(x[1]))*((x[1]-x[0])/(f(x[1])-f(x[0])))
        if abs(((x_new-x[1])/x[1])) < 1e-6:
            return x_new
        else:
            x[0] = x[1]
            x[1] = x_new

def Secant_Root_Finder_delm(x,minimum,theta):
    def f(x):
        return NLL_Finder(theta, x, 295, energy_midpoints) - minimum - 0.5
    while True:
        x_new = x[1] - (f(x[1]))*((x[1]-x[0])/(f(x[1])-f(x[0])))
        if abs(((x_new-x[1])/x[1])) < 1e-6:
            return x_new
        else:
            x[0] = x[1]
            x[1] = x_new

def Secant_Root_Finder_theta_cs(x,minimum,delm,cs):
    def f(x):
        return NLL_Finder_cs(x, delm, cs, 295, energy_midpoints) - minimum - 0.5
    while True:
        x_new = x[1] - (f(x[1]))*((x[1]-x[0])/(f(x[1])-f(x[0])))
        if abs(((x_new-x[1])/x[1])) < 1e-6:
            return x_new
        else:
            x[0] = x[1]
            x[1] = x_new

def Secant_Root_Finder_delm_cs(x,minimum,theta,cs):
    def f(x):
        return NLL_Finder_cs(theta, x, cs, 295, energy_midpoints) - minimum - 0.5
    while True:
        x_new = x[1] - (f(x[1]))*((x[1]-x[0])/(f(x[1])-f(x[0])))
        if abs(((x_new-x[1])/x[1])) < 1e-6:
            return x_new
        else:
            x[0] = x[1]
            x[1] = x_new
            
def Secant_Root_Finder_cs_cs(x,minimum,theta,delm):
    def f(x):
        return NLL_Finder_cs(theta, delm, x, 295, energy_midpoints) - minimum - 0.5
    while True:
        x_new = x[1] - (f(x[1]))*((x[1]-x[0])/(f(x[1])-f(x[0])))
        if abs(((x_new-x[1])/x[1])) < 1e-6:
            return x_new
        else:
            x[0] = x[1]
            x[1] = x_new