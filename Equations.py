import pandas as pd
import numpy as np


energy_edges = np.linspace(0, 10, num = 201, endpoint = True)

#Using the midpoints of the bins for lambda
energy_midpoints = []
for i in range(len(energy_edges)-1):
    new_midpoint = (energy_edges[i+1] - energy_edges[i])/2 + energy_edges[i]
    energy_midpoints.append(new_midpoint)
    
energy_midpoints = np.array(energy_midpoints)


events = np.array(pd.read_csv('Data.txt'))
events = events[:,0]
flux_pred = np.array(pd.read_csv('Unoscillated_Flux_Prediction.txt'))
flux_pred = flux_pred[:,0]

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def Oscillation_Probability(mixing_angle,delm,L,E):
    P = 1 - ((np.sin(2*mixing_angle))**2)*((np.sin((1.267*delm*L)/E))**2)
    return P

def find_energy_midpoints():
    energy_edges = np.linspace(0, 10, num = 201, endpoint = True)

    #Using the midpoints of the bins for lambda
    energy_midpoints = []
    for i in range(len(energy_edges)-1):
        new_midpoint = (energy_edges[i+1] - energy_edges[i])/2 + energy_edges[i]
        energy_midpoints.append(new_midpoint)
        
    energy_midpoints = np.array(energy_midpoints)
    return energy_midpoints

#%% No cross-section
def find_lambdas(mixing_angle,delm,L,E):
    lambdas = flux_pred * Oscillation_Probability(mixing_angle,delm,L,E)
    return lambdas

def lambdas_delm_derivative(mixing_angle,delm,L,E):
    lambdas = flux_pred * -(1267*L*((np.sin(2*mixing_angle))**2)*np.cos((1267*L*delm)/(1000*E))*np.sin((1267*L*delm)/(1000*E)))/(500*E)
    return lambdas

def lambdas_theta_derivative(mixing_angle,delm,L,E):
    lambdas = flux_pred * -4*((np.sin((1267*L*delm)/(1000*E)))**2)*np.cos(2*mixing_angle)*np.sin(2*mixing_angle)
    return lambdas

def ln_lambdas_delm_derivative(mixing_angle,delm,L,E):
    lambdas = (1267*L*((np.sin(2*mixing_angle))**2)*(np.cos((1267*L*delm)/(1000*E)))*(np.sin((1267*L*delm)/(1000*E))))/(500*E*(((np.sin(2*mixing_angle))**2)*((np.sin((1267*L*delm)/(1000*E)))**2)-(1)))
    return lambdas
    
def ln_lambdas_theta_derivative(mixing_angle,delm,L,E):
    lambdas = (4*((np.sin((1267*L*delm)/(1000*E)))**2)*(np.cos(2*mixing_angle))*(np.sin(2*mixing_angle)))/(((np.sin((1267*L*delm)/(1000*E)))**2)*((np.sin(2*mixing_angle))**2)-1)
    return lambdas

def NLL_Finder(mixing_angle,delm,L,E):
    val = 0
    lambdas = find_lambdas(mixing_angle,delm,L,E)
    for i in range(len(energy_midpoints)):
        val = val + lambdas[i] - events[i]*np.log(lambdas[i])
    return val

def NLL_delm_derivative_finder(mixing_angle,delm,L,E):
    val = 0
    lambdas_derivative = lambdas_delm_derivative(mixing_angle,delm,L,E)
    ln_lambdas_derivative = ln_lambdas_delm_derivative(mixing_angle,delm,L,E)
    for i in range(len(energy_midpoints)):
        val = val + lambdas_derivative[i] - events[i]*ln_lambdas_derivative[i]
    return val

def NLL_theta_derivative_finder(mixing_angle,delm,L,E):
    val = 0
    lambdas_derivative = lambdas_theta_derivative(mixing_angle,delm,L,E)
    ln_lambdas_derivative = ln_lambdas_theta_derivative(mixing_angle,delm,L,E)
    for i in range(len(energy_midpoints)):
        val = val + lambdas_derivative[i] - events[i]*ln_lambdas_derivative[i]
    return val

def NLLs_Finder(thetas,m):
    NLLs = []
    for j in range(len(m)):
        print(j+1)
        thetaNLLs = []
        for i in range(len(thetas)):
            newnll = NLL_Finder(thetas[i],m[j],295,energy_midpoints)
            thetaNLLs.append(newnll)
        NLLs.append(thetaNLLs)
    return np.array(NLLs)

#%%
def find_lambdas_cs(mixing_angle,delm,cs,L,E):
    lambdas = find_lambdas(mixing_angle,delm,L,E) * cs * E
    return lambdas

def lambdas_delm_derivative_cs(mixing_angle,delm,cs,L,E):
    lambdas = lambdas_delm_derivative(mixing_angle, delm, L, E) * cs * E
    return lambdas

def lambdas_theta_derivative_cs(mixing_angle,delm,cs,L,E):
    lambdas = lambdas_theta_derivative(mixing_angle, delm, L, E) * cs * E
    return lambdas

def NLL_delm_derivative_finder_cs(mixing_angle,delm,cs,L,E):
    val = 0
    lambdas_derivative = lambdas_delm_derivative_cs(mixing_angle,delm,cs,L,E)
    ln_lambdas_derivative = ln_lambdas_delm_derivative(mixing_angle,delm,L,E)
    for i in range(len(energy_midpoints)):
        val = val + lambdas_derivative[i] - events[i]*ln_lambdas_derivative[i]
    return val

def NLL_theta_derivative_finder_cs(mixing_angle,delm,cs,L,E):
    val = 0
    lambdas_derivative = lambdas_theta_derivative_cs(mixing_angle,delm,cs,L,E)
    ln_lambdas_derivative = ln_lambdas_theta_derivative(mixing_angle,delm,L,E)
    for i in range(len(energy_midpoints)):
        val = val + lambdas_derivative[i] - events[i]*ln_lambdas_derivative[i]
    return val

def NLL_cs_derivative_finder_cs(mixing_angle,delm,cs,L,E):
    val = 0
    old_lambdas = find_lambdas(mixing_angle, delm, L, E)
    for i in range(len(energy_midpoints)):
        val = val + old_lambdas[i]*E[i] - events[i]*(1/cs)
    return val

def NLL_Finder_cs(mixing_angle,delm,cs,L,E):
    val = 0
    lambdas = find_lambdas_cs(mixing_angle,delm,cs,L,E)
    for i in range(len(energy_midpoints)):
        val = val + lambdas[i] - events[i]*np.log(lambdas[i])
    return val

def NLLs_Finder_cs(thetas,m,cs):
    NLLs = []
    for j in range(len(m)):
        print(j+1)
        thetaNLLs = []
        for i in range(len(thetas)):
            newnll = NLL_Finder_cs(thetas[i],m[j],cs,295,energy_midpoints)
            thetaNLLs.append(newnll)
        NLLs.append(thetaNLLs)
    return np.array(NLLs)
#%%
def linear_interpolation(exes,x,y):
    fs = ((x[1]-exes)*y[0]+(exes-x[0])*y[1])/(x[1]-x[0])
    return fs
#%%
def Inverted_Hessian(mixing_angle, delm, L, E,h,N):
    Hessian = np.zeros((2,2))
    #df_dm = (NLL_Finder(mixing_angle, delm+h1, L, E)-NLL_Finder(mixing_angle, delm-h1, L, E))/(2*h)
    #df_dtheta = (NLL_Finder(mixing_angle+h2, delm, L, E)-NLL_Finder(mixing_angle-h2, delm, L, E))/(2*h)
    df_dm2 = (NLL_Finder(mixing_angle, delm+h, L, E)-2*NLL_Finder(mixing_angle, delm, L, E)+NLL_Finder(mixing_angle, delm-h, L, E))/(h**2)
    df_theta2 = (NLL_Finder(mixing_angle+h, delm, L, E)-2*NLL_Finder(mixing_angle, delm, L, E)+NLL_Finder(mixing_angle-h, delm, L, E))/(h**2)
    df_dm_dtheta = (NLL_Finder(mixing_angle-h, delm-h, L, E)+NLL_Finder(mixing_angle+h, delm+h, L, E)-NLL_Finder(mixing_angle+h, delm-h, L, E)-NLL_Finder(mixing_angle-h, delm+h, L, E))/(4*(h**2))
    Hessian[0,0] = df_dm2
    Hessian[0,1] = df_dm_dtheta
    Hessian[1,0] = df_dm_dtheta
    Hessian[1,1] = df_theta2
    
    L = np.zeros((N,N))
    U = np.zeros((N,N))

    for i in range(N):
        L[i,i] = 1
    for j in range(N):
        i = 0
        while i <= j:
            U[i,j] = Hessian[i,j]
            for k in range(0,i):
                U[i,j] -= L[i,k]*U[k,j]
            i += 1
        i = N-1
        while i > j:
            L[i,j] = Hessian[i,j]
            for k in range(0,j):
                L[i,j] -= L[i,k]*U[k,j]
            L[i,j] = L[i,j]/U[j,j]
            i -= 1

    if np.matmul(L, U).all() == Hessian.all():
        print('W')
    else:
        print('L')
        
    def sigma_special(Matrix,i,j,k,Vector): #For multiple x
        return Matrix[i,j]*Vector[j,k]

    X = np.zeros((N,N))
    Y = np.zeros((N,N))
    B = np.identity(N)
    for k in range(N):
        
        Y[0,k] = B[0,k]/L[0,0]
        
        for i in range(1,N):
            Y[i,k] = (B[i,k] - sum(sigma_special(L,i,j,k,Y) for j in range(0, i)))/L[i,i]
            
        X[N-1,k] = Y[N-1,k]/U[N-1,N-1]   
        
        for i in range(N-2,-1,-1):
            X[i,k] = (Y[i,k] - sum(sigma_special(U,i,j,k,X) for j in range(i+1, N)))/U[i,i]
            
        if np.matmul(U, X).all() == Y.all():
            print('W')
        else:
            print('L')
            
    Hessian_inverted = X

    if np.matmul(Hessian, X).all() == B.all():
        print('W')
    else:
        print('L')
    if np.matmul(Hessian_inverted, Hessian).all() == np.identity(N).all():
        print('W')
    else:
        print('L')
        
    return Hessian_inverted

def Create_Hessian(mixing_angle, delm, L, E,h,N):
    Hessian = np.zeros((2,2))
    #df_dm = (NLL_Finder(mixing_angle, delm+h1, L, E)-NLL_Finder(mixing_angle, delm-h1, L, E))/(2*h)
    #df_dtheta = (NLL_Finder(mixing_angle+h2, delm, L, E)-NLL_Finder(mixing_angle-h2, delm, L, E))/(2*h)
    df_dm2 = (NLL_Finder(mixing_angle, delm+h, L, E)-2*NLL_Finder(mixing_angle, delm, L, E)+NLL_Finder(mixing_angle, delm-h, L, E))/(h**2)
    df_theta2 = (NLL_Finder(mixing_angle+h, delm, L, E)-2*NLL_Finder(mixing_angle, delm, L, E)+NLL_Finder(mixing_angle-h, delm, L, E))/(h**2)
    df_dm_dtheta = (NLL_Finder(mixing_angle-h, delm-h, L, E)+NLL_Finder(mixing_angle+h, delm+h, L, E)-NLL_Finder(mixing_angle+h, delm-h, L, E)-NLL_Finder(mixing_angle-h, delm+h, L, E))/(4*(h**2))
    Hessian[0,0] = df_dm2
    Hessian[0,1] = df_dm_dtheta
    Hessian[1,0] = df_dm_dtheta
    Hessian[1,1] = df_theta2
    
    return Hessian
    

    