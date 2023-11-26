import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from Parabola_Minimiser import Gradient_Method_Numerical_plane,Gradient_Method_plane,Parabola_Minimiser, Univariate_Minimisation, Univariate_Minimisation_cs,Univariate_Minimisation_plane, Gradient_Method, Newton_Method,Gradient_Method_cs,Gradient_Method_Numerical,Gradient_Method_Numerical_cs
from Equations import NLL_Finder, NLLs_Finder, find_lambdas, NLL_Finder_cs, NLLs_Finder_cs
#%%
events = np.array(pd.read_csv('Data.txt'))
events = events[:,0]
flux_pred = np.array(pd.read_csv('Unoscillated_Flux_Prediction.txt'))
flux_pred = flux_pred[:,0]

energy_edges = np.linspace(0, 10, num = 201, endpoint = True)

#Using the midpoints of the bins for lambda
energy_midpoints = []
for i in range(len(energy_edges)-1):
    new_midpoint = (energy_edges[i+1] - energy_edges[i])/2 + energy_edges[i]
    energy_midpoints.append(new_midpoint)
    
energy_midpoints = np.array(energy_midpoints)

def Plot_Histograms():
    plt.hist(energy_edges[:-1], energy_edges, weights = events)
    plt.figure()
    plt.stairs(events, edges = energy_edges)
    plt.figure()
    plt.stairs(flux_pred, edges = energy_edges)
#%%
thetas = np.linspace(0.5,1.2, num = 100, endpoint = True)
diffs = np.linspace(1.6e-3,2.4e-3, num = 100, endpoint = True)
#diffs are in rows, thetas are in columns
NLLs = NLLs_Finder(thetas,diffs)

X, Y = np.meshgrid(thetas, diffs)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs, 30)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
plt.show()
#%% For Left
#%% Calculation
thetas_l = np.linspace(0.69,0.74, num = 100, endpoint = True)
diffs_l = np.linspace(2.04e-3,2.18e-3, num = 100, endpoint = True)
#diffs are in rows, thetas are in columns
NLLs_l = NLLs_Finder(thetas_l,diffs_l)
#%% Plotting
l = Univariate_Minimisation(0.6, [0.65, 0.73, 0.74], [2.05e-3, 2.1e-3, 2.175e-3])
X, Y = np.meshgrid(thetas_l, diffs_l)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs_l, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot Left')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
plt.plot(l[0][1],l[0][0],'X')
plt.show()

#%% For Right
#%% Calculation
thetas_r = np.linspace(0.8,0.9, num = 100, endpoint = True)
diffs_r = np.linspace(0.0020,0.0022, num = 100, endpoint = True)
#diffs are in rows, thetas are in columns
NLLs_r = NLLs_Finder(thetas_r,diffs_r)
#%% Plotting
r = Univariate_Minimisation(0.8, [0.84, 0.86, 0.89], [2.06e-3, 2.1e-3, 2.175e-3])
X, Y = np.meshgrid(thetas_r, diffs_r)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs_r, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot Right')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
plt.plot(r[0][1],r[0][0],'X')
plt.show()

#%%
X, Y = np.meshgrid(thetas_r, diffs_r)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs_r, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot Right')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
plt.plot(r[1],r[0],'X')

X, Y = np.meshgrid(thetas_l, diffs_l)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs_l, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot Left')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
plt.plot(l[1],l[0],'X')
#%% Comparing analytical and numerical gradient methods
X, Y = np.meshgrid(thetas_r, diffs_r)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs_r, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot Right')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
gradient_min1 = Gradient_Method(0.002,0.85,[1e-9,1e-4], 'red')
numerical_gradient1 = Gradient_Method_Numerical(0.0023,0.85,[1e-9,1e-4], 'green',1e-12)
plt.plot(gradient_min1[0][1],gradient_min1[0][0],'X')
plt.plot(numerical_gradient1[0][1],numerical_gradient1[0][0],'X')
print(gradient_min1)
print(numerical_gradient1)
plt.show()
#%%
gradient_min2 = Gradient_Method(0.003125116047106075,0.9,[1e-9,1e-4],'orange')
gradient_min3 = Gradient_Method(4e-3,1,[1e-9,1e-4],'yellow')
gradient_min4 = Gradient_Method(0.002125116047106075,0.6,[1e-9,1e-4],'green')
gradient_min5 = Gradient_Method(4e-3,0.4,[1e-9,1e-4],'blue')

print(gradient_min1)
print(gradient_min2)
print(gradient_min3)
print(gradient_min4)
print(gradient_min5)

#%%
X, Y = np.meshgrid(thetas_r, diffs_r)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs_r, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot Right')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
plt.plot(r[1],r[0],'X')
#%%
X, Y = np.meshgrid(thetas_r, diffs_r)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs_r, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot Right')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
plt.plot(r[1],r[0],'X')
#%%
gradienthessian_min2 = Newton_Method(0.003,1,'red',1e-12,2)
#%% NLLs with defined cs
thetas = np.linspace(0,np.pi/2, num = 100)
diffs = np.linspace(0.0002,0.0040, num = 100)

NLLs = NLLs_Finder_cs(thetas,diffs,0.5)
X, Y = np.meshgrid(thetas, diffs)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot cs = 0.5')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')

NLLs = NLLs_Finder_cs(thetas,diffs,0.75)
X, Y = np.meshgrid(thetas, diffs)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot cs = 0.75')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')

NLLs = NLLs_Finder_cs(thetas,diffs,1)
X, Y = np.meshgrid(thetas, diffs)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot cs = 1')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')

NLLs = NLLs_Finder_cs(thetas,diffs,1.25)
X, Y = np.meshgrid(thetas, diffs)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot cs = 1.25')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')

NLLs = NLLs_Finder_cs(thetas,diffs,1.5)
X, Y = np.meshgrid(thetas, diffs)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot cs = 1.5')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
#%%
alpha = [1e-9,1e-4,1e-6]
minNLLs = []
mins = []
css = np.linspace(1,1.1,num = 10)
for i in range(len(css)):
    print(i+1)
    x = Gradient_Method_cs(0.0021,0.8,css[i],alpha,'red')
    mins.append(x)
    minNLLs.append(NLL_Finder_cs(x[0][1], x[0][0], x[0][2], 295, energy_midpoints))
compare = np.array([css,minNLLs])

plt.figure()
plt.plot(compare[0,:],compare[1,:])
#%%
alpha = [1e-9,1e-4,1e-6]
minNLLs = []
mins = []
css = np.linspace(0.01,30,num = 200)
for i in range(len(css)):
    print(i+1)
    x = Gradient_Method_cs(0.00217211,0.654461,css[i],alpha,'red')
    mins.append(x)
    minNLLs.append(NLL_Finder_cs(x[1], x[0], x[2], 295, energy_midpoints))
compare = np.array([css,minNLLs])

plt.figure()
plt.plot(compare[0,:],compare[1,:])
#%%
css = np.linspace(0.5,1.5, num = 200)
NLLs = []
for i in range(len(css)):
    NLLs.append(NLL_Finder_cs(0.916318486560335, 0.0021721051807371324,css[i],295,energy_midpoints))
plt.figure()
plt.plot(css,NLLs)
#%%
alpha = [1e-9,1e-4,1e-6]
gradient_min1 = Gradient_Method_cs(0.0020,0.6,1.08,alpha, 'red')
#%%
gradient_min1 = Gradient_Method_cs(0.0020,0.6,1.08,alpha,'orange')
gradient_min2 = Gradient_Method_cs(0.0021,0.8,1.08,alpha,'orange')

num_gradient_min1 = Gradient_Method_Numerical_cs(0.0020,0.6,1.08,alpha,'orange',1e-12)
num_gradient_min2 = Gradient_Method_Numerical_cs(0.0021,0.8,1.08,alpha,'orange',1e-12)

#%% Trying out cs univariate

minimised1 = Univariate_Minimisation_cs(0.65, 1, [0.5, 0.57, 0.74], [2.06e-3, 2.1e-3, 2.175e-3], [0.9,1,1.1])

minimised2 = Univariate_Minimisation_cs(0.9, 1.08, [0.84, 0.86, 0.89], [2.06e-3, 2.1e-3, 2.175e-3], [0.9,1,1.2])

#%%Making the plane at cs = 1.087
thetas = np.linspace(0,np.pi/2, num = 100)
diffs = np.linspace(0.0002,0.0040, num = 100)
cs = 1.0870132844709115
NLLs = NLLs_Finder_cs(thetas,diffs,cs)
X, Y = np.meshgrid(thetas, diffs)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs, 1000)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot cs = 1.5')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')
l = Univariate_Minimisation_plane(0.6, [0.4, 0.5, 0.6], [2.06e-3, 2.1e-3, 2.175e-3],cs)
r = Univariate_Minimisation_plane(0.85, [0.85, 0.86, 1], [2.06e-3, 2.1e-3, 2.175e-3],cs)
plt.plot(l[1],l[0],'X')
plt.plot(r[1],r[0],'X')
#%%At different planes:
c = np.linspace(0.5,2,num=50)
NLLs = []
for i in range(len(c)):
    print(i+1)
    minim = Gradient_Method_Numerical_plane(0.002,1,c[i],[1e-9,1e-5],'orange',1e-12,ax)
    #minim = Gradient_Method_plane(0.0021,1,c[i],[1e-9,1e-5],'orange',ax)
    #minim = Univariate_Minimisation_plane(0.90, [0.80, 0.86, 1], [2.06e-3, 2.1e-3, 2.175e-3],c[i])
    NLLs.append(NLL_Finder_cs(minim[0][1], minim[0][0],c[i],295,energy_midpoints))
plt.figure()
plt.plot(c,NLLs)
#%%
thetas = np.linspace(0,np.pi/2, num = 50)
diffs = np.linspace(0.0002,0.0040, num = 50)
#%%
NLLs = NLLs_Finder_cs(thetas,diffs,1.08)
X, Y = np.meshgrid(thetas, diffs)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, NLLs, 100)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot cs = 0.5')
ax.set_xlabel('thetas')
ax.set_ylabel('diffs')

#a = Gradient_Method_Numerical_plane(0.002,1,0.3,[1e-9,1e-4],'orange',1e-12,ax)
#ax.plot(a[0][1],a[0][0],'X')
#%%
thetas = np.linspace(0,np.pi/2, num = 50)
diffs = np.linspace(0.0002,0.0040, num = 50)
c = np.linspace(0.5,5,num=50)
NLLss = []
for i in range(len(c)):
    NLLs = NLLs_Finder_cs(thetas,diffs,c[i])
    NLLss.append()
#%%
'''
gradient_min3 = Gradient_Method_cs(0.0021249,0.85732012,0.5,alpha, 'red')
gradient_min4 = Gradient_Method_cs(0.0021249,0.7134762,0.5,alpha,'orange')
gradient_min5 = Gradient_Method_cs(4e-3,1,1,alpha,'yellow')
gradient_min6 = Gradient_Method_cs(0.002125116047106075,0.6,1,alpha,'green')
gradient_min7 = Gradient_Method_cs(4e-3,0.4,1,alpha,'blue')
'''
'''
gradient_special = Gradient_Method_cs(6e-3,0.9,1,alpha,'blue')
'''

print(gradient_min1)
print(gradient_min2)

a = NLL_Finder_cs(gradient_min1[1], gradient_min1[0], gradient_min1[2], 295, energy_midpoints)
b = NLL_Finder_cs(gradient_min1[1], gradient_min1[0]+0.0001, gradient_min1[2]+0.001, 295, energy_midpoints)
c = NLL_Finder_cs(gradient_min1[1], gradient_min1[0]-0.000025, gradient_min1[2], 295, energy_midpoints)
d = NLL_Finder_cs(gradient_min1[1], gradient_min1[0], gradient_min1[2]-0.005, 295, energy_midpoints)
print(a)
print(b)
print(c)
print(d)
'''
print(gradient_min3)
print(gradient_min4)
print(gradient_min5)
print(gradient_min6)
print(gradient_min7)
'''
'''
print(gradient_min3)
print(gradient_min4)
print(gradient_min5)
print(gradient_special)
'''
#%%
css = np.linspace(0.01,10, num = 200)
NLLs = []
for i in range(len(css)):
    NLLs.append(NLL_Finder_cs(0.00724957,1.20074269,css[i],295,energy_midpoints))
plt.figure()
plt.plot(css,NLLs)
'''
#%%
'''
'''
thetas_for_theta1 = np.linspace(0.6,0.75, num = 2000)
NLLs_theta1 = NLLs_Finder(thetas_for_theta1,diff)
theta1_minimisee = Parabola_Minimiser(0.201*np.pi,0.21*np.pi,0.22*np.pi,thetas_for_theta1,NLLs_theta1)

thetas_for_theta2 = np.linspace(0.8,0.95, num = 2000)
NLLs_theta2 = NLLs_Finder(thetas_for_theta2,diff)
theta2_minimisee = Parabola_Minimiser(0.2822*np.pi,0.284*np.pi,0.286*np.pi,thetas_for_theta2,NLLs_theta2)


theta1 = (theta1_minimisee.minimise())
theta1std = theta1_minimisee.standard_dev_finder()

theta2 = (theta2_minimisee.minimise())
theta2std = theta2_minimisee.standard_dev_finder()
'''

'''
y_vals = parabola_finder(thetas, 0.26, 0.27, 0.31, NLLs[find_nearest(thetas,0.26)], NLLs[find_nearest(thetas,0.27)], NLLs[find_nearest(thetas,0.31)])
plt.figure()
plt.plot(0.26,NLLs[find_nearest(thetas,0.26)],'x')
plt.plot(0.27,NLLs[find_nearest(thetas,0.27)],'x')
plt.plot(0.31,NLLs[find_nearest(thetas,0.31)],'x')

plt.plot(thetas,y_vals)
'''

#%%
x = np.linspace(-1,1,num=100000000)
y = (x-0.2)**2
minimisee = Parabola_Minimiser(-0.9,-0.5,0.3,x,y)
minimum = minimisee.minimise()
print(minimum)
'''

new_lambdas = find_lambdas(0.8573978067025839,0.002125116047106075,295,energy_midpoints)
plt.figure()
plt.stairs(new_lambdas, edges = energy_edges)
plt.title('Predicted right')
new_lambdas = find_lambdas(0.7134461132491364,0.0021251377742067794,295,energy_midpoints)
plt.figure()
plt.stairs(new_lambdas, edges = energy_edges)
plt.title('Predicted left')
plt.figure()
plt.stairs(events, edges = energy_edges)
plt.title('Data')
plt.figure()
plt.stairs(flux_pred, edges = energy_edges)
plt.title('Unoscillated Prediction')


def Plot_Graphs(lambdas,data):
    plt.figure()
    plt.stairs(lambdas, edges = energy_edges)
    plt.title('Predicted')
    plt.figure()
    plt.stairs(data, edges = energy_edges)
    plt.title('Data')
    plt.figure()
    plt.stairs(flux_pred, edges = energy_edges)
    plt.title('Unoscillated Prediction')

'''