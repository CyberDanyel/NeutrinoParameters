#DOESNT WORK FOR A MINIMUM OF 0
from Equations import NLL_Finder,NLL_Finder_cs,NLL_theta_derivative_finder,NLL_delm_derivative_finder,NLL_delm_derivative_finder_cs,NLL_theta_derivative_finder_cs,NLL_cs_derivative_finder_cs,Inverted_Hessian,Create_Hessian,linear_interpolation
from Secant_Root_Finder import Secant_Root_Finder_theta,Secant_Root_Finder_delm,Secant_Root_Finder_theta_cs,Secant_Root_Finder_delm_cs,Secant_Root_Finder_cs_cs
import numpy as np
import matplotlib.pyplot as plt

energy_edges = np.linspace(0, 10, num = 201, endpoint = True)

#Using the midpoints of the bins for lambda
energy_midpoints = []
for i in range(len(energy_edges)-1):
    new_midpoint = (energy_edges[i+1] - energy_edges[i])/2 + energy_edges[i]
    energy_midpoints.append(new_midpoint)
    
energy_midpoints = np.array(energy_midpoints)

class Parabola_Minimiser:
    def __init__(self,exes):
        self.__x0 = exes[0]
        self.__x1 = exes[1]
        self.__x2 = exes[2]
        #self.__y_axis = y_axis
        
    def x3_Finder_theta(self,delm):
        x0 = self.__x0
        x1 = self.__x1
        x2 = self.__x2
        self.__prev_x0 = x0
        self.__prev_x1 = x1
        self.__prev_x2 = x2
        #y_axis = self.__y_axis
        y0 = NLL_Finder(x0, delm, 295, energy_midpoints)
        y1 = NLL_Finder(x1, delm, 295, energy_midpoints)
        y2 = NLL_Finder(x2, delm, 295, energy_midpoints)
        self.__y0 = y0
        self.__y1 = y1
        self.__y2 = y2
        x3 = (1/2)*((x2**2-x1**2)*y0+(x0**2-x2**2)*y1+(x1**2-x0**2)*y2)/((x2-x1)*y0+(x0-x2)*y1+(x1-x0)*y2)
        y3 = NLL_Finder(x3, delm, 295, energy_midpoints)
        max_y = max(y0,y1,y2,y3)
        if max_y == y0:
            self.__x0 = x1
            self.__x1 = x2
            self.__x2 = x3
        elif max_y == y1:
            self.__x0 = x0
            self.__x1 = x2
            self.__x2 = x3       
        elif max_y == y2:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x3           
        elif max_y == y3:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x2
        return x3
      
    def x3_Finder_delm(self,theta):
        x0 = self.__x0
        x1 = self.__x1
        x2 = self.__x2
        self.__prev_x0 = x0
        self.__prev_x1 = x1
        self.__prev_x2 = x2
        #y_axis = self.__y_axis
        y0 = NLL_Finder(theta, x0, 295, energy_midpoints)
        y1 = NLL_Finder(theta, x1, 295, energy_midpoints)
        y2 = NLL_Finder(theta, x2, 295, energy_midpoints)
        self.__y0 = y0
        self.__y1 = y1
        self.__y2 = y2
        x3 = (1/2)*((x2**2-x1**2)*y0+(x0**2-x2**2)*y1+(x1**2-x0**2)*y2)/((x2-x1)*y0+(x0-x2)*y1+(x1-x0)*y2)
        y3 = NLL_Finder(theta, x3, 295, energy_midpoints)
        max_y = max(y0,y1,y2,y3)
        if max_y == y0:
            self.__x0 = x1
            self.__x1 = x2
            self.__x2 = x3
        elif max_y == y1:
            self.__x0 = x0
            self.__x1 = x2
            self.__x2 = x3       
        elif max_y == y2:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x3           
        elif max_y == y3:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x2
        return x3
    
    def x3_Finder_theta_cs(self,delm,cs):
        x0 = self.__x0
        x1 = self.__x1
        x2 = self.__x2
        #y_axis = self.__y_axis
        y0 = NLL_Finder_cs(x0, delm, cs,295, energy_midpoints)
        y1 = NLL_Finder_cs(x1, delm, cs,295, energy_midpoints)
        y2 = NLL_Finder_cs(x2, delm, cs,295, energy_midpoints)
        x3 = (1/2)*((x2**2-x1**2)*y0+(x0**2-x2**2)*y1+(x1**2-x0**2)*y2)/((x2-x1)*y0+(x0-x2)*y1+(x1-x0)*y2)
        y3 = NLL_Finder_cs(x3, delm, cs,295, energy_midpoints)
        max_y = max(y0,y1,y2,y3)
        if max_y == y0:
            self.__x0 = x1
            self.__x1 = x2
            self.__x2 = x3
        elif max_y == y1:
            self.__x0 = x0
            self.__x1 = x2
            self.__x2 = x3       
        elif max_y == y2:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x3           
        elif max_y == y3:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x2
        return x3

    def x3_Finder_delm_cs(self,theta,cs):
        x0 = self.__x0
        x1 = self.__x1
        x2 = self.__x2
        #y_axis = self.__y_axis
        y0 = NLL_Finder_cs(theta, x0,cs, 295, energy_midpoints)
        y1 = NLL_Finder_cs(theta, x1,cs, 295, energy_midpoints)
        y2 = NLL_Finder_cs(theta, x2,cs, 295, energy_midpoints)
        x3 = (1/2)*((x2**2-x1**2)*y0+(x0**2-x2**2)*y1+(x1**2-x0**2)*y2)/((x2-x1)*y0+(x0-x2)*y1+(x1-x0)*y2)
        y3 = NLL_Finder_cs(theta, x3,cs, 295, energy_midpoints)
        max_y = max(y0,y1,y2,y3)
        if max_y == y0:
            self.__x0 = x1
            self.__x1 = x2
            self.__x2 = x3
        elif max_y == y1:
            self.__x0 = x0
            self.__x1 = x2
            self.__x2 = x3       
        elif max_y == y2:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x3           
        elif max_y == y3:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x2
        return x3
       
    def x3_Finder_cs_cs(self,theta,delm):
        x0 = self.__x0
        x1 = self.__x1
        x2 = self.__x2
        #y_axis = self.__y_axis
        y0 = NLL_Finder_cs(theta,delm, x0, 295, energy_midpoints)
        y1 = NLL_Finder_cs(theta,delm, x1, 295, energy_midpoints)
        y2 = NLL_Finder_cs(theta,delm, x2, 295, energy_midpoints)
        x3 = (1/2)*((x2**2-x1**2)*y0+(x0**2-x2**2)*y1+(x1**2-x0**2)*y2)/((x2-x1)*y0+(x0-x2)*y1+(x1-x0)*y2)
        y3 = NLL_Finder_cs(theta,delm, x3, 295, energy_midpoints)
        max_y = max(y0,y1,y2,y3)
        if max_y == y0:
            self.__x0 = x1
            self.__x1 = x2
            self.__x2 = x3
        elif max_y == y1:
            self.__x0 = x0
            self.__x1 = x2
            self.__x2 = x3       
        elif max_y == y2:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x3           
        elif max_y == y3:
            self.__x0 = x0
            self.__x1 = x1
            self.__x2 = x2
        return x3
     
    def parabolic_error(self):
        x = [self.__prev_x0,self.__prev_x1,self.__prev_x2]
        y = [self.__y0,self.__y1,self.__y2]
        d = (x[1]-x[0])*(x[2]-x[0])*(x[2]-x[1])
        error = 2*((x[1]-x[0])*y[2]+(x[0]-x[2])*y[1]+(x[2]-x[1])*y[0])/d
        return 1/np.sqrt(error)
     
    def minimise_theta(self,delm):
        j = 0
        #print(j)
        min_x = self.__x0
        while True:
            j = j+1
            #print(j)
            new_min_x = Parabola_Minimiser.x3_Finder_theta(self,delm)
            if abs((new_min_x - min_x)/min_x) < 1e-9:
                self.__min_x = new_min_x
                return new_min_x
            else:
                min_x = new_min_x
                pass

    def minimise_delm(self,theta):
        j = 0
        #print(j)
        min_x = self.__x0
        while True:
            j = j+1
            #print(j)
            new_min_x = Parabola_Minimiser.x3_Finder_delm(self,theta)
            if abs((new_min_x - min_x)/min_x) < 1e-9:
                self.__min_x = new_min_x
                return new_min_x
            else:
                min_x = new_min_x
                pass

    def minimise_theta_cs(self,delm,cs):
        j = 0
        #print(j)
        min_x = self.__x0
        while True:
            j = j+1
            #print(j)
            new_min_x = Parabola_Minimiser.x3_Finder_theta_cs(self,delm,cs)
            if abs((new_min_x - min_x)/min_x) < 1e-9:
                self.__min_x = new_min_x
                return new_min_x
            else:
                min_x = new_min_x
                pass    

    def minimise_delm_cs(self,theta,cs):
        j = 0
        #print(j)
        min_x = self.__x0
        while True:
            j = j+1
            #print(j)
            new_min_x = Parabola_Minimiser.x3_Finder_delm_cs(self,theta,cs)
            if abs((new_min_x - min_x)/min_x) < 1e-9:
                self.__min_x = new_min_x
                return new_min_x
            else:
                min_x = new_min_x
                pass
            
    def minimise_cs_cs(self,theta,delm):
        j = 0
        #print(j)
        min_x = self.__x0
        while True:
            j = j+1
            #print(j)
            new_min_x = Parabola_Minimiser.x3_Finder_cs_cs(self,theta,delm)
            if abs((new_min_x - min_x)/min_x) < 1e-9:
                self.__min_x = new_min_x
                return new_min_x
            else:
                min_x = new_min_x
                pass
            
    def secant_theta(self,delm):
        min_NLL = NLL_Finder(self.__min_x, delm, 295, energy_midpoints)
        right_std = Secant_Root_Finder_theta([self.__min_x+0.001,self.__min_x+0.005],min_NLL,delm)
        left_std = Secant_Root_Finder_theta([self.__min_x-0.001,self.__min_x-0.005],min_NLL,delm)
        rstd = right_std - self.__min_x
        lstd = self.__min_x - left_std
        return lstd,rstd

    def secant_delm(self,theta):
        min_NLL = NLL_Finder(theta, self.__min_x, 295, energy_midpoints)
        right_std = Secant_Root_Finder_delm([self.__min_x+0.0001,self.__min_x+0.0002],min_NLL,theta)
        left_std = Secant_Root_Finder_delm([self.__min_x-0.0001,self.__min_x-0.0002],min_NLL,theta)
        rstd = right_std - self.__min_x
        lstd = self.__min_x - left_std
        return lstd,rstd

def secant_theta(minval,delm):
    min_NLL = NLL_Finder(minval, delm, 295, energy_midpoints)
    right_std = Secant_Root_Finder_theta([minval+0.001,minval+0.005],min_NLL,delm)
    left_std = Secant_Root_Finder_theta([minval-0.001,minval-0.005],min_NLL,delm)
    rstd = right_std - minval
    lstd = minval - left_std
    return lstd,rstd

def secant_delm(minval,theta):
    min_NLL = NLL_Finder(theta, minval, 295, energy_midpoints)
    right_std = Secant_Root_Finder_delm([minval+0.0001,minval+0.0002],min_NLL,theta)
    left_std = Secant_Root_Finder_delm([minval-0.0001,minval-0.0002],min_NLL,theta)
    rstd = right_std - minval
    lstd = minval - left_std
    return lstd,rstd

def secant_theta_cs(minval,delm,cs):
    min_NLL = NLL_Finder_cs(minval, delm,cs, 295, energy_midpoints)
    right_std = Secant_Root_Finder_theta_cs([minval+0.001,minval+0.005],min_NLL,delm,cs)
    left_std = Secant_Root_Finder_theta_cs([minval-0.001,minval-0.005],min_NLL,delm,cs)
    rstd = right_std - minval
    lstd = minval - left_std
    return lstd,rstd

def secant_delm_cs(minval,theta,cs):
    min_NLL = NLL_Finder_cs(theta, minval,cs, 295, energy_midpoints)
    right_std = Secant_Root_Finder_delm_cs([minval+0.0001,minval+0.0002],min_NLL,theta,cs)
    left_std = Secant_Root_Finder_delm_cs([minval-0.0001,minval-0.0002],min_NLL,theta,cs)
    rstd = right_std - minval
    lstd = minval - left_std
    return lstd,rstd

def secant_cs_cs(minval,delm,theta):
    min_NLL = NLL_Finder_cs(theta, delm, minval,295, energy_midpoints)
    right_std = Secant_Root_Finder_cs_cs([minval+0.01,minval+0.02],min_NLL,theta,delm)
    left_std = Secant_Root_Finder_cs_cs([minval-0.01,minval-0.02],min_NLL,theta,delm)
    rstd = right_std - minval
    lstd = minval - left_std
    return lstd,rstd

def Univariate_Minimisation(init_theta,thetas,delms,colour,ax):
    theta_minimisee = Parabola_Minimiser(thetas)
    delm_minimisee = Parabola_Minimiser(delms)
    delm_minimised = delm_minimisee.minimise_delm(init_theta)
    x0 = [delm_minimised,init_theta]
    ax.plot(x0[1],x0[0],'o',color=colour,label = 'Starting Point')
    j = 0
    while True:
        j = j+1
        print(j)
        theta_minimised = theta_minimisee.minimise_theta(delm_minimised)
        ax.plot([x0[1],theta_minimised],[x0[0],x0[0]],color = colour)
        #print(x0[1])
        #print(theta_minimised)
       # print(x0[0])
        delm_minimised = delm_minimisee.minimise_delm(theta_minimised)
        ax.plot([theta_minimised,theta_minimised],[x0[0],delm_minimised],color = colour)
        #print(theta_minimised)
        #print(x0[1])
        #print(delm_minimised)
        x1 = [delm_minimised,theta_minimised]
        
        #plt.plot(x1[1],x1[0],'X')
        if abs((np.sqrt((x1[0]**2+x1[1]**2))-np.sqrt((x0[0]**2+x0[1]**2)))/np.sqrt((x0[0]**2+x0[1]**2))) < 1e-3:
            theta_err = secant_theta(theta_minimised,delm_minimised)
            delm_err = secant_delm(delm_minimised,theta_minimised)
            return [delm_minimised, theta_minimised], [delm_err, theta_err]
        else:
            x0 = x1

def Univariate_Minimisation_plane(init_theta,thetas,delms,cs):
    theta_minimisee = Parabola_Minimiser(thetas)
    delm_minimisee = Parabola_Minimiser(delms)
    delm_minimised = delm_minimisee.minimise_delm_cs(init_theta,cs)
    x0 = [delm_minimised,init_theta]
    #j = 0
    while True:
        #j = j+1
        #print(j)
        theta_minimised = theta_minimisee.minimise_theta_cs(delm_minimised,cs)
        delm_minimised = delm_minimisee.minimise_delm_cs(theta_minimised,cs)
        x1 = [delm_minimised,theta_minimised]
        if abs((np.sqrt((x1[0]**2+x1[1]**2))-np.sqrt((x0[0]**2+x0[1]**2)))/np.sqrt((x0[0]**2+x0[1]**2))) < 1e-3:
            theta_err = secant_theta_cs(theta_minimised,delm_minimised,cs)
            delm_err = secant_delm_cs(delm_minimised,theta_minimised,cs)
            return [delm_minimised, theta_minimised], [delm_err, theta_err]
        else:
            x0 = x1

def Univariate_Minimisation_cs(init_theta,init_cs,thetas,delms,css):
    theta_minimisee = Parabola_Minimiser(thetas)
    delm_minimisee = Parabola_Minimiser(delms)
    cs_minimisee = Parabola_Minimiser(css)
    delm_minimised = delm_minimisee.minimise_delm_cs(init_theta, init_cs)
    theta_minimised = theta_minimisee.minimise_theta_cs(delm_minimised, init_cs)
    x0 = [delm_minimised,theta_minimised,init_cs]
    j = 0
    while True:
        j = j+1
        print(j)
        cs_minimised = cs_minimisee.minimise_cs_cs(theta_minimised, delm_minimised)
        delm_minimised = delm_minimisee.minimise_delm_cs(theta_minimised,cs_minimised)
        theta_minimised = theta_minimisee.minimise_theta_cs(delm_minimised,cs_minimised)
        x1 = [delm_minimised,theta_minimised,cs_minimised]
        if abs((np.sqrt((x1[0]**2+x1[1]**2+x1[2]**2))-np.sqrt((x0[0]**2+x0[1]**2+x0[2]**2)))/np.sqrt(x0[0]**2+x0[1]**2+x0[2]**2)) < 1e-4:
            theta_err = secant_theta_cs(theta_minimised,delm_minimised,cs_minimised)
            delm_err = secant_delm_cs(delm_minimised,theta_minimised,cs_minimised)
            cs_err = secant_cs_cs(cs_minimised,delm_minimised,theta_minimised)    
            return [delm_minimised, theta_minimised, cs_minimised], [delm_err, theta_err, cs_err]
        else:
            x0 = x1
            
def Gradient_Method(init_delm,init_theta,alpha,colour,ax):
    x0 = np.array([init_delm,init_theta])
    ax.plot(x0[1],x0[0], 'o', color = colour, label = 'Starting Point')
    alpha = np.array(alpha)
    j = 0
    while True:
        j = j+1
        print(j)
        x1 = x0 - alpha*np.array([NLL_delm_derivative_finder(x0[1], x0[0], 295, energy_midpoints), NLL_theta_derivative_finder(x0[1], x0[0], 295, energy_midpoints)])
        #ax.plot(x1[1],x1[0], 'X', color = colour)
        exes = np.linspace(x0[1],x1[1])
        fs = linear_interpolation(exes, [x0[1],x1[1]], [x0[0],x1[0]])
        ax.plot(exes,fs,color = colour)
        if abs((np.sqrt((x1[0]**2+x1[1]**2))-np.sqrt((x0[0]**2+x0[1]**2)))/np.sqrt((x0[0]**2+x0[1]**2))) < 1e-6:
            theta_err = secant_theta(x1[1],x1[0])
            delm_err = secant_delm(x1[0],x1[1])
            #ax.plot(x1[1],x1[0],'X', color = colour)
            return x1,[delm_err,theta_err]
        else:
            x0 = x1

def Gradient_Method_plane(init_delm,init_theta,cs,alpha,colour,ax):
    x0 = np.array([init_delm,init_theta])
    alpha = np.array(alpha)
    j = 0
    while True:
        #j = j+1
        #print(j)
        x1 = x0 - alpha*np.array([NLL_delm_derivative_finder_cs(x0[1], x0[0], cs,295, energy_midpoints), NLL_theta_derivative_finder_cs(x0[1], x0[0], cs,295, energy_midpoints)])
        ax.plot(x1[1],x1[0], 'X', color = colour)
        exes = np.linspace(x0[1],x1[1])
        fs = linear_interpolation(exes, [x0[1],x1[1]], [x0[0],x1[0]])
        ax.plot(exes,fs,color = colour)
        if abs((np.sqrt((x1[0]**2+x1[1]**2))-np.sqrt((x0[0]**2+x0[1]**2)))/np.sqrt((x0[0]**2+x0[1]**2))) < 1e-6:
            theta_err = secant_theta_cs(x1[1],x1[0],cs)
            delm_err = secant_delm_cs(x1[0],x1[1],cs)
            ax.plot(x1[1],x1[0],'X', color = colour)
            return x1,[delm_err,theta_err]
        else:
            x0 = x1    

def Gradient_Method_Numerical_plane(init_delm,init_theta,cs,alpha,colour,h):
    x0 = np.array([init_delm,init_theta])
    #ax.plot(x0[1],x0[0], 'o', color = colour)
    alpha = np.array(alpha)
    #j = 0
    while True:
        #j = j+1
        #print(j)
        dNLL_ddelm = (NLL_Finder_cs(x0[1], x0[0]+h, cs,295, energy_midpoints)-NLL_Finder_cs(x0[1], x0[0]-h, cs,295, energy_midpoints))/(2*h)
        dNLL_dtheta = (NLL_Finder_cs(x0[1]+h, x0[0], cs,295, energy_midpoints)-NLL_Finder_cs(x0[1]-h, x0[0], cs,295, energy_midpoints))/(2*h)
        x1 = x0 - alpha*np.array([dNLL_ddelm,dNLL_dtheta])
        #exes = np.linspace(x0[1],x1[1])
        #fs = linear_interpolation(exes, [x0[1],x1[1]], [x0[0],x1[0]])
        #ax.plot(exes,fs,color = colour)
        if abs((np.sqrt((x1[0]**2+x1[1]**2))-np.sqrt((x0[0]**2+x0[1]**2)))/np.sqrt((x0[0]**2+x0[1]**2))) < 1e-5:
            theta_err = secant_theta_cs(x1[1],x1[0],cs)
            delm_err = secant_delm_cs(x1[0],x1[1],cs)
            return x1,[delm_err,theta_err]
        else:
            x0 = x1
            pass

def Gradient_Method_Numerical(init_delm,init_theta,alpha,colour,h,ax):
    x0 = np.array([init_delm,init_theta])
    ax.plot(x0[1],x0[0], 'o', color = colour)
    alpha = np.array(alpha)
    j = 0
    while True:
        j = j+1
        print(j)
        dNLL_ddelm = (NLL_Finder(x0[1], x0[0]+h, 295, energy_midpoints)-NLL_Finder(x0[1], x0[0]-h, 295, energy_midpoints))/(2*h)
        dNLL_dtheta = (NLL_Finder(x0[1]+h, x0[0], 295, energy_midpoints)-NLL_Finder(x0[1]-h, x0[0], 295, energy_midpoints))/(2*h)
        x1 = x0 - alpha*np.array([dNLL_ddelm,dNLL_dtheta])
        exes = np.linspace(x0[1],x1[1])
        fs = linear_interpolation(exes, [x0[1],x1[1]], [x0[0],x1[0]])
        ax.plot(exes,fs,color = colour)
        if abs((np.sqrt((x1[0]**2+x1[1]**2))-np.sqrt((x0[0]**2+x0[1]**2)))/np.sqrt((x0[0]**2+x0[1]**2))) < 1e-3:
            theta_err = secant_theta(x1[1],x1[0])
            delm_err = secant_delm(x1[0],x1[1])
            return x1,[delm_err,theta_err]
        else:
            x0 = x1
            pass

def Gradient_Method_cs(init_delm,init_theta,init_cs,alpha):
    x0 = np.array([init_delm,init_theta,init_cs])
    #plt.plot(x0[1],x0[0], 'o', color = colour)
    alpha = np.array(alpha)
    while True:
        #print(i+1)
        x1 = x0 - alpha*np.array([NLL_delm_derivative_finder_cs(x0[1], x0[0], x0[2], 295, energy_midpoints), NLL_theta_derivative_finder_cs(x0[1], x0[0], x0[2], 295, energy_midpoints), NLL_cs_derivative_finder_cs(x0[1], x0[0], x0[2], 295, energy_midpoints)])
        if abs((np.sqrt((x1[0]**2+x1[1]**2+x1[2]**2))-np.sqrt((x0[0]**2+x0[1]**2+x0[2]**2)))/np.sqrt(x0[0]**2+x0[1]**2+x0[2]**2)) < 1e-6:
            theta_err = secant_theta_cs(x1[1],x1[0],x1[2])
            delm_err = secant_delm_cs(x1[0],x1[1],x1[2])
            cs_err = secant_cs_cs(x1[2],x1[0],x1[1])
            return x1,[delm_err,theta_err,cs_err]
        else:
            x0 = x1
            pass

def Gradient_Method_Numerical_cs(init_delm,init_theta,init_cs,alpha,h):
    x0 = np.array([init_delm,init_theta,init_cs])
    alpha = np.array(alpha)
    while True:
        dNLL_ddelm = (NLL_Finder_cs(x0[1], x0[0]+h,x0[2],295, energy_midpoints)-NLL_Finder_cs(x0[1], x0[0]-h,x0[2], 295, energy_midpoints))/(2*h)
        dNLL_dtheta = (NLL_Finder_cs(x0[1]+h, x0[0],x0[2],295, energy_midpoints)-NLL_Finder_cs(x0[1]-h, x0[0],x0[2], 295, energy_midpoints))/(2*h)
        dNLL_dcs = (NLL_Finder_cs(x0[1], x0[0],x0[2]+h,295, energy_midpoints)-NLL_Finder_cs(x0[1], x0[0],x0[2]-h, 295, energy_midpoints))/(2*h)
        x1 = x0 - alpha*np.array([dNLL_ddelm,dNLL_dtheta,dNLL_dcs])
        if abs((np.sqrt((x1[0]**2+x1[1]**2+x1[2]**2))-np.sqrt((x0[0]**2+x0[1]**2+x0[2]**2)))/np.sqrt(x0[0]**2+x0[1]**2+x0[2]**2)) < 1e-6:
            theta_err = secant_theta_cs(x1[1],x1[0],x1[2])
            delm_err = secant_delm_cs(x1[0],x1[1],x1[2])
            cs_err = secant_cs_cs(x1[2],x1[0],x1[1])
            return x1,[delm_err,theta_err,cs_err]
        else:
            x0 = x1
            pass

def Gradient_Method_2D(init_x,init_y,diff_x_f,diff_y_f,alpha):
    x0 = np.array([init_x,init_y])
    while True:
        x1 = x0 - alpha*np.array(diff_x_f(x0), diff_y_f(x0))
        plt.plot(x1[0],x1[1], 'X')
        if abs((np.sqrt((x1[0]**2+x1[1]**2))-np.sqrt((x0[0]**2+x0[1]**2)))/np.sqrt((x0[0]**2+x0[1]**2))) < 1e-6:
            return x1
        else:
            x0 = x1

def Newton_Method(init_delm,init_theta,colour,h):
    plt.figure()
    x0 = np.array([init_delm,init_theta])
    plt.plot(x0[1],x0[0], 'o', color = colour)
    for i in range(600):
        print(i+1)
        Hessian = Create_Hessian(x0[1],x0[0],295,energy_midpoints,h,2)
        #Inv_Hessian = Inverted_Hessian(x0[1], x0[0], 295, energy_midpoints,h,N)
        Inv_Hessian = np.linalg.inv(Hessian)
        x1 = x0 - np.matmul(Inv_Hessian,np.array([NLL_delm_derivative_finder(x0[1], x0[0], 295, energy_midpoints), NLL_theta_derivative_finder(x0[1], x0[0], 295, energy_midpoints)]))
        x0 = x1
        plt.plot(x1[1],x1[0], 'X', color = colour)
    return x1

    '''
        if (np.sqrt(x1[0]**2+x1[1]**2)-np.sqrt(x0[0]**2+x0[1]**2))/(np.sqrt(x0[0]**2+x0[1]**2)) < 1e-9:
            return x1
        else:
            x0 = x1
            i = i + 1
            pass
        '''
        