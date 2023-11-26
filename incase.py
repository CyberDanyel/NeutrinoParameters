class Parabola_Minimiser:
    def __init__(self, x0, x1, x2, x_axis, y_axis):
        self.__x0 = x0
        self.__x1 = x1
        self.__x2 = x2
        self.__x_axis = x_axis
        self.__y_axis = y_axis
        
    def x3_Finder(self):
        x0 = self.__x0
        x1 = self.__x1
        x2 = self.__x2
        x_axis = self.__x_axis
        y_axis = self.__y_axis
        y0 = y_axis[find_nearest(x_axis,x0)]
        y1 = y_axis[find_nearest(x_axis,x1)]
        y2 = y_axis[find_nearest(x_axis,x2)]
        x3 = (1/2)*((x2**2-x1**2)*y0+(x0**2-x2**2)*y1+(x1**2-x0**2)*y2)/((x2-x1)*y0+(x0-x2)*y1+(x1-x0)*y2)
        y3 = y_axis[find_nearest(x_axis,x3)]
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
            
    def minimise(self):
        j = 0
        print(j)
        min_x = self.__x0
        while True:
            j = j+1
            print(j)
            new_min_x = Parabola_Minimiser.x3_Finder(self)
            if abs((new_min_x - min_x)/min_x) < 1e-3:
                self.__min_x = new_min_x
                return new_min_x
            else:
                min_x = new_min_x
                pass
            
    def standard_dev_finder(self):
        min_NLL = self.__y_axis[find_nearest(self.__x_axis,self.__min_x)]
        stdNLL = min_NLL + 0.5
        stdx = self.__x_axis[find_nearest(self.__y_axis,stdNLL)]
        std = abs(stdx-self.__min_x)
        return std
    
def Univariate_Minimisation(init_theta,theta0,theta1,theta2,diff0,diff1,diff2,thetas,diffs,NLLs):
    diff_minimisee = Parabola_Minimiser(diff0,diff1,diff2,diffs,NLLs[:,find_nearest(thetas,init_theta)])
    diff_minimised = diff_minimisee.minimise()
    for i in range(20):
        theta_minimisee = Parabola_Minimiser(theta0,theta1,theta2,thetas,NLLs[find_nearest(diffs,diff_minimised),:])
        theta_minimised = theta_minimisee.minimise()
        diff_minimisee = Parabola_Minimiser(diff0,diff1,diff2,diffs,NLLs[:,find_nearest(thetas,theta_minimised)])
        diff_minimised = diff_minimisee.minimise()
    return diff_minimised, theta_minimised