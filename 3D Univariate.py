import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from Parabola_Minimiser import Gradient_Method_Numerical_plane,Gradient_Method_plane,Parabola_Minimiser, Univariate_Minimisation, Univariate_Minimisation_cs,Univariate_Minimisation_plane, Gradient_Method, Newton_Method,Gradient_Method_cs,Gradient_Method_Numerical,Gradient_Method_Numerical_cs
from Equations import NLL_Finder, NLLs_Finder, find_lambdas, NLL_Finder_cs, NLLs_Finder_cs

minimum_left = Univariate_Minimisation_cs(0.651, 1.08, [0.6,0.65,0.7], [2.15e-3,2.19e-3,2.23e-3], [1.07,1.08,1.9])
minimum_right = Univariate_Minimisation_cs(0.919, 1.08, [0.85,0.91,0.95], [2.15e-3,2.19e-3,2.23e-3], [1.07,1.08,1.9])
print(minimum_left)
print(minimum_right)
