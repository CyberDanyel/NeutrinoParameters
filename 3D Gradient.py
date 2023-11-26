import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from Parabola_Minimiser import Gradient_Method_Numerical_plane,Gradient_Method_plane,Parabola_Minimiser, Univariate_Minimisation, Univariate_Minimisation_cs,Univariate_Minimisation_plane, Gradient_Method, Newton_Method,Gradient_Method_cs,Gradient_Method_Numerical,Gradient_Method_Numerical_cs
from Equations import NLL_Finder, NLLs_Finder, find_lambdas, NLL_Finder_cs, NLLs_Finder_cs

minimum_left_analytical = Gradient_Method_cs(2.19e-3, 0.651, 1.08, [1e-9,1e-4,1e-6])
minimum_right_analytical = Gradient_Method_cs(2.19e-3, 0.919, 1.08, [1e-9,1e-4,1e-6])

minimum_left_numerical = Gradient_Method_Numerical_cs(2.19e-3, 0.651, 1.08, [1e-9,1e-4,1e-6], 1e-12)
minimum_right_numerical = Gradient_Method_Numerical_cs(2.19e-3, 0.919, 1.08, [1e-9,1e-4,1e-6], 1e-12)

print(minimum_left_analytical)
print(minimum_right_analytical)

print(minimum_left_numerical)
print(minimum_right_numerical)
