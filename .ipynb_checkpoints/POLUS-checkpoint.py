# POLUS.py

"""
POLUS.py: a program to aid computations for experimental content at UTAS
Additional information can be found at https://utasphys.cloud.edu.au/POLUS/reference/computing/
Written and curated by AJM, created May 2022, Version 0.0.1 dated May 2022.
"""

######### Make default plot style #########
import matplotlib

# Define the default plot style through rcParams
def setplotstyle():
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['figure.figsize'] = (10, 7)
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['axes.linewidth'] = 2.5
    for p in ['xtick', 'ytick']:
        matplotlib.rcParams[p+'.major.size'] = 10
        matplotlib.rcParams[p+'.minor.size'] = 5
        matplotlib.rcParams[p+'.major.width'] = 2.5
        matplotlib.rcParams[p+'.minor.width'] = 1.5

######### Import packages #########
import pandas as pd # panads for data manipulation
import numpy as np # numpy for all things mathematical/numerical
import matplotlib.pyplot as plt # matplotlib.pyplot for plotting
import scipy.constants as const # scipy.constants for physical constants
from scipy.optimize import curve_fit # scipy.optimize for curve fitting
import scipy.integrate as integrate #scipy.integrate for numerical integration
from scipy.signal import find_peaks # scipy.signal for peak finding
        
######### "Global variables" (not actually gloabl variables in the python sense) #########
R = const.R
hbar = const.hbar
kb = const.k
m_e = const.m_e
mu_0 = const.mu_0
e = const.e

setplotstyle() # Set the plot style