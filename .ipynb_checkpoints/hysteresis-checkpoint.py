#!/usr/bin/python

"""
hysteresis.py: a program to deliver required content for a the hysteresis experiment jupyter notebook
"""

######### Make default plot style #########
import matplotlib

# Define the default plot style through rcParams
def setplotstyle():
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['figure.figsize'] = (10, 7)
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['lines.linewidth'] = 3
    for p in ['xtick', 'ytick']:
        matplotlib.rcParams[p+'.major.size'] = 10
        matplotlib.rcParams[p+'.minor.size'] = 5
        matplotlib.rcParams[p+'.major.width'] = 2.5
        matplotlib.rcParams[p+'.minor.width'] = 1.5

def draw_classic_axes(ax, x=0, y=0, xlabeloffset=.1, ylabeloffset=.07):
    ax.set_axis_off()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.annotate(
        ax.get_xlabel(), xytext=(x1, y), xy=(x0, y),
        arrowprops=dict(arrowstyle="<-"), va='center'
    )
    ax.annotate(
        ax.get_ylabel(), xytext=(x, y1), xy=(x, y0),
        arrowprops=dict(arrowstyle="<-"), ha='center'
    )
    for pos, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        ax.text(pos, y - xlabeloffset, label.get_text(),
                ha='center', va='bottom')
    for pos, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        ax.text(x - ylabeloffset, pos, label.get_text(),
                ha='right', va='center')

# The theoretical model for Hysteresis comes from the [Jiles–Atherton model](https://en.wikipedia.org/wiki/Jiles%E2%80%93Atherton_model) which someone had conveniently already coded up [here](https://github.com/Ryan-O-Connor/pyjam/tree/main/src).
        
######### Import packages #########
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import root_scalar, minimize
from functools import partial

def langevin(x):
    # Computes langevin function:
    #   L(x) = coth(x) - 1/x
    return 1/np.tanh(x) - 1/x


def anhysteretic(M, H, Ms, a, alpha):
    # Computes anhysteretic function for M
    #   y == M/Ms - L(He/a)
    He = H + alpha*M
    y = M/Ms - langevin(He/a)
    return y

def bracket_anhysteretic(H, Ms, a, alpha):
    # Find bracket for solving for anhysteretic curve at H
    Man = partial(anhysteretic, H=H, Ms=Ms, a=a, alpha=alpha)
    N = 20
    limits = [-1 + 0.1*x for x in range(N+1)]
    for i in range(N):
        Mlo = Man(limits[i]*Ms)
        Mhi = Man(limits[i+1]*Ms)
        if Mlo * Mhi < 0:
            return [limits[i]*Ms, limits[i+1]*Ms]
    return None

def solve_anhysteretic(H, Ms, a, alpha):
    if H == 0 or a == 0 or Ms == 0:
        M = 0
    else:
        bracket = bracket_anhysteretic(H, Ms, a, alpha)
        if bracket is not None:
            Man = partial(anhysteretic, H=H, Ms=Ms, a=a, alpha=alpha)
            sol = root_scalar(Man, bracket=bracket, method='bisect')
            M = sol.root
            if not sol.converged:
                print('Root finding did not converge for H={}'.format(H))
        else:
            print('Could not find bracket for H={}'.format(H))
            M = 0
    return M

def plot_anhysteretic(axes, Hmin, Hmax, Ms, a, alpha):
    H = np.linspace(Hmin, Hmax, 100)
    M = [solve_anhysteretic(h, Ms, a, alpha) for h in H]
    M = np.array(M)
    axes.plot(H, M/Ms)

###############################################################################
#
# Jiles-Atherton Equation solving functions
#
###############################################################################


def coth(x):
    # Hyperbolic cotangent (syntactic sugar)
    return 1 / np.tanh(x)


def L(x):
    # Langevin function
    if x == 0:
        return 0
    else:
        return coth(x) - 1 / x


def dLdx(x):
    # Derivative of langevin function
    if x == 0:
        return 1 / 3
    else:
        return 1 - coth(x) ** 2 + 1 / x ** 2


def dMdH(M, H, Ms, a, alpha, k, c, delta):
    # Derivative of magnetization
    He = H + alpha * M
    Man = Ms * L(He / a)
    dM = Man - M
    dMdH_num = dM / (delta * k - alpha * dM) + c * Ms / a * dLdx(He / a)
    dMdH_den =  (1 + c - c * alpha * Ms / a * dLdx(He / a))
    return dMdH_num / dMdH_den


def euler(dMdH, M0, H):
    # Euler ODE integrator for J-A equation
    M = [M0]
    for i in range(len(H) - 1):
        dH_i = H[i + 1] - H[i]
        dMdH_i = dMdH(M[i], H[i + 1], delta=np.sign(dH_i))
        M.append(M[i] + dMdH_i * dH_i)
    return M


def H_arr(Hlimit, curve_type):
    # External field intensity input
    if curve_type == 'initial':
        H = np.linspace(0, Hlimit, 500, endpoint=True)
    elif curve_type == 'loop':
        H1 = np.linspace(Hlimit, -Hlimit, 1000, endpoint=False)
        H2 = np.linspace(-Hlimit, Hlimit, 1000, endpoint=True)
        H = np.append(H1, H2)
    elif curve_type == 'full':
        H1 = np.linspace(0, Hlimit, 500, endpoint=False)
        H2 = np.linspace(Hlimit, -Hlimit, 1000, endpoint=False)
        H3 = np.linspace(-Hlimit, Hlimit, 1000, endpoint=True)
        H = np.append(H1, np.append(H2, H3))
    else:
        print('Invalid curve type')
        H = None
    return H


setplotstyle() # Set the plot style