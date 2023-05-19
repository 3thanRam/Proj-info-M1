
import numpy as np
import time
from itertools import product
from scipy.optimize import minimize
from collections import Counter
import matplotlib.pyplot as plt
import scipy.optimize as spopt


def fct_fit(to,ko):
    """
    Returns the fitted function values based on the given time and diffusion constant.

    Args:
    to (numpy.ndarray): 1D array of time values.
    ko (float): Diffusion constant.

    Returns:
    numpy.ndarray: 1D array of function values.
    """
    global DO
    do,d1=DO
    RES=np.concatenate((do/(1+do*ko*to),d1/(1+d1*ko*to)))
    return(RES)


def fct_fit2(to,ko):
    """
    Returns the fitted function values based on the given time, diffusion constant, and dimensional number.

    Args:
    to (numpy.ndarray): 1D array of time values.
    ko (float): Diffusion constant.

    Returns:
    numpy.ndarray: 1D array of function values.
    """
    global DO,DIM_Numb,p_var
    do,d1=DO
    RES=np.concatenate((do**(1/2)*p_var*(ko*to)**(-DIM_Numb/4),d1**(1/2)*p_var*(ko*to)**(-DIM_Numb/4)))
    return(RES)



def Denfct(DIM_Numb0,Density,Trange,ax):
    """
    Plots the particle densities in a DIM_Numb-dimensional box as a function of time.

    Parameters:
        DIM_Numb0 (int): The number of dimensions.
        Dens (list): A list of the particle and antiparticle densities.
        Trange (numpy.ndarray): The time array.

    Returns:
        None
    """
    global DO,p_var,DIM_Numb
    # Initialize parameters
    DIM_Numb=DIM_Numb0
    DT=['',0.1,0.1,0.05]
    dt=float(DT[DIM_Numb])


    ax.set_xlabel('Time(s)')
    ax.set_ylabel('N')
    ax.set_title('Particle Densities in a '+str(DIM_Numb)+'D box\n as a function of Time')
    if DIM_Numb==1:
        p_var=(2*np.pi)**(-3/4)
    elif DIM_Numb==2:
        p_var=2**(-3/2)/np.pi
    elif DIM_Numb==3:
        p_var=2**(-7/4)*(np.pi)**(-5/4)
    
    Ttheo=np.linspace(Trange[0],Trange[-1],10**3)
    d0=Density[0][0]
    d1=Density[1][0]
    DO=d0,d1 #initial densities for each particle type
    k2=spopt.curve_fit(fct_fit,Trange,Density[0]+Density[1])[0][0]
    Dpart_theo=d0/(1+d0*k2*Ttheo)
    
    #DO=np.array(Density[0][1:]),np.array(Density[1][1:])
    D=spopt.curve_fit(fct_fit2,Trange[1:],Density[0][1:]+Density[1][1:],bounds=(0,np.inf))[0][0]
    DLine=d0*Ttheo[1:]**(-1)/Ttheo[1]**(-1)
    D_theo0=[d0]+[d0**(1/2)*p_var*(D*dt)**(-DIM_Numb/4) for dt in Ttheo[1:]]



    ax.loglog(Trange,Density[0],color='red',label='Particle Density')
    ax.loglog(Ttheo,Dpart_theo,color='purple',label='Particle Theo')
    ax.loglog(Ttheo[1:],DLine,color='black',label=' t^-1')
    ax.loglog(Ttheo,D_theo0,color='grey',label='Particle FIT')
    if Density[0]!=Density[1]: 
        ax.loglog(Trange,Density[1],color='blue',label='Anti-Particle Density')
        Dantipart_theo=d1/(1+d1*k2*Ttheo)
        ax.loglog(Ttheo,Dantipart_theo,color='green',label='Antiparticle Theo')
        D_theo1=[d1]+[d1**(1/2)*p_var*(D*dt)**(-DIM_Numb/4) for dt in Ttheo[1:]]
        ax.loglog(Ttheo,D_theo1,color='grey',label='AntiParticle FIT')
    
    ax.legend()
    

