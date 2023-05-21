
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



def Denfct(DIM_Numb0,Dens,Trange):
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

    
    

    if DIM_Numb==1:
        p_var=(2*np.pi)**(-3/4)
    elif DIM_Numb==2:
        p_var=2**(-3/2)/np.pi
    elif DIM_Numb==3:
        p_var=2**(-7/4)*(np.pi)**(-5/4)
    #eps=1
    #D_var=[eps**2 /((4-DIM_Numb)*2*dt) for i in range(len(Trange))]
    #DT=[D_var[tr]*Trange[tr] for tr in range(1,len(Trange))]
    #D_theo0=[Dens[0][0]]+[Dens[0][0]**(1/2)*p_var*(dt)**(-DIM_Numb/4) for dt in DT]
    #plt.loglog(Trange,D_theo0,ls='dashed',color='black',label='Theoretical Particle Density')

    Ttheo=np.linspace(Trange[0],Trange[-1],10**2)

    d0=Dens[0][0]
    d1=Dens[1][0]


    DO=d0,d1 #initial densities for each particle type
    k2=spopt.curve_fit(fct_fit,Trange,Dens[0]+Dens[1])[0][0]


    Dpart_theo=d0/(1+d0*k2*Ttheo)
    
    DLine=d0*Ttheo[1:]**(-0.5)/Ttheo[1]**(-0.5)
    DLine2=d0*Ttheo[1:]**(-1)/Ttheo[1]**(-1)
    #DO=np.array(Dens[0][1:]),np.array(Dens[1][1:])
    D=spopt.curve_fit(fct_fit2,Trange[1:],Dens[0][1:]+Dens[1][1:],bounds=(0,np.inf))[0][0]
    D_theo0=[d0]+[d0**(1/2)*p_var*(D*dt)**(-DIM_Numb/4) for dt in Ttheo[1:]]
    #graph are not in log/log anymore, need to put it manually
    plt.scatter(Trange,Dens[0],color='black',label='Particle Density')
    #plt.plot(Ttheo,Dpart_theo,color='purple',label='Particle Theo')
    plt.plot(Ttheo[1:],DLine,color='grey',label='t^-1/2')
    plt.plot(Ttheo[1:],DLine2,color='navy',label='t^-1')
    #plt.plot(Ttheo,D_theo0,color='grey',label='Particle FIT')

    if Dens[0]!=Dens[1]: 
        plt.plot(Trange,Dens[1],color='blue',label='Anti-Particle Density')
        Dantipart_theo=d1/(1+d1*k2*Ttheo)
        plt.plot(Ttheo,Dantipart_theo,color='green',label='Antiparticle Theo')
        D_theo1=[d1]+[d1**(1/2)*p_var*(D*dt)**(-DIM_Numb/4) for dt in Ttheo[1:]]
        plt.plot(Ttheo,D_theo1,color='grey',label='AntiParticle FIT')

    # Set plot attributes
    plt.grid()
    plt.xlabel('Time(s)')
    plt.ylabel('Density')
    plt.title('Particle Densities in a '+str(DIM_Numb)+'D box\n as a function of Time')
    plt.legend()
    plt.show()

