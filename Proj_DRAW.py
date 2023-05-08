import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
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

def PLOT(t,x,Color,ax,Pt_type=None,style=None):
    """
    Plot the position of a particle or set of particles at a particular time.

    Args:
    t (float or np.ndarray): The time or times at which the particle position(s) are to be plotted.
    x (np.ndarray): A 1D or 2D numpy array representing the position of the particle(s) at the given time(s).
    Color (str): The color to use for the plotted points or lines.
    ax (matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D): The axis object to which the plot should be added.
    Pt_type (str, optional): A string indicating the type of particle to plot ('part' for particles, 'antipart' for antiparticles, 'point' for a single point in space). Default is None.
    style (str, optional): A string indicating the marker style to use for the plotted points. Default is None.

    Returns:
    None
    """
    if Pt_type=='part':
        Ms=None
        LW=3
    else:
        Ms=4
        LW=None
    if Pt_type=='point' or len(t)!=1 :
        if DIM_Numb==1:
            ax.plot(t,x[0],ms=Ms,marker=style,color=Color)
        elif DIM_Numb==2:
            ax.plot3D(x[0],x[1],t,linewidth=LW,ms=Ms,marker=style,color=Color)

DIM_Numb=0

def DRAW(Tt,dt,L,DIM_Numb0,COLPTS,TRACKING,Density):
    """
    Draw trajectories and particle densities of a simulation.

    Parameters:
    -----------
    Tt: int
        Number of time steps in the simulation.
    dt: float
        Size of time step in the simulation.
    L: float
        Size of the simulation box.
    DIM_Numb0: int
        Number of dimensions of the simulation.
    COLPTS: list
        List of colored points to plot. Each element is a tuple of two lists: the first one contains the x coordinates of
        the points, and the second one contains the y coordinates of the points. The third element is an integer that
        specifies the color of the points (0 for red, 1 for blue, 2 for black, and 3 for yellow).
    TRACKING: list
        List of particle trajectories to plot. Each element is a list of tuples, where each tuple contains two elements:
        the first one is the time at which the particle was tracked, and the second one is a tuple that contains the
        particle's position coordinates in each dimension.
    Density: list
        List of particle densities to plot. Each element is a tuple of two lists: the first one contains the time steps,
        and the second one contains the densities. The densities are represented as integers, and the time steps are
        represented as floats.
    DIM_Numb: int, optional (default=0)
        Number of dimensions of the simulation.

    Returns:
    --------
    None
    """
    global DIM_Numb,DO,p_var

    DIM_Numb=DIM_Numb0
    colors=['red','blue','black','yellow']
    STYLE=['_','_','o','o']
    LSTYLE=['-','-','None','None']
    Trange=np.linspace(0,(Tt-1)*dt,len(Density[0]))

    #fig,ax=Gen_fig()# Generate the figure and axes for plotting
    fig = plt.figure(0)
        
    if DIM_Numb==1:
        ax=fig.add_subplot(1,2,1)
        ax.set_xlabel('Position X')
        ax.set_ylabel('Time(s)')
    else:
        ax=fig.add_subplot(1,2,1,projection='3d')
        ax.set_xlabel('Position X')
        ax.set_ylabel('Position Y')
        ax.set_zlabel('Time(s)')


    ax.set_title('Trajectory of particles \n as a function of Time\n  in '+str(DIM_Numb)+' Dimensions')
    # Iterate over the particles and their trajectories to plot them
    for p in range(2):
        for tr in range(len(TRACKING[p])):
            T=[]
            X=[[] for i in range(DIM_Numb)]
            for elem in TRACKING[p][tr]:
                Ts=elem[0]
                if type(Ts)==float or type(Ts)==np.float64 or type(Ts)==int or type(Ts)==str:
                    T.append(Ts)
                else:
                    T.append(Ts[0])
                Xs=elem[1]
                if type(Xs)==str:
                    for xd in range(DIM_Numb):
                        X[xd].append('X')
                elif type(Xs[0])==float or type(Xs[0])==np.ndarray or type(Xs[0])==np.float64:
                    for xd in range(DIM_Numb):
                        X[xd].append(Xs[xd])
            st=0
            for i in range(T.count('T')+1):
                if T.count('T')>0:
                    end=T.index('T')
                    PLOT(T[st:end],[Xi[st:end] for Xi in X],colors[p],ax,'part')
                    T.remove('T')
                    for xd in range(DIM_Numb): 
                        X[xd].remove('X')
                    st=end
                else:
                    PLOT(T[st:],[Xi[st:] for Xi in X],colors[p],ax,'part')


    if DIM_Numb==1:    
        for tl in Trange:
            ax.vlines(tl,0,L,'black','dotted')
    for col in COLPTS:
        PLOT(col[0],col[1],colors[col[2]],ax,'point',STYLE[col[2]])


    lines = [Line2D([], [], color=colors[c], linewidth=3, marker=STYLE[c],linestyle=LSTYLE[c]) for c in range(4)]
    labels = ['Particule', 'Anti Particule', 'Annihilation','Collision']
    ax.legend(lines, labels)

    if Density: #Plot densities as a function of time in another window
        ax=fig.add_subplot(1,2,2)
        
        #ax.tight_layout()
        #ax.set_xlim(0,Trange[-1])
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('N')

        ax.set_title('Particle Densities in a '+str(DIM_Numb)+'D box\n as a function of Time')

        if DIM_Numb==1:
            p_var=(2*np.pi)**(-3/4)
        elif DIM_Numb==2:
            p_var=2**(-3/2)/np.pi
        elif DIM_Numb==3:
            p_var=2**(-7/4)*(np.pi)**(-5/4)

        
        Ttheo=np.linspace(Trange[0],Trange[-1],10**2)
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
        
        '''
        


        for d in range(len(Density)):
            ax.loglog(Trange,Density[d],color=colors[d],label=labels[d])
        D_theo0=[Density[0][0]]+[Density[0][1]*Trange[Tr]**(-DIM_Numb/4)/(Trange[1]**(-DIM_Numb/4)) for Tr in range(1,len(Trange))]
        ax.loglog(Trange,D_theo0,ls='dashed',color='black',label='Theoretical Particle Density')
        if Density[0][0]!=Density[1][0]:
            D_theo1=[Density[1][0]]+[Density[1][1]*Trange[Tr]**(-DIM_Numb/4)/(Trange[1]**(-DIM_Numb/4)) for Tr in range(1,len(Trange))]
            ax.loglog(Trange,D_theo1,ls='dashed',color='grey',label='Theoretical Anti-Particle Density')
        '''


    plt.show()
