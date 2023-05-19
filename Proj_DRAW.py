import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
import numpy as np
import scipy.optimize as spopt
import Proj_Density
import matplotlib.animation as animation
DENS_FCT=Proj_Density.Denfct

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
    x (np.ndarray): A 1D numpy array representing the position of the particle at the given time(s).
    Color (str): The color to use for the plotted points or lines.
    ax (matplotlib.axes.Axes ): The axis object to which the plot should be added.
    Pt_type (str, optional): A string indicating the type of particle to plot ('part' for particles, 'antipart' for antiparticles, 'point' for a single point in space). Default is None.
    style (str, optional): A string indicating the marker style to use for the plotted points. Default is None.

    Returns:
    None
    """
    if Pt_type=='part':
        Ms=None
    else:
        Ms=4
    if Pt_type=='point' or len(t)!=1 :
        ax.plot(t,x[0],ms=Ms,marker=style,color=Color)
    

def TRAJ_1D(ax,TRACKING,Trange,COLPTS):
    '''
    Plot trajectories of particles in for 1D case 
    '''
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


   
    #for tl in Trange:
    #    ax.vlines(tl,Linf[0],L[0],'black','dotted')
    for col in COLPTS:
        PLOT(col[0],col[1],colors[col[2]],ax,'point',STYLE[col[2]])


    lines = [Line2D([], [], color=colors[c], linewidth=3, marker=STYLE[c],linestyle=LSTYLE[c]) for c in range(4)]
    labels = ['Particule', 'Anti Particule', 'Annihilation','Collision']
    ax.legend(lines, labels)
    plt.show()







def TRAJ_2D(fig,ax,TRACKING,ALLtimes,Density,COLPTS):
    '''
    Plot trajectories of particles in for 2D case 
    '''
    Extrtime=0.8*dt #prolong display time to help visualisation
    anim_running=True
    IntervalVAR=0
    
    class Index:
        
        ind = 0

        def PlayPause(self, event):
            nonlocal anim_running
            if anim_running:
                ani.pause()
                anim_running = False
            else:
                ani.resume()
                anim_running = True
        def SpeedDown(self, event):
            nonlocal ani
            ani.event_source.interval *=1.25
            print('ms between frames',ani.event_source.interval,end='\r')

            ani.event_source.stop()
            #ani.frame_seq = ani.new_frame_seq() 
            ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=ani.event_source.interval,repeat=True,save_count=len(ALLtimes)-2)
            ani.event_source.start()
            event.canvas.draw()
            fig.canvas.draw()
        def SpeedUp(self, event):
            nonlocal ani
            ani.event_source.interval *=0.75
            print('ms between frames',ani.event_source.interval,end='\r')

            ani.event_source.stop()
            ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=ani.event_source.interval,repeat=True,save_count=len(ALLtimes)-2)
            ani.event_source.start()
            event.canvas.draw()
            fig.canvas.draw()



    callback = Index()

    SpeeddownAx = fig.add_axes([0.1, 0.05, 0.13, 0.075])
    Downbutton = Button(SpeeddownAx, 'Slow Down')
    Downbutton.on_clicked(callback.SpeedDown)

    PlaypauseAx = fig.add_axes([0.25, 0.05, 0.13, 0.075])
    Pbutton = Button(PlaypauseAx, 'Play/Pause')
    Pbutton.on_clicked(callback.PlayPause)

    SpeedupAx = fig.add_axes([0.40, 0.05, 0.13, 0.075])
    Upbutton = Button(SpeedupAx, 'Speed Up')
    Upbutton.on_clicked(callback.SpeedUp)

    



    def data_gen():
        '''
        Generates data about th e pints to be plotted at each instant
        '''
        t = data_gen.t
        cnt = 0
        while t < ALLtimes[-2]:
            P_TYPE=[[],[]]
            cnt+=1
            t=ALLtimes[cnt]
            for p in range(2):
                for tr in range(len(TRACKING[p])):
                    for elemID,elem in enumerate(TRACKING[p][tr]):
                        Tpart=elem[0]
                        if type(Tpart)!=str:
                            if ALLtimes[cnt]-Extrtime<=Tpart<=ALLtimes[cnt]+Extrtime:
                                P_TYPE[p].append(np.array([elem[1][0],elem[1][1]]))
                                if elemID!=len(TRACKING[p][tr])-1 and cnt!=len(ALLtimes)-1 :
                                    Tnextpart=TRACKING[p][tr][elemID+1][0]
                                    if type(Tnextpart)!=str:
                                        if ALLtimes[cnt+1]-Extrtime<=Tpart<=ALLtimes[cnt+1]+Extrtime and ALLtimes[cnt+1]-Extrtime<=Tnextpart<=ALLtimes[cnt+1]+Extrtime:
                                            del elem
                                break
            COLS,ANNILS=[],[]
            for colpts in COLPTS:
                interT=colpts[0]
                if abs(ALLtimes[cnt]-interT)<=Extrtime:
                    if colpts[-1]==3:
                        COLS.append([*colpts[1]])
                    else:
                        ANNILS.append([*colpts[1]])
            yield cnt,t,P_TYPE[0],P_TYPE[1],COLS,ANNILS

    data_gen.t = 0
    ax.set_title('Animation of Particle-Antiparticle\n Trajectories in a '+str(DIM_Numb)+'D box over Time')
    ax.set_xlim(Linf[0], L[0])
    ax.set_ylim(Linf[1], L[1]) 
    #ax.figure.canvas.draw()
    ax.grid()

    #initialise pens
    Particles = ax.scatter([], [],color='red')
    AntiParticles = ax.scatter([], [],color='blue')
    COLLISION = ax.scatter([], [],marker='*',color='yellow')
    ANNIHILATION = ax.scatter([], [],marker='*',color='black')
    
    

    oldTime=np.array([(i+1)*dt for i in range(len(Density[0]))])
    DENSITY=[[0 for i in ALLtimes],[0 for i in ALLtimes]]
    VOL=(L[0]-Linf[0])*(L[1]-Linf[1])
    for t_ind,a_time in enumerate(ALLtimes):
        oldt_ind=(abs(a_time-oldTime[:])).argmin()
        DENSITY[0][t_ind]=int(Density[0][oldt_ind]*VOL)
        DENSITY[1][t_ind]=int(Density[1][oldt_ind]*VOL)

    def Disp(tparam,Npartparam,Nantiparam):
        return('Time:'+str(round(tparam,2))+'s \n Numb Particles:'+str(Npartparam)+' \n Numb AntiParticles:'+str(Nantiparam))

    time_text = ax.text(Linf[0]+0.5,Linf[1]+0.5, Disp(0,Density[0][0],Density[1][0]), fontsize=7)

    PEN=[Particles,AntiParticles,COLLISION,ANNIHILATION,time_text]
    def run(data):
        '''
        Called by the FuncAnimation function and Draws the points given by the data_gen function 
        '''
        ti,t,Partdata, Antidata,Coldata,Annildata=data
        if len(Partdata)!=0:
            PEN[0].set_offsets(Partdata)
        if len(Antidata)!=0:
            PEN[1].set_offsets(Antidata)
        if len(Coldata)!=0:
            PEN[2].set_offsets(Coldata)
        if len(Annildata)!=0:
            PEN[3].set_offsets(Annildata)

        PEN[4].set_text(Disp(t,DENSITY[0][ti],DENSITY[0][ti]))
        return PEN

    # Create animation from series of images of matplotlib figures updated at each time instant
    IntervalVAR=int(750*ALLtimes[-1]/(len(ALLtimes)-2))
    ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=IntervalVAR,repeat=True,save_count=len(ALLtimes)-2)
    plt.show()

DIM_Numb=0
dt=0
def DRAW(Tt,dt0,Lparam,DIM_Numb0,COLPTS,TRACKING,Density,ALLtimes=None):
    """
    Draw trajectories and particle densities of a simulation.

    Parameters:
    -----------
    Tt: int
        Number of time steps in the simulation.
    dt: float
        Size of time step in the simulation.
    Lparam: float
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
    ALL_TIME: list
        collection of all the time values (only needed in 2D case)

    Returns:
    --------
    None
    """
    global DIM_Numb,DO,p_var,colors,STYLE,LSTYLE,L,Linf,dt
    dt=dt0
    Linf,L=Lparam
    DIM_Numb=DIM_Numb0
    colors=['red','blue','black','yellow']
    STYLE=['_','_','o','o']
    LSTYLE=['-','-','None','None']
    Trange=np.linspace(0,(Tt-1)*dt,len(Density[0]))

    #fig,ax=Gen_fig()# Generate the figure for plotting
    fig = plt.figure(0)
    
    #Plot densities as a function of time in one subplot
    ax2=fig.add_subplot(1,2,2)
    DENS_FCT(DIM_Numb0,Density,Trange,ax2)

    #Plot Trajectories as a function of time in another subplot
    ax=fig.add_subplot(1,2,1)
    if DIM_Numb==1:
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Position X')
        TRAJ_1D(ax,TRACKING,Trange,COLPTS)
    else:
        plt.subplots_adjust(bottom=0.225)
        ax.set_xlabel('Position X')
        ax.set_ylabel('Position Y')
        TRAJ_2D(fig,ax,TRACKING,ALLtimes,Density,COLPTS)

