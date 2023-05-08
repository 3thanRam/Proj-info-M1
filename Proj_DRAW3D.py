import numpy as np
from mayavi import mlab
from itertools import permutations
import imageio
from scipy.special import sph_harm


def CHECKBOUNDS(LISTS):
    """
    Checks if all elements in the input lists are within the bounds of [0, L].

    Args:
    lists (list): A list of lists containing numerical elements.
    L (float): The upper bound of the domain.

    Returns:
    bool: True if all elements in the input lists are within the domain [0, L], False otherwise.
    """
    for List0 in LISTS:
        for l in List0:
            if l>L or l<0:
                return(False)
    return(True)

L=0
def DRAW3D(n1,n2,L0,dt,BOUNDARY_COND,DIM_Numb,ALL_TIME,TRACKING,COLPTS,Density,File_path_name):
    """
    Save a 3D video Plot of a set of particles

    Parameters:
    -----------
    n1: float
        Initial number of particles
    n2: float
        Initial number of antiparticles
    L0: float
        Size of the simulation box.
    dt: float
        Size of time step in the simulation.
    BOUNDARY_COND: int
        Represents if boundaries are periodic(0) or hard(1)
    DIM_Numb: int
        Number of dimensions of the simulation.
    ALL_TIME: list
        collection of all the time values
    TRACKING: list
        List of particle trajectories to plot. Each element is a list of tuples, where each tuple contains two elements:
        the first one is the time at which the particle was tracked, and the second one is a tuple that contains the
        particle's position coordinates in each dimension.
    COLPTS: list
        List of colored points to plot. Each element is a tuple of two lists: the first one contains the x coordinates of
        the points, and the second one contains the y coordinates of the points. The third element is an integer that
        specifies the color of the points (0 for red, 1 for blue, 2 for black, and 3 for yellow).
    Density: list
        List of particle densities to plot. Each element is a tuple of two lists: the first one contains the time steps,
        and the second one contains the densities. The densities are represented as integers, and the time steps are
        represented as floats.
    File_path_name: str
        Path to where to save the video

    Returns:
    --------
    None
    """
    global L
    L=L0
    Time=ALL_TIME
    
    Nu=[n1,n2]
    PART=[[[[] for di in range(DIM_Numb)] for n in range(Nu[i])] for i in range(2)]


    #Particles gain extra time values during collisions which causes some particles to have more saved points than others and at different times
    # this loop adds the necesary points by interpolating the position before/after the time value to be added 
    # Future mod: Fix for periodic boundaries
    for p in range(2):
        for tr in range(len(TRACKING[p])):
            T_part=[]
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
                    T_part.extend(T[st:end])
                    for dim in range(DIM_Numb):
                        PART[p][tr][dim].extend(X[dim][st:end])
                    T.remove('T')
                    for xd in range(DIM_Numb): 
                        X[xd].remove('X')
                    st=end+1
                else:
                    T_part.extend(T[st:])
                    for dim in range(DIM_Numb):
                        PART[p][tr][dim].extend(X[dim][st:])
            tmax=T_part[-1]
            for t_all in Time:
                if len(T_part)==len(Time):
                    break
                if T_part.count(t_all)==0:
                    T_part.append(t_all)
                    T_part.sort()
                    t_ai=T_part.index(t_all)
                    if t_all>tmax:
                        for dim in range(DIM_Numb):
                            PART[p][tr][dim].extend(['D' for d in range(len(Time)-len(T_part)+1)])
                        break
                    if t_ai==0:
                        if len(PART[p][tr])>t_ai+2:
                            if CHECKBOUNDS([PART[p][tr][t_ai],PART[p][tr][t_ai+1]])==True:
                                interAv=[(PART[p][tr][ind][t_ai]+PART[p][tr][ind][t_ai+1])/2 for ind in range(len(PART[p][tr]))]
                            else:
                                interAv=[PART[p][tr][ind][t_ai] for ind in range(len(PART[p][tr]))]
                        else:
                            interAv=[PART[p][tr][ind][t_ai] for ind in range(len(PART[p][tr]))]
                    else:
                        if len(PART[p][tr][0])-1>t_ai:
                            if t_ai>0:
                                interAv=[(PART[p][tr][ind][t_ai-1]+PART[p][tr][ind][t_ai]+PART[p][tr][ind][t_ai+1])/3 for ind in range(len(PART[p][tr]))]
                            else:
                                interAv=[(PART[p][tr][ind][t_ai]+PART[p][tr][ind][t_ai+1])/2 for ind in range(len(PART[p][tr]))]
                        elif t_ai>0:
                            interAv=[(PART[p][tr][ind][t_ai-1]+PART[p][tr][ind][t_ai])/2 for ind in range(len(PART[p][tr]))]
                        else:
                            interAv=[PART[p][tr][ind][t_ai] for ind in range(len(PART[p][tr]))]

                    for ind in range(len(PART[p][tr])):
                        PART[p][tr][ind].insert(t_ai,interAv[ind])
                    



    
    oldTime=np.array([(i+1)*dt for i in range(len(Density[0]))])
    DENSITY=[[0 for i in Time],[0 for i in Time]]

    for t_ind,a_time in enumerate(Time):
        oldt_ind=(abs(a_time-oldTime[:])).argmin()
        DENSITY[0][t_ind]=Density[0][oldt_ind]
        DENSITY[1][t_ind]=Density[1][oldt_ind]

    COLPTS=[[colpts[0],*colpts[1],colpts[2]] for colpts in COLPTS]
    COLPTS=np.array(COLPTS)
    colors=['red','blue','black','yellow']
    
    
    # Set up the figure
    width = 320
    height = 320
    mlab.options.offscreen = True  # Stops the view window popping up and makes sure you get the correct size screenshots.

    fig = mlab.figure(size=(width, height))#    
    PartSize=5
    ExtTparam_min=3*dt
    ExtTparam_max=3*dt

    #Initialise the "pens" that will draw the particles/antiparticles
    PARTICLES=[mlab.points3d([0.5], [0.5], [0.5],color=(1,0,0),mode="sphere",scale_mode='none',scale_factor=PartSize,opacity=1).mlab_source for i in range(n1)]
    ANTI_PARTICLES=[mlab.points3d([0.5], [0.5], [0.5],color=(0,0,1),mode="sphere",scale_mode='none',scale_factor=PartSize,opacity=1).mlab_source for i in range(n2)]
    
    #create list of values which will be used to draw collisions/annihilations
    # Each added "pen" slows the program so it's faster to use the minimum possible and reuse them
    # Hence only initiliase a number equal to the max number of Annihilations/Collisions that will be drawn at the same time
    #(Note: Annihilations/Collisions are seperated even though the only difference is color because changing the color of a "pen" requires resetting it so it's just as fast and easier to describe by seperating them )
    if COLPTS.ndim>1:
        freqmaxCol=0
        freqmaxAnn=0
        Onlycol=COLPTS[np.where(COLPTS[:,-1]==3)]
        OnlyAnn=COLPTS[np.where(COLPTS[:,-1]!=3)]

        for timef in Time:
            freqc=Onlycol[(Onlycol[:,0]<=timef+ExtTparam_max)&(Onlycol[:,0]>=max(timef-ExtTparam_min,0))].shape[0]
            freqa=OnlyAnn[(OnlyAnn[:,0]<=timef+ExtTparam_max)&(OnlyAnn[:,0]>=max(timef-ExtTparam_min,0))].shape[0]

            freqmaxCol=max(freqc,freqmaxCol)
            freqmaxAnn=max(freqa,freqmaxAnn)
    else:
        freqmaxCol=0
        freqmaxAnn=0

    #I couldn't find a simple way to draw a star for Annihilations/Collisions so I decided to draw a harmonic function instead
    m,n=3,4
    r = 0.3
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]
    s = sph_harm(m, n, theta, phi).real
    s[s < 0] *= 0.97
    s /= s.max()
    x_v = PartSize*s*r * sin(phi) * cos(theta)
    y_v = PartSize*s*r * sin(phi) * sin(theta)
    z_v = PartSize*s*r * cos(phi)
    
    #Initialise the "pens" that will draw the Annihilations/Collisions
    COLLISION=[mlab.mesh(0*x_v,0*y_v, 0*z_v,scalars=s,color=(1,1,0)).mlab_source for i in range(freqmaxCol)]
    ANNIHILATION=[mlab.mesh(0*x_v,0*y_v, 0*z_v,scalars=s,color=(0,0,0)).mlab_source for i in range(freqmaxAnn)]

    # I couldn't find a clear way to display the box using the grid so I create visible boundaries by drawing the box
    radius=5 #particle radius to make bouncing of wall look more realistic
    R0 = np.arange(-radius,L+radius+0.1,0.1)
    R1 = (L+radius)*np.ones_like(R0)
    R_zero=(-radius)*np.ones_like(R0)
    RA=[R1,R1,R0]
    RB=[R_zero,R_zero,R0]
    RC=[R1,R0,R_zero]
    #plot lines of the box
    for ri in range(3): 
        mlab.plot3d(RB[2-ri],RB[1-ri],RB[-ri],line_width=0.01,color=(0,0,0),tube_radius=0.1)
        mlab.plot3d(RA[2-ri],RA[1-ri],RA[-ri],line_width=0.01,color=(0,0,0),tube_radius=0.1)
    for rb1,rb2,rb3 in permutations(RC, 3):
        mlab.plot3d(rb1,rb2,rb3,line_width=0.01,color=(0,0,0),tube_radius=0.1)



    TIME_DISP=mlab.text(0,0,str(0),width=0.6)#initiliase time and particle number display
    f = mlab.gcf()
    f.scene.disable_render = True #turning off rendering when not necesary greatly reduces computation time
    def update_scene(i):
        f.scene.disable_render = True
        theta=0.5*i*360/len(Time) #angle to spin camera to avoid blocking certain parts of the box
        mlab.view(azimuth=theta,elevation=theta/2,roll=theta,distance=3.5*L,focalpoint=(L/2,L/2,L/2))
        TIME_DISP.set(text='Time={0:.2f},Npart:{1},Nanti:{2}'.format(round(Time[i], 2),DENSITY[0][i],DENSITY[1][i]))#Display time and particle number

        if COLPTS.ndim>1:
            COL=COLPTS[np.where((COLPTS[:,0]<Time[i]+ExtTparam_max)&(COLPTS[:,0]>max(Time[i]-ExtTparam_min,0)))]
        else:
            COL=[]
        c,a=0,0
        #Reset Collisions/Annihilations "pens" to make them disappear
        for col in COL:
            if col[-1]==3:
                if (COLLISION[c].x!=0*x_v).all():
                    COLLISION[c].set(x=x_v+col[1],y=y_v+col[2],z=z_v+col[3])
                else:
                    COLLISION[c].reset(x=x_v+col[1],y=y_v+col[2],z=z_v+col[3])
                c+=1
            else:
                if (ANNIHILATION[a].x!=0*x_v).all():
                    ANNIHILATION[a].set(x=x_v+col[1],y=y_v+col[2],z=z_v+col[3])
                else:
                    ANNIHILATION[a].reset(x=x_v+col[1],y=y_v+col[2],z=z_v+col[3])
                a+=1
        #Plot Collisions/Annihilations
        for cempty in range(len(COLLISION)-c):
            if (COLLISION[-1-cempty].x!=0*x_v).all():
                COLLISION[-1-cempty].reset(x=0*x_v,y=0*y_v,z=0*z_v,color=(0,1,0))
        for aempty in range(len(ANNIHILATION)-a):
            if (ANNIHILATION[-1-aempty].x!=0*x_v).all():
                ANNIHILATION[-1-aempty].reset(x=0*x_v,y=0*y_v,z=0*z_v,color=(0,1,0))
        
        #Plot particles/antiparticles
        for ipart,partcl in enumerate(PARTICLES):
            if PART[0][ipart][0][i]=='D':
                if partcl.opacity!=0:
                    partcl.reset(x=[], y=[], z=[],color=(1,0,0),mode="sphere",opacity=0)
            else:
                partcl.set(x=[PART[0][ipart][0][i]],y=[PART[0][ipart][1][i]],z=[PART[0][ipart][2][i]],opacity=1)
        for ianti,antipart in enumerate(ANTI_PARTICLES):
            if PART[1][ianti][0][i]=='D':
                if antipart.opacity!=0:
                    antipart.reset(x=[], y=[], z=[],color=(0,0,1),mode="sphere",opacity=0)
            else:
                antipart.set(x=[PART[1][ianti][0][i]],y=[PART[1][ianti][1][i]],z=[PART[1][ianti][2][i]],opacity=1)
        f.scene.disable_render = False

    BO=['_Periodic','_BOUNCE']#text to add to name of file to distinguish boundary types
    writer = imageio.get_writer(File_path_name+BO[BOUNDARY_COND]+'.mp4', fps=15) #Writer object which can be used to write data and meta data to the specified file

    Dper=0
    #loop to get screenshot at each time value 
    for i in range(len(Time)):
        perc=int(100*i/len(Time))
        if perc>Dper and perc%5==0:
            Dper=perc
            print(Dper,'%',end='\r')
        update_scene(i)
        writer.append_data(mlab.screenshot(mode='rgb', antialiased=True))#add each screenshot to writer for it to convert to video
    writer.close()
    print('Done')

