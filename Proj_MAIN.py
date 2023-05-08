
import numpy as np
import time
from itertools import product,combinations
from collections import Counter
import Proj_Density

DENS_FCT=Proj_Density.Denfct


#Initialise Global Variables
DIM_Numb=0
Numb_part,Numb_antipart=0,0
Ntot=[Numb_part,Numb_antipart]
V0,L=0,0
TRACKING,Xini,COLPTS,SYSTEM,HIST,Vflipinfo=[],[],[],[],[],[]

DT=['',0.1,0.1,0.05]
DistList=['',0.1,0.2,8]

BOUNDARY_COND=1 #0/1: ['Periodic','Bounce']
ROUNDDIGIT=6





def ROUND(x):
    """
    Rounds the given number 'x' to the number of digits specified by 'ROUNDDIGIT'.
    :param x: the number to be rounded
    :return: the rounded number
    """
    return round(x, ROUNDDIGIT)

def Listn1n2(n1, n2):
    """
    Returns a list containing two empty lists, with lengths 'n1' and 'n2' respectively.
    :param n1: the length of the first list
    :param n2: the length of the second list
    :return: a list containing two empty lists with lengths 'n1' and 'n2' respectively
    """
    return [[[] for n in range(n1)], [[] for n in range(n2)]]

def COUNTFCT(LIST, elem):
    """
    Counts the number of times 'elem' appears in the second element of the tuples in 'LIST'.
    :param LIST: a list of tuples, where each tuple has two elements
    :param elem: a tuple of length 2
    :return: the number of times 'elem' appears in the second element of the tuples in 'LIST'
    """
    # Convert the second element of each tuple in 'LIST' to a tuple.
    for lis in LIST:
        lis[1] = tuple(lis[1])
    # Convert 'elem[1]' to a tuple.
    elem[1] = tuple(elem[1])
    # Convert 'elem' to a tuple.
    elem = tuple(elem)
    if len(LIST) > 0: # If 'LIST' is not empty, count the number of times 'elem' appears in the second element of the tuples in 'LIST'.
        cnt = Counter(map(tuple, LIST))
        return cnt[elem]
    else: # If 'LIST' is empty, return 0.
        return 0

    

def BOUNDS(xi,xf,V,t,id,p):
    """
    Computes the boundaries for a particle based on its position and velocity.

    Args:
    - xi (array): The initial position of the particle.
    - xf (array): The final position of the particle.
    - V (array): The velocity of the particle.
    - t (float): The current time.
    - id (int): The ID of the particle.
    - p (int): The ID of the process that is executing the function.

    Returns:
    - A list with the following elements:
        - Xini (array): The initial position of the particle.
        - Xinter (array): The intermediate positions of the particle (if any).
        - Xfin (array): The final position of the particle.
        - V (array): The velocity of the particle.
        - t_list (list): A list with the time values at which the particle reaches an intermediate position.
        - id (int): The ID of the particle.
        - NZ (int): The number of intermediate positions.
    """
    Xini=xi
    Xfin=xf
    Xinter=[]
    t_list=[t]
    
    # Check if the final position is inside the boundaries
    if (0<xf).all() and (xf<L).all() :
        return([Xini,Xinter,Xfin,V,t_list,id,0])
    
    # Initialize parameters
    t_params=np.zeros(DIM_Numb)
    x_params=np.zeros((DIM_Numb,2))

    # Check if the particle hits any boundary
    for d in range(DIM_Numb):
        xd=xf[d]
        if 0>=xd:# Check if the particle hits the lower boundary
            b=xd-V[d]*t
            t0=ROUND(-b/V[d])
            if t-dt<=t0<=t:
                if xd==0:
                    xd+=L*10**(-9) 
                t_params[d]=t0
                if BOUNDARY_COND==0:
                    x_params[d][0],x_params[d][1]=0,L 
                    xf[d]=L-abs(xd)
                else:
                    x_params[d][0],x_params[d][1]=0,0 
                    xf[d]=abs(xd)
        elif xd>=L:# Check if the particle hits the upper boundary
            b=xd-V[d]*t
            t0=ROUND((L-b)/V[d])
            if t-dt<=t0<=t:
                t_params[d]=t0
                if xd==L:
                    xd+=10**(-9)
                if BOUNDARY_COND==0:
                    x_params[d][0],x_params[d][1]=L,0
                    xf[d]=xd-L
                else:
                    x_params[d][0],x_params[d][1]=L,L
                    xf[d]=2*L-(xd)
    
    # Check if the particle hits any boundary and get the number of hits
    NZ=np.count_nonzero(t_params)
    
    # Compute intermediate positions if the particle hits any boundary
    if NZ>0:
        Xinter=np.zeros((NZ,2,DIM_Numb))
        d_prev=0
        for nz in range(NZ):
            mask=np.where((t_params>0)&(t_params!=np.inf))
            d_arg=mask[0][t_params[mask].argmin()]
            t_val=t_params[d_arg]
            t_list.append(t_val)
            if nz==0:
                Xinter[nz][0]=np.where((Xini+V*dt>0)&(Xini+V*dt<L),Xini+V*dt,Xini)
                Xinter[nz][1]=np.where((Xini+V*dt>0)&(Xini+V*dt<L),Xini+V*dt,Xini)
            else:
                val0=Xinter[nz-1][0]+V*(t_val-t_list[-2])
                Xinter[nz][0]=np.where((val0>0)&(val0<L),val0,Xinter[nz-1][0])
                val1=Xinter[nz-1][1]+V*(t_val-t_list[-2])
                Xinter[nz][1]= np.where((val1>0)&(val1<L),val1,Xinter[nz-1][1])

                Xinter[nz][0][d_prev]=Xinter[nz][1][d_prev]

            Xinter[nz][0][d_arg]=x_params[d_arg][0]
            Xinter[nz][1][d_arg]=x_params[d_arg][1]
            t_params[d_arg]=np.inf
            d_prev=d_arg

            Vflipinfo[p][id].append(d_arg)

    return([Xini,Xinter,Xfin,V,t_list,id,NZ])


def dist(t, Da, Db):
    """
    Compute the distance between two objects moving in a straight line.

    Args:
    - t (float): the time elapsed since the objects started moving.
    - Da (float): the velocity of object A.
    - Db (float): the velocity of object B.

    Returns:
    - d (float): the distance between the two objects at time t.
    """
    d = np.sum(np.sqrt((Da * t + Db) ** 2))
    return d


def MINIMISE(difA, difB, t, Sa, Sb):
    """
    This function takes in five arguments, `difA`, `difB`, `t`, `Sa`, and `Sb`, and returns a float value.

    Args:
    - difA (numpy array): an array containing the differences between the x-component velocities of two particles.
    - difB (numpy array): an array containing the differences between the x-component positions of two particles.
    - t (float): a time value.
    - Sa (numpy array): an array containing the x-component of the acceleration of particle a.
    - Sb (numpy array): an array containing the x-component of the acceleration of particle b.

    Returns:
    - A float value representing the time at which two particles meet, or -1 if they do not meet.
    """

    # Find the possible time values where the two particles meet.
    Tsol = np.where(difA != 0, -difB / difA, -1)
    
    # Filter the time values that are within the acceptable range.
    Tsol = Tsol[(0.5*(Sa*Tsol+Sb) > 0) & (0.5*(Sa*Tsol+Sb) < L) & (Tsol < t) & (Tsol > t - dt)]
    
    # If there are possible meeting times, return the minimum value.
    if Tsol.size > 0:
        return np.round(np.min(Tsol), ROUNDDIGIT)
    
    # If there are no possible meeting times and there is only one dimension, return -1.
    elif DIM_Numb == 1:
        return -1
    
    # If there are no possible meeting times and there are multiple dimensions,
    # find the earliest possible meeting time among the dimensions.
    else:
        if (difA == 0).all():
            return t - 0.5 * dt
        else:
            tmini = t
            for dim in range(DIM_Numb):
                if difA[dim] != 0:
                    td = -difB[dim] / difA[dim]
                    if td < t - dt:
                        tminid = t - 0.99 * dt
                    else:
                        tminid = t * 0.99
                    if tminid < tmini:
                        tmini = tminid
            return ROUND(tmini)

def INTERCHECK(a1, b1, p1, a2, b2, p2, t, z1, z2, Tstart, Tend):
    """
    Check if the line segments defined by points a1, b1 and a2, b2 intersect
    during the time interval [Tstart, Tend] and return information about the
    intersection.

    Args:
    - a1, b1: numpy arrays with the coordinates of the endpoints of the first segment.
    - p1: an integer representing the index of the first segment.
    - a2, b2: numpy arrays with the coordinates of the endpoints of the second segment.
    - p2: an integer representing the index of the second segment.
    - t: current time.
    - z1, z2: numpy arrays with the velocities of the first and second segment, respectively.
    - Tstart, Tend: start and end times of the time interval.

    Returns:
    - If the segments intersect during the time interval, returns a list with the following elements:
      - 1 if the intersection point is inside the first segment, 2 if it is inside the second segment.
      - tmini: time of intersection.
      - xo: coordinates of intersection point.
      - z1, z2: velocities of the first and second segment, respectively.
    - If the segments do not intersect during the time interval, returns [0].
    """
    # Calculate the difference of the slopes and intercepts of the two lines
    difA = np.array(a1 - a2, dtype=float)
    difB = np.array(b1 - b2, dtype=float)

    # Calculate the sum of the slopes and intercepts of the two lines
    SA = np.array(a1 + a2, dtype=float)
    SB = np.array(b1 + b2, dtype=float)

    # Check if the lines are parallel
    if (difA != 0).any():
        # Find the time t for the intersection point with minimum distance between the lines
        tmini = MINIMISE(difA, difB, t, SA, SB)

        # Calculate the minimum distance D between the lines at the intersection point
        D = dist(tmini, difA, difB)

        # Calculate the coordinates of the intersection point
        xo = np.array((tmini * (a2 + a1) + b2 + b1) / 2, dtype=np.float64)
        xo = np.round(xo, decimals=ROUNDDIGIT)

        # Check if the intersection point is valid
        if Tstart <= tmini <= Tend and D <= Dist_min and (xo < L).all() and (xo > 0).all() and tmini > 0:
            return [2 - 1 * (p1 == p2), tmini, xo, z1, z2]

    # If the lines are parallel or do not intersect during the time interval, return [0]
    return [0]


def ANNIHILATE(FirstAnn,F):
    """Remove the particles involved in a collision and update tracking information and collision points.

    Args:
    - FirstAnn (list): List containing information about the collision, including time, position, and IDs of the particles involved, as well as other variables used for tracking.
    - F (list): List of dictionaries containing information about the particles in the simulation, indexed by particle type and ID.

    Returns:
    - F (list): Updated list of dictionaries containing information about the particles in the simulation.
    """

    global COLPTS
    global Ntot

    # Extract information about the collision
    ti,xo,z1,z2,p1,id1,p2,id2=FirstAnn[1:]

    # Remove the particles involved in the collision from their respective particle lists
    for s in range(len(SYSTEM[p1])):
        if SYSTEM[p1][s].ID==id1:
            SYSTEM[p1].remove(SYSTEM[p1][s])
            break
    for s in range(len(SYSTEM[p2])):
        if SYSTEM[p2][s].ID==id2:
            SYSTEM[p2].remove(SYSTEM[p2][s])
            break

    # If the particles have a history of collisions, update the tracking information
    if z1 > 0:
        for zi in range(z1):
            targs=F[p1][id1][4]
            xinterargs=F[p1][id1][1]
            TRACKING[p1][id1].extend([[targs[zi+1],xinterargs[zi][0]],['T','X'],[targs[zi+1],xinterargs[zi][1]]])
            if DIM_Numb==3:
                ALL_TIME.extend(targs[1:])
    if z2 > 0:
        for zi in range(z2):
            targs=F[p2][id2][4]
            xinterargs=F[p2][id2][1]
            TRACKING[p2][id2].extend([[targs[zi+1],xinterargs[zi][0]],['T','X'],[targs[zi+1],xinterargs[zi][1]]])
            if DIM_Numb==3:
                ALL_TIME.extend(targs[1:])

    # Update the tracking information for the current collision
    TRACKING[p1][id1].append([ti,xo])
    TRACKING[p2][id2].append([ti,xo])
    if DIM_Numb==3:
        ALL_TIME.append(ti)

    # Remove the information about the particles involved in the collision from the collision point list and decrement the total number of particles
    F[p1][id1], F[p2][id2] = [], []
    COLPTS.append([ti,xo,2])
    Ntot[0], Ntot[1] = Ntot[0] - 1, Ntot[1] - 1

    return(F)

def CHECK_time(elem,t, start_stop):
    """
    Check if a given element is within the time range specified by a current time t and a time window of length dt.
    If the element is an array, the function returns the minimum or maximum value of the elements that are within the time
    window depending on the value of the start_stop parameter.
    
    Args:
    elem: An array of elements or a float.
    t: A float representing the current time.
    start_stop: A string indicating whether to return the minimum or maximum value of elem within the time window.
        Should be 'start' or 'end'.
    
    Returns:
    A float representing the value of elem that is within the time window specified by t and dt.
    If elem is an array, returns either the minimum or maximum value of the elements that are within the time window
    depending on the value of the start_stop parameter. If there are no elements within the time window, returns the time
    at the start or end of the window depending on the value of the start_stop parameter.
    """
    if start_stop == 'start':
        compare_fct = [max, np.max]
        t_extr = t - dt
    elif start_stop == 'end':
        compare_fct = [min, np.min]
        t_extr = t
    
    if isinstance(elem, np.ndarray):
        elem = elem[(elem >= t - dt) & (elem <= t)]
        if elem.size > 1:
            elem = compare_fct[1](elem)
        elif elem.size == 1:
            elem = elem[0]
        else:
            elem = t_extr
    elif not isinstance(elem, float):
        elem = t_extr
    
    return elem


def timestat_end(B1, B2, t, z1, z2, t1PARA, t2PARA):
    """
    Computes the start and end times for a given time t and its corresponding
    indices z1 and z2 within the time arrays t1PARA and t2PARA, respectively.
    These indices are used to obtain the start and end times tstart and tend for
    the corresponding segments of the signal.

    Args:
    - B1: numpy.ndarray. First signal array.
    - B2: numpy.ndarray. Second signal array.
    - t: float. Time value.
    - z1: int. Index of t in the first signal time array t1PARA.
    - z2: int. Index of t in the second signal time array t2PARA.
    - t1PARA: numpy.ndarray. First signal time array.
    - t2PARA: numpy.ndarray. Second signal time array.

    Returns:
    A tuple of floats (tstart, tend) corresponding to the start and end times for
    the segments of the signals B1 and B2 that correspond to the time t.
    """

    # Determine the start and end times for the segments of the signal B1
    # that correspond to the time t.
    if z1 != 0:
        tstart1 = t1PARA[z1]
    if z1 != len(B1) - 1:
        tend1 = t1PARA[z1 + 1]

    # Determine the start and end times for the segments of the signal B2
    # that correspond to the time t.
    if z2 != 0:
        tstart2 = t2PARA[z2]
    if z2 != len(B2) - 1:
        tend2 = t2PARA[z2 + 1]

    # Determine the start time tstart for the corresponding segments of B1 and B2.
    if z1 == 0:
        if z2 == 0:
            tstart = t - dt
        else:
            tstart = tstart2
    else:
        if z2 == 0:
            tstart = tstart1
        else:
            tstart = max(tstart1, tstart2)

    # Determine the end time tend for the corresponding segments of B1 and B2.
    if z1 == len(B1) - 1:
        if z2 == len(B2) - 1:
            tend = t
        else:
            tend = tend2
    else:
        if z2 == len(B2) - 1:
            tend = tend1
        else:
            tend = min(tend1, tend2)

    return (tstart, tend)


def SIGNFLIP(a, p, id, z):
    """
    Flip the signs of selected elements of the input array a, based on the indices provided in Vflipinfo[p][id][:z].

    Args:
    a (array-like): The input array to modify.
    p (int): Index of the first-level list in Vflipinfo that contains the indices to be flipped.
    id (int): Index of the second-level list in Vflipinfo that contains the indices to be flipped.
    z (int): Index in the second-level list of the last index to be flipped.

    Returns:
    array-like: The modified input array, with the signs of the selected elements flipped.

    """
    for find in range(z-1+1):
        a[Vflipinfo[p][id][find]] *= -1
    return a

def GetCoefs(F):
    """Calculate coefficients of the interpolation polynomials for each particle.

    Args:
    - F (list): List of Fock coefficients for particles and antiparticles.

    Returns:
    - COEFS (list): List of coefficients of the interpolation polynomials for each particle.
                    Each element in the list is a list containing the following elements:
                    [a, b, ts, id, ends]

    The interpolation polynomials for each particle are defined as follows:
        f(t) = a * t + b[0]         if ends==0
        f(t) = a * t + b[0] + ... + b[ends] * (t - ts[1]) * ... * (t - ts[ends]) if ends>0
    """
    COEFS = [[[] for _ in range(Numb_part)], [[] for _ in range(Numb_antipart)]]
    for p in range(2):
        # Get all non-empty elements of F for current p
        non_empty = [f for f in F[p] if f!=[]]

        # Extract required values from non-empty F elements 
        xis = np.array([f[0] for f in non_empty], dtype=object)
        xinters = np.array([f[1] for f in non_empty], dtype=object)

        xfs = np.array([f[2] for f in non_empty], dtype=object)
        Vs = np.array([f[3] for f in non_empty])
        ts = np.array([f[4] for f in non_empty], dtype=object)
        ids = np.array([f[5] for f in non_empty])
        ends = np.array([f[6] for f in non_empty])

        for ind0 in range(len(non_empty)):
            a = Vs[ind0]
            if ends[ind0] == 0:
                b = [xfs[ind0] - a * float(*ts[ind0])]
            else:
                b = []
                if BOUNDARY_COND == 0:
                    for r in range(ends[ind0]):
                        b.append(xinters[ind0][r][0] - a * float(ts[ind0][1+r]))
                    b.append(xfs[ind0] - a * float(ts[ind0][0]))
                else:
                    A = np.copy(a)
                    for r in range(ends[ind0]):
                        b.append(xinters[ind0][r][0] - A * float(ts[ind0][1+r]))
                        A[Vflipinfo[p][ids[ind0]][r]] *= -1
                    b.append(xfs[ind0] - A * float(ts[ind0][0]))

            COEFS[p][ids[ind0]] = [a, b, ts[ind0], int(ids[ind0]), ends[ind0]]
    return COEFS


def LOOPCHECK(COEFS,t,MODE,p=None):
    """
    Check for particle interactions between pairs of particles, given their coefficients and time of collision.

    Parameters:
        COEFS (list): A list of particle coefficients in the form [coefs_p1, coefs_p2], where coefs_p1 and coefs_p2
                      are lists of coefficients for particles in partitions p1 and p2, respectively. Each coefficient
                      is in the form [A, B, tparams, id, ends], where A is a list of initial particle conditions, B is
                      a list of boundary conditions, tparams is a list of time parameters, id is the particle ID, and
                      ends is a flag indicating if the particle is an endpoint.
        t (float): The time at which the particles are expected to collide.
        MODE (int): A flag indicating the mode of operation. 0 for annihilation, 1 for collision, and any other value
                    for gathering history.
        p (int, optional): The partition ID in which to search for collisions. Only used if MODE is 1.

    Returns:
        If MODE is 0, returns a tuple (MINI, Annilnumb), where MINI is a list containing information about the
        annihilation event with the earliest time of collision, and Annilnumb is the number of annihilation events.
        If MODE is 1, returns a tuple (BOUNCE, Colnumb), where BOUNCE is a list containing information about the
        collisions, and Colnumb is the number of collisions.
        If MODE is any other value, returns a list of lists Hist, where Hist[i] contains a list of particle IDs that
        particle i has interacted with.
    """
    if MODE==0:# Annihilation mode
        MINI=[]
        Annilnumb=0
        p1,p2=0,1    
        C1=COEFS[p1]
        C2=COEFS[p2]
        Iteration=product(C1,C2)
        #product(C1,C2)
    elif MODE==1:# Collision mode
        BOUNCE=[]
        Colnumb=0
        C2,C1=COEFS[p],COEFS[p]
        p1,p2=p,p 
        Iteration=combinations(C1,2)
    else: # Gathering history mode
        Hist=[[] for _ in range(len(COEFS[0]))]
        p1,p2=0,1
        C1=COEFS[p1]
        C2=COEFS[p2]
        Iteration=product(C1,C2)

    for coefs1, coefs2 in Iteration:    
        if not coefs1 or not coefs2 :
            continue
        if (coefs1[0]==coefs2[0]).all():
            continue


        A1,B1,t1params,id1=coefs1[:4]
        A2,B2,t2params,id2=coefs2[:4]
        if MODE==0:
            if id2 not in HIST[id1]:
                continue
        for z1, b1 in enumerate(B1):
            if BOUNDARY_COND==1 and z1!=0:
                a1=SIGNFLIP(np.copy(A1),p1,id1,z1)
            else:
                a1=A1
            for z2, b2 in enumerate(B2):
                if BOUNDARY_COND==1 and z2!=0:
                    a2=SIGNFLIP(np.copy(A2),p2,id2,z2)
                else:
                    a2=A2

                tstart, tend = timestat_end(B1,B2,t,z1,z2,t1params,t2params)
                INTER = INTERCHECK(a1,b1,p1,a2,b2,p2,t,z1,z2,tstart,tend)
                if MODE==0:
                    if INTER[0] == 2:
                        Annilnumb += 1
                        if not MINI or INTER[1] < MINI[1]:
                            MINI = [*INTER, p1, id1, p2, id2]
                elif MODE==1:
                    if INTER[0] ==1:
                        colinfo = [INTER[1], INTER[2], 3]
                        if COUNTFCT(BOUNCE,colinfo)==0:
                            BOUNCE.append(colinfo)
                            Colnumb += 1
                else:
                    if INTER[0] != 0:
                        if id2 not in Hist[id1]:
                            Hist[id1].append(id2)

    if MODE==0:
        return(MINI,Annilnumb)
    elif MODE==1:
        return(BOUNCE,Colnumb)
    else:
        return(Hist)



def COLLISIONS(F,t,MODE,kill_list):
    '''
    This function simulates the collisions between particles in the system.

    Parameters:
        F (list): List of coefficients that describe the state of the system.
        t (float): The current time.
        MODE (int): The mode of the simulation (0: no collisions, 1: collisions allowed).
        kill_list (list): A list of particles that should be removed from the simulation.

    Returns:
        If MODE=0, the function returns the updated state of the system (F) and the kill list.
        If MODE=1, the function returns the updated state of the system (F).
    '''
    global COLPTS,HIST
    if MODE==0:
        Annilnumb=-1
        COEFS=GetCoefs(F)  
        HIST=LOOPCHECK(COEFS,t,-1)
        while Annilnumb!=0:
            COEFS=GetCoefs(F)  
            MINI,Annilnumb=LOOPCHECK(COEFS,t,0)
            if MINI!=[]:
                if MINI[0]==2:
                    F=ANNIHILATE(MINI,F)
                    kill_list.append([COEFS[MINI[5]][MINI[6]],MINI[5],MINI[1]])
                    kill_list.append([COEFS[MINI[7]][MINI[8]],MINI[7],MINI[1]])
                    Annilnumb-=1
        return(F,kill_list)
    else:
        COEFS=GetCoefs(F)  
        BOUNCE0,Colnumb0=LOOPCHECK(COEFS,t,1,0)
        BOUNCE1,Colnumb1=LOOPCHECK(COEFS,t,1,1)
        BOUNCE=BOUNCE0+BOUNCE1
        Colnumb=Colnumb0+Colnumb1
        for part in kill_list:
            coefs1,p1,tdeath=part
            A1,B1,t1params,id1=coefs1[:4]
            p2=p1
            for coefs2 in COEFS[p2]:
                if not coefs2 or (coefs1[0]==coefs2[0]).all():
                    continue

                A2,B2,t2params,id2=coefs2[:4]
                for z1, b1 in enumerate(B1):
                    if BOUNDARY_COND==1 and z1!=0:
                        a1=SIGNFLIP(np.copy(A1),p1,id1,z1)
                    else:
                        a1=A1
                    for z2, b2 in enumerate(B2):
                        if BOUNDARY_COND==1 and z2!=0:
                            a2=SIGNFLIP(np.copy(A2),p2,id2,z2)
                        else:
                            a2=A2
                        tstart,tend=timestat_end(B1,B2,t,z1,z2,t1params,t2params)
                        if tend>tdeath:
                            tend=tdeath
                        INTER=INTERCHECK(a1,b1,p1,a2,b2,p2,t,z1,z2,tstart,tend)
                        if INTER[0]==1:
                            colinfo=[INTER[1],INTER[2],3]
                            if COUNTFCT(BOUNCE,colinfo)==0:
                                BOUNCE.append(colinfo)
                                Colnumb+=1
            for part2 in kill_list:
                coefs2,p2,tdeath2=part2
                if (coefs1[0]!=coefs2[0]).all():
                    coefs2,p2,tdeath2=part2
                    A2,B2,t2params,id2=coefs2[:4]
                    for z1, b1 in enumerate(B1):
                        if BOUNDARY_COND==1 and z1!=0:
                            a1=SIGNFLIP(np.copy(A1),p1,id1,z1)
                        else:
                            a1=A1
                        for z2, b2 in enumerate(B2):
                            if BOUNDARY_COND==1 and z2!=0:
                                a2=SIGNFLIP(np.copy(A2),p2,id2,z2)
                            else:
                                a2=A2
                            tstart,tend=timestat_end(B1,B2,t,z1,z2,t1params,t2params)
                            if tend>min(tdeath,tdeath2):
                                tend=min(tdeath,tdeath2)
                            INTER=INTERCHECK(a1,b1,p1,a2,b2,p2,t,z1,z2,tstart,tend)
                            if INTER[0]==1:
                                colinfo=[INTER[1],INTER[2],3]
                                if COUNTFCT(BOUNCE,colinfo)==0 :
                                    BOUNCE.append(colinfo)
                                    Colnumb+=1
        COLPTS+=BOUNCE
        return(F)
                

rng=np.random.default_rng()


def Pos_fct(min_value, max_value):
    """Generate a random position vector with DIM_Numb dimensions and values between min_value and max_value (both inclusive)

    Args:
        min_value (float or int): The minimum value for each dimension of the position vector
        max_value (float or int): The maximum value for each dimension of the position vector

    Returns:
        A list of length DIM_Numb representing the position vector
    """
    # Create a list of DIM_Numb random values between min_value and max_value (both inclusive)
    pos = [ROUND(rng.uniform(min_value, max_value)) for n in range(DIM_Numb)]
    return pos


def CHECKstartPos(testPOS):
    """
    Checks if the initial position of a particle is valid.

    Args:
    testPOS (List[float]): The initial position of the particle.

    Returns:
    True if the position is valid, False otherwise.
    """
    if (testPOS<=0).any() or (testPOS>=L).any():
        return(True)
    if len(Xini)>0:
        XiArray=np.array(Xini)
        argmini=(np.sum((testPOS-XiArray)**2)).argmin()
        dmin=np.sqrt(np.sum((testPOS-Xini[argmini])**2))
        return(dmin>Dist_min)
    else:
        return(True)

Epsilon=1e-12#np.finfo(np.float).eps #exclude 0 from random.uniform

def GEN_X():
    """Generates a random position within a given distance of minimum separation from other particles.

    Args:
    - Epsilon (float): minimum distance between particles.
    - L (int): maximum position value for each coordinate.
    - Dist_min (float): minimum distance allowed between generated position and existing positions.
    - Xini (List[List[int]]): list of existing particle positions.

    Returns:
    - List[int]: a new particle position that meets the minimum distance requirement from other particles.
    """
    # Generate a new position randomly within the bounds of L.
    POS = Pos_fct(Epsilon, L)

    # While the generated position is not far enough from existing positions,
    # generate a new position.
    while not CHECKstartPos(np.array(POS)):
        POS = Pos_fct(Epsilon, L)
    
    # Add the new position to the list of existing positions.
    Xini.append(POS)

    return(POS)


def GEN_V():
    """Generates a random velocity vector with components between -V0 and V0.

    Args:
    - V0 (float): maximum absolute velocity value for each component.
    - DIM_Numb (int): number of dimensions.
    - ROUNDDIGIT (int): number of decimal places to round velocity components to.

    Returns:
    - np.ndarray: a random velocity vector with values rounded to ROUNDDIGIT decimal places.
    """
    # Generate a random velocity vector with components between -V0 and V0.
    # Round each component to ROUNDDIGIT decimal places.
    return np.round(rng.uniform(-V0, V0, DIM_Numb), ROUNDDIGIT)

class Particle:
    """A class representing a particle in a simulation.

    Attributes:
    -----------
    par : int
        The parity of the particle, which determines its behavior when interacting with the boundary.
    ID : int
        A unique identifier for the particle.
    X : list of floats
        The position of the particle in the simulation space.
    V_av : list of floats
        The average velocity of the particle.
    
    Methods:
    --------
    DO(t):
        Performs a single step of the simulation for the particle.
    
    """
    def __init__(self,parity,id):
        self.par=parity
        self.ID=id
        X=GEN_X()
        self.X=X
        TRACKING[parity][id].append([0,X])
        self.V_av=GEN_V()
    def DO(self,t):
        global Ntot
        xi=self.X
        V=GEN_V()+self.V_av
        
        xf=np.round(xi+dt*V,ROUNDDIGIT)
        for x_ind,Xf_i in enumerate(xf):
            if Xf_i<0 :
                self.V_av[x_ind]=abs(self.V_av[x_ind])
            if Xf_i>L:
                self.V_av[x_ind]=-abs(self.V_av[x_ind])

        return(BOUNDS(xi,xf,V,t,self.ID,self.par))



def main(T,n1,n2,vo,l,Numb_Dimensions,BoundsCond,D,File_path_name=None):
    """
    Simulates the behavior of n1 particles and n2 antiparticles in a D dimensional box

    Args:
    T (float): The time range to simulate.
    D (int): The type of representation: draw just densities/0 or also draw trajectories/1
    n1 (int): The initial number of particles.
    n2 (int): The initial number of antiparticles.
    vo (float): The order of the velocities of the particles.
    l (float): The length of the box.
    Numb_Dimensions (int): The number of dimensions to simulate
    BoundsCond(int): The type of boundaries periodic/0 or hard/1
    File_path_name: Where to save the video if the simulation is 3D and D=1

    Returns:
    None: The function does not return anything, but it prints the total time of the simulation and 
    draws the points if D=1.

    """
    global dt,Dist_min,DIM_Numb,Numb_part,Numb_antipart,Ntot,V0,L,BOUNDARY_COND,TRACKING,SYSTEM,ALL_TIME,Vflipinfo

    BOUNDARY_COND=BoundsCond
    DIM_Numb=Numb_Dimensions
    Dist_min=float(DistList[DIM_Numb])
    dt=float(DT[DIM_Numb])

    if DIM_Numb==3:
        ALL_TIME=[]
    # Setting global variables
    Numb_part=n1
    Numb_antipart=n2
    Ntot=[Numb_part,Numb_antipart]
    V0=vo
    L=l
    TRACKING,Vflipinfo=Listn1n2(Numb_part,Numb_antipart),Listn1n2(Numb_part,Numb_antipart)

    SYSTEM=[[Particle(0,i) for i in range(Numb_part)],[Particle(1,i) for i in range(Numb_antipart)]]
    
    T=int(10*T)
    print('\nT=',T,' Ntot=',(Numb_part+Numb_antipart),'Spacial Dimensions:',DIM_Numb)
    

    time0=time.time()# Starting time for the simulation

    Dens=[[Ntot[0]/L**DIM_Numb],[Ntot[1]/L**DIM_Numb]] # Initializing density values

    if DIM_Numb==3:# Adds time steps to the time list for 3D simulations
        ALL_TIME.extend([ROUND(i*dt) for i in range(T)])
    print('Generating Points')

    # Simulation loop
    for ti in range(1,T):
        t=ROUND(ti*dt)
        # Initializing list to track particle positions at time t
        Xc=[[[] for n in range(Numb_part)],[[] for n in range(Numb_antipart)]]
        Vflipinfo=Listn1n2(Numb_part,Numb_antipart)

        # Updating positions for all particles and antiparticles
        for p in range(2):
            for s in SYSTEM[p]:
                Xc[s.par][s.ID]=s.DO(t)

         # Handle collisions between particles
        Xc,kill_list=COLLISIONS(Xc,t,0,[])
        Xc=COLLISIONS(Xc,t,1,kill_list)

        # Update particle positions and track their movement
        for p in range(2):
            for xc in Xc[p]:
                if not xc:
                    continue
                for s in SYSTEM[p]:
                    id_s=xc[-2]
                    if s.ID==id_s:
                        xf=xc[2]
                        s.X=xf
                        if xc[-1]>0:
                            targs=xc[4]
                            xinterargs=xc[1]
                            for nz in range(xc[-1]):
                                TRACKING[p][id_s].extend([[targs[nz+1],xinterargs[nz][0]],['T','X'],[targs[nz+1],xinterargs[nz][1]]])
                            if DIM_Numb==3:
                                ALL_TIME.extend(targs[1:])
                        TRACKING[p][id_s].append([t,xf])
                        break
        # Update the densities of particles
        Dens[0].append(Ntot[0]/L**DIM_Numb)
        Dens[1].append(Ntot[1]/L**DIM_Numb)

    Dt=round(time.time()-time0,2)

    #post-simulation operations depending on the input parameter D, which determines the type of output the function produces.
    
    if D==1: #if D is equal to 1, then the function produces a graphical output of the simulation results using the DRAW function.
        if DIM_Numb==3:
            import Proj_DRAW3D
            DRAW=Proj_DRAW3D.DRAW3D
            ALL_TIME=[*set(ALL_TIME)]
            ALL_TIME.sort()
            print('Total time:',Dt,'s')
            print('Drawing Points')
            DRAW(n1,n2,L,dt,BOUNDARY_COND,DIM_Numb,ALL_TIME,TRACKING,COLPTS,Dens,File_path_name)
        else:
            import Proj_DRAW
            DRAW=Proj_DRAW.DRAW
            print('Total time:',Dt,'s')
            print('Drawing Points')
            DRAW(T,dt,L,DIM_Numb,COLPTS,TRACKING,Dens)
    else: #the function produces a density plot of the simulation results using the DENS_FCT function
        print('Total time:',Dt,'s')
        Trange=np.linspace(0,(T-1)*dt,len(Dens[0]))
        DENS_FCT(DIM_Numb,Dens,Trange)



if __name__ == "__main__":
    '''
    main block of the script that calls the main function with parameters based on T,n1,n2,vo,l,DIM,Bounds,D,File_path_name variables.
    Args:
    T (float): The time range to simulate.
    D (int): The type of representation: draw just densities/0 or also draw trajectories/1
    n1 (int): The initial number of particles.
    n2 (int): The initial number of antiparticles.
    vo (float): The order of the velocities of the particles.
    l (float): The length of the box.
    DIM (int): The number of dimensions to simulate
    Bounds(int): The type of boundaries periodic/0 or hard/1
    File_path_name: Where to save the video if the simulation is 3D and D=1
    '''

    Answer=''
    while Answer not in ['r','c']:
        Answer=input('Would you like Recommended or custom settings?(r/c) ').lower()

    if Answer=='c':
        print('You have chosen custom settings, depending on the configuration the computation time might be long!')

    REPR,DIM,Bounds=2,0,2
    while REPR not in [0,1]:
        try:
            REPR=int(input('Would you like to only see densities or also plot trajectories?(0/1) '))
        except:
            print('Please choose between: ',[0,1])
    while DIM not in [1,2,3]:
        try:
            DIM=int(input('Please choose number of dimensions (1,2,3) '))
        except:
            print('Please choose between: ',[1,2,3])
    while Bounds not in [0,1]:
        try:
            Bounds=int(input('Please choose either periodic or hard boundaries (0/1) '))
        except:
            print('Please choose between: ',[0,1])
    
    
    

    Timeparam,N_part,N_antipart,V_param,Box_size=0,0,0,0,0
    

    PARAMS=['Time parameter ','number of particles ','number of antiparticles ','velocity parameter ','Box lenght '] #each parameter
    Paramtype=[float,int,int,float,float] #type of each parameter
    PARAMBounds=[(0,np.inf),(0,np.inf),(0,np.inf),(-np.inf,np.inf),(0,np.inf)] #Boundaries for valid values for each parameter
    if Answer=='r':
        RECOMMENDED=[[5,100,100,10,100],[10,15,15,5,20],[20,15,15,10,40]]
        PARAMS=RECOMMENDED[DIM-1]
    else:
        for para in range(len(PARAMS)):
            Z=0
            while Z==0:
                try:
                    Paramtest=Paramtype[para](input('Please Choose '+str(PARAMS[para])))
                    if PARAMBounds[para][0]<Paramtest<PARAMBounds[para][1]:
                        PARAMS[para]=Paramtest
                        Z=1
                    else:
                        print('Outside of parameter bounds',PARAMBounds[para],'Try again')
                    
                except:
                    print('Not a '+str(Paramtype[para])+' try again')

    if DIM==3:
        filename='/home/ethan/Documents/Code/PYTHON/UNIcode/M1/INFO/PROJ/Proj_animation'
        while filename=='':
            print('Enter location to store video in "filename" variable then restart the program')
        main(*PARAMS,DIM,Bounds,REPR,filename)
    else:
        main(*PARAMS,DIM,Bounds,REPR)




