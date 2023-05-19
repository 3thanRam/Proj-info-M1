# Proj-info-M1

Computer Sciences Project M1 : Brownian and ballistic annihilation processes

Project Description:

Simulates the trajectories and densities of particles and antiparticles when trapped in a box.

Necessary Parameters:  
-Input setting "r"/"c" Meaning Recommended values known to behave correctly or Custom lets you choose all the Specific parameters  
-Representation Type 0/1 Plot only densities(0) or also plot trajectories (1)  
-Dimensions Choose (1,2,3) Number of spatial dimensions (if 3D then an animation is saved in a given directory)  
-Boundary Conditions 0/1 Either periodic(0) boundaries where edges rap around on themselves,or hard(1) boundaries where particles bounce of walls  
-Velocity Distribution 0/1 Either uniform or normal velocity distribution (then value of standard deviation if 1)  

Specific Parameters:  
-Duration of simulation number of "time" iterations  
-Initial number of Particles & Antiparticles not necessarily the same for both  
-Velocity parameter Order of magnitude of velocities,exact meaning depends on the velocity distribution used  
-Lenght of box  


How to Use:  
Save all files except "old" folder to the same directory.  
Install the requirements: pip install -r requirements.txt  
Run Proj_MAIN.py  


Detailed Explanation:  
First you give the necessary parameters to the main function, this initialises all the variables properly and creates a list of particles and antiparticles using the particles class at starting positions determined using the positioon distribution function GEN_X.  
Then for each time you generate a list of the (anti)particle positions before and after a time step using the DO particle class function, which calculates the position of the (anti)particle after a time step dt when moving at speed determined by the speed distribution function GEN_V, away from it's initial position while taking into account the boundaries thanks to the BOUNDS function.  
Then we need to handle the collisions & annihilations:  
We first call the COLLISIONS on mode 0, this deals with the annihilations, since by first getting rid of those as soon as possible speeds up future calculations.  
An interatction is found by calling the INTERCHECK function for each particle/antiparticle pair using the affine coeffecients for each of their trajectories obtained with GetCoefs, as arguments.  
This function finds the closest point between their trajectories using MINIMISE and then returns information on wether an interation is possible, it's type (collision or annihilation),where & when it happended, and information about the (anti)particles involved.  
We then remove the first particle antiparticles involved in the annihilation chronologically.  
The problem is that after removing a annhilation the interactions change unpredictably and so we need to recheck them after each annihilation.   
To do this more efficiently we first collect a history of all the annihilations that could take place, so that when rechecking the events we only check possible annihilations between particles and antiparticles that we know could interact. 
We can now call the COLLISIONS function on mode 1, this deals with the collisions. Again we use the INTERCHECK function but this time  we use it between each particle/particle pair then each antiparticle/antiparticle pair. Instead of removing particles we add each collision's information to a list to use later.  
having finished withe annhihilations/collisions we can know loop through the remaining particle positions and update the TRACKING list, which tracks positions for latter use in plotting.  
We can also update the density of each particle type for latter study.  
Once this is done the most computationaly heavy part of the program is done and the next part depends on the representation type:  
If you choose to only display densities then the relevant information is sent to the Proj_Density script to plot the densities as a function of time and compare them to the theoretical values.  
If you choose to also show the trajectories then using either Proj_DRAW or Proj_DRAW3D will extract the inforamtion needed to follow the evolution of the particles/antiparticles positions and the densities as a function of the time. (For the 3D case an animation is saved unlike the other cases where the trajectories are directly displayed) 
