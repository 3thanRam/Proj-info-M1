# Proj-info-M1

Computer Sciences Project M1 : Brownian and ballistic annihilation processes

Project Description:

Simulates the trajectories and densities of particles and antiparticles when trapped in a box.

Necessary Parameters:  
-Input setting "r"/"c" Meaning Recommended values known to behave correctly or Custom lets you choose all the Specific parameters  
-Representation Type 0/1 Plot only densities(0) or also plot trajectories (1)  
-Dimensions Choose (1,2,3) Number of spatial dimensions (if 3D then an animation is saved in a given directory)  
-Boundary Conditions 0/1 Either periodic(0) boundaries where edges rap around on themselves,or hard(1) boundaries where particles bounce of walls  

Specific Parameters:  
-Duration of simulation number of "time" iterations  
-Initial number of Particles & Antiparticles not necessarily the same for both  
-Velocity parameter Order of magnitude of velocities,exact meaning depends on the velocity distribution used  
-Lenght of box  


How to Use:  
Save all files to the same directory.  
Install the requirements: pip install -r requirements.txt  
Run Proj_MAIN.py  
