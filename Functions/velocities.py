import numpy as np
import scipy.stats as scistats
import matplotlib.pyplot as plt
import sys

def calculate_velocities(als,v_peak,v_min,v_max,ntraj):
    '''
    Random truncated gaussian distribution for velocities of NO & Rg 
    About 20% of velocities will fail sob1 > sob2, faster to use excess than loop
    '''
    #Calculate standard deviation for a gaussian
    std = als/np.sqrt(2)
    
    #Create a truncated gaussian with specified minimum and maximum values
    tgvx = scistats.truncnorm(
        (v_min - v_peak) / std,
        (v_max - v_peak) / std,
        loc = v_peak, 
        scale = std
    )
    #Increase the number of trajectories rather than loop to fix failed
    gtraj = int(ntraj)
    
    #Obtain a specified number of trajectories from the truncated gaussian
    v_x_traj = tgvx.rvs(gtraj)

    return v_x_traj