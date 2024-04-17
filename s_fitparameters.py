##############################################################################
# Parameters for s_fit 13                                                    #
#                                                                            #
# Input Parameters that might need to be changed more often are in Section 1 #
#                                                                            #
# General Parameters are in Section 2                                        #
#                                                                            #
# NEED TO LOOK TROUGH ALL NON-COMMENTED VARIABLES TO CHECK THEY ARE ACTUALLY #
# USED                                                                       #
#                                                                            #
# ALSO NEED TO CHECK ABOUT USING V_NO AND V_Rg TO CALCULATE TIMES            #
##############################################################################


import numpy as np

##############################################################################
# SECTION 1                                                                  #
##############################################################################

nproc_main = 7 # Number of Threads
jprime = 8.5 # Value of jprime
Init_state_designation = 1 # 1: f, 1/2, 2: f, 3/2, 3: e 1/2, 4: e 3/2
Final_state_designation = 3 # 1: f, 1/2, 2: f, 3/2, 3: e 1/2, 4: e 3/2
QFlag = True # Generate Q-branch simulations
Pure_Branch = False # this should be "T" for pure branches

ntraj = 8e4 # Number of trial trajectories in one run
tot_ntraj = 1e6 # Total number of trial trajectories run
number_of_runs = int(np.ceil(tot_ntraj/ntraj))
sliceno = 65 #Number of slices
VZtoT = 51/35 # Conversion factor of wNOzprime to time
norderl = 18 # Number of legendre moments
OFlag = 'F' # Are we measuring jprime orientation using cicularly polarised light?
BFlag = True # Are we orientating the bond axis of NO?
Orientation_axis = 'x' # Along which axis are we orientating NO? (x or z)
xdet = 0.0 # r_{laser} x component - See Chris Eyles' Thesis Pg 105
ydet = 0.0 # r_{laser} y component - See Chris Eyles' Thesis Pg 105
dtlaser = 0.0 # delay time of laser
fwhmblur = 6.0 # Controls the power of the blurring function (mentioned on page 108 of Chris Eyles' Thesis)
PIX2V = 5.2 #3 # Proportionality constant from pixel on camera to velocity # was 5.2

Newton_Triangle = False # Should The Program output a file showing the Newton Triangle along with the data?

#Parameters from Ellipse Program
rstretchx = 0.8676 # 1.0
rstretchy = 1.1324
rtheta = -40.72*np.pi/180.0 #-54.21614818016238*np.pi/180.0 #-40.72*np.pi/180.0 #-54.21614818016238*np.pi/180.0

##############################################################################
# SECTION 2                                                                  #
##############################################################################

img_size = 150

Rg_mass = 83.8*1.66053886e-27 #20*1.66053886e-27
NO_mass = 30.0*1.66053886e-27
ROTCONST = 3.359e-23
SpinO = 2.445e-21
V_NO = 617.0 
V_Rg = 531.0 #856
V1_Peak = 617#618.4209560708562
V2_Peak = 531 #531.4783634720409


mu = (Rg_mass*NO_mass)/(Rg_mass + NO_mass)
half_mu = mu*0.5
two_mu = mu*2.0
twoOmu = 2.0/mu
inv_mNOmRg = 1.0/(Rg_mass+NO_mass)
mass_const = Rg_mass/(NO_mass + Rg_mass)


# Collision Parameters Based on Experimental Geometry

dist_NO = 3.0 # distance NO travels to scattering centre
dist_Rg = -0.085 # distance Rg travels to scattering centre

zdet = 0.0 # r_{laser} z component - See Chris Eyles' Thesis Pg 105

if OFlag == 'T':
    oangle = 45.0 # oangle is electric vector of 226 laser - see: M. Brouard, H. Chadwick, C. J. Eyles, B. Hornung, B. Nichols, F. J. Aoiz, P. G. Jambrina, and S. Stolte , "Rotational alignment effects in NO(X) + Ar inelastic collisions: An experimental study", J. Chem. Phys. 138, 104310 (2013)
else:
    oangle = 90.0

# Bond Orientation Parameters

alphaE = 0.64 # e and f state mixing paramters
betaE = 1.26
alphabeta = alphaE*betaE # See Victoria Walpole Thesis chapter 5

#Molecular Beam Timings

t_NO = dist_NO/V_NO # Time takes for average NO molecule to reach scattering centre
t_Rg = -dist_Rg/V_Rg # Time takes for average Rg atom to reach scattering centre
dtNORg = t_NO - t_Rg # Differnce in the above times

# Rare Gas Molecular Beam Profile

fwhm_Rg_V = 26.55 # Rare Gas Velocity Distribution fwhm
fwhm_Rg_t = 0.000145 # Rare Gas Time Distribution at start of trajectory (ie distribution of 'birth' of molecule) FWHM
fwhm_Rg_w = 0.004 # Rare Gas Density Distribution in terms of distance from line of centre of molecular beam at 'birth' FWHM
fwhm_Rg_ang = 4.0 # Rare gas Density distribution in terms of polar angle of a Rg atom from line of centre of molecular beam FWHM

ALS2 = fwhm_Rg_V/(2.0*np.sqrt(np.log(2))) # sqrt(2)*(Standard Deviation) of velocity distribution
V2_min = V_Rg - 2.0*ALS2
V2_max = V_Rg + 2.55*ALS2
s_Rg_t = 4.0 * np.log(2)/(fwhm_Rg_t**2) # an alpha factor - See equation 3.10 of Chris Eyles Thesis. Equivalent to 1/2*STDDEV^2 (see wikipedia page on FWHM)

invALS2 = 1.0/ALS2 # 1/sqrt(2)*STDDEV

# NO Molecular Beam Profile

fwhm_NO_V = 49.36 # NO Velocity Distribution fwhm
fwhm_NO_t = 0.0001 # NO Time Distribution at start of trajectory (ie distribution of 'birth' of molecule) FWHM
fwhm_NO_w = 0.002 # NO Density Distribution in terms of distance from line of centre of molecular beam at 'birth' FWHM
fwhm_NO_ang = 0.0 # NO Density distribution in terms of polar angle of a NO molecule from line of centre of molecular beam FWHM

ALS1 = fwhm_NO_V/(2.0*np.sqrt(np.log(2))) # sqrt(2)*(Standard Deviation) of velocity distribution
V1_min = V_NO - 2.0*ALS1
V1_max = V_NO + 2.55*ALS1

invALS1 = 1.0/ALS1 # 1/sqrt(2)*STDDEV

gas_width = fwhm_NO_w*fwhm_Rg_w

# Detection Volume Parameters

fwhm_226 = 0.003 # Intenisty Distriburion of 226nm laser fwhm # was 0.0015
fwhm_266 = 0.003 # Intenisty Distriburion of 266nm laser fwhm

s_266 = 4.0*np.log(2.0)/(fwhm_266**2.0) # an alpha factor - See equation 3.10 of Chris Eyles Thesis. Equivalent to 1/2*STDDEV^2 (see wikipedia page on FWHM)
s_226 = 4.0*np.log(2.0)/(fwhm_226**2.0) # an alpha factor - See equation 3.10 of Chris Eyles Thesis. Equivalent to 1/2*STDDEV^2 (see wikipedia page on FWHM)

# Monte Carlo Parameters

tfwhmNOangpidiv360 = fwhm_NO_ang*2.0*np.pi/360 # fwhm_NO_ang in radians
tfwhmRgangpidiv360 = fwhm_Rg_ang*2.0*np.pi/360 # fwhm_Rg_ang in radians
invssqrt2als1 = ALS1*(1.0/np.sqrt(2.0)) #STDDEV of NO velocity distribution
invssqrt2als2 = ALS2*(1.0/np.sqrt(2.0)) #STDDEV of NO velocity distribution

# Blurring Function Quantities

Tolc_Blur = 0.05

#Precalculate variables to increase speed
PI = np.pi
TPI = 2 * PI
HPI = 0.5 * PI
NPSQ2 = np.sqrt(2)
RPI = PI / 180.0
PIR = 180.0 / PI
ntraj = int(ntraj)
