# Program Name : sfit (Version 14.0)
#
# Authors: Max McCrea 2021
#
# Based on fortran code by: Sean Gordon
#                           Balasz Hornung
#                           Bethan Nichols
#                           Helen Chadwick
#                           Chris Eyles
#                           Victoria Walpole
#                           Cornelia Heid
#                           Razvan Gheorghe
#
# This program simulates trajectories in order to take into account the
# instrument fucnction, so that basis sets can be created. Therefore, this
# program takes into account the differences in detection probability caused
# by: the flux-density transformation, collision inducced alignment, the
# choice of laser polarisation used, and some other effects.
import os
import sys

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

from time import time

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter as gf

from functions import *
from s_fitparameters import *
from Functions.sfit_class import array_functions
from Functions.velocities import calculate_velocities

directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(directory)
st = time()
np.random.seed()
plt.figure(figsize = (12, 9))
ias = 2*img_size+1

polimh1_B, polimh2_B = np.zeros((99,ias,ias)), np.zeros((99,ias,ias))
polimv1_B, polimv2_B= np.zeros((99,ias,ias)), np.zeros((99,ias,ias))
polimh1, polimh2 = np.zeros((99,ias,ias)), np.zeros((99,ias,ias))
polimv1, polimv2= np.zeros((99,ias,ias)), np.zeros((99,ias,ias))
polimh, polimv = np.zeros((99, ias,ias)), np.zeros((99, ias,ias))
polimh_B, polimv_B = np.zeros((99, ias,ias)), np.zeros((99, ias,ias))
Instr_Func = np.zeros((99,ias,ias))

Instr_Func_B = np.zeros((99,ias,ias))

# Determining Rotational Energies
EROTI = HUND_ENERGY(SpinO, ROTCONST, 0.5, Init_state_designation)
EROTF = HUND_ENERGY(SpinO, ROTCONST, jprime, Final_state_designation)
EROTd = EROTF - EROTI # NO rotational energy difference

if Pure_Branch:
    branch_val = 1.0
else:
    branch_val = 0.0

hcoe1 = hfact(1, jprime, branch_val)
hcoe2 = hfact(2, jprime, branch_val)

if QFlag:
    hcoe_sim1 = 1.0 + hfact(1, jprime, 1.0)
    hcoe_sim2 = 1.0 + hfact(2, jprime, 1.0)
else:
    hcoe_sim1 = hcoe1
    hcoe_sim2 = hcoe2

Dep_fact1 = Depolarisation_coeff(jprime, 1.0, 1)
Dep_fact2 = Depolarisation_coeff(jprime, 1.0, 2)

oangle = (oangle*np.pi)/180.0 # Convert Electic Vector Angle of Probe Laser to Radians


rPDDCS = np.loadtxt(f'rpddcs_jp{int(jprime)}.dat')
rPDDCS[:, 1] = np.loadtxt('BackScattR00.dat').T[1]
PDDCS = np.loadtxt(f'ApseModelPDDCS{int(jprime)}.dat')
iso = np.loadtxt(f'iso_jp{int(jprime)}.dat')

# rlaser defines the vector r_{laser} - See Chris Eyles' Thesis Page 104
rlaser = np.array([xdet, ydet, 0.0])

# Define Propagation Direction of 226 laser
# For Linearly Polarised Light
Vector_226_Prop = np.array([-1/NPSQ2, 1/NPSQ2, 0.0]) # NOT SURE ABOUT INCLUSION OF RLASER

# Now propagation of 266 Laser
Vector_266_Prop = np.array([1/NPSQ2, 1/NPSQ2, 0.0]) # NOT SURE ABOUT INCLUSION OF RLASER

# Define Unit Vectors in the x, y and z directions
Vec_X = np.array([1.0, 0.0, 0.0])
Vec_Y = np.array([0.0, 1.0, 0.0])
Vec_Z = np.array([0.0, 0.0, 1.0])

# With linearly polarised light, a rotation matrix is defined to allow transformation from the scattering frame to the detector frame.
# Well technically it rotates the vectors that make up the scattering frame from their positions in the LAB frame, into the detector frame.
Scat_To_Det_Frame = np.zeros((3,3)) #Initialise a 3x3 rotation matrix
Scat_To_Det_Frame[:, 0] = np.array([1/np.sqrt(2), 0.0, -1/np.sqrt(2)])
Scat_To_Det_Frame[:, 1] = np.array([1/np.sqrt(2), 0.0, 1/np.sqrt(2)])
Scat_To_Det_Frame[:, 2] = np.cross(Scat_To_Det_Frame[:, 0], Scat_To_Det_Frame[:, 1])

# Gives Newton Triangle
if Newton_Triangle:
    NT = np.zeros((5,2))
    NT[0] = np.array([0.0, 0.0])
    NT[1] = np.array([-(V_NO*NO_mass)*inv_mNOmRg/PIX2V, (V_Rg*Rg_mass)*inv_mNOmRg/PIX2V])
    NT[2] = np.array([-(V_NO*NO_mass)*inv_mNOmRg/PIX2V, ((V_Rg*Rg_mass)*inv_mNOmRg-V_Rg)/PIX2V])
    NT[3] = np.array([-((V_NO*NO_mass*inv_mNOmRg)-V_NO)/PIX2V, ((V_Rg*Rg_mass)*inv_mNOmRg)/PIX2V])
    NT[4] = np.array([-(V_NO*NO_mass)*inv_mNOmRg/PIX2V, (V_Rg*Rg_mass)*inv_mNOmRg/PIX2V])
    plt.plot(NT[:, 0], NT[:, 1])
    plt.savefig('NewtonTriangleOutput.png')
    plt.clf()

def run_traj(q_in,q_out):
    '''
    Function thread that calculates trajectories

    q_in contains the number of loops to go

    q_out contains the polimh/polimv data
    '''
    while True:
        #get the current id from the in direction queue (q_sub_in)
        cid = q_in.get()
        #When no ids remain None has been added to queue, kill thread
        if cid is None:
            break
        maxPIMMSslices = 0 #Is overwritten, no need to change
        ### Calculate initial velocities for ntraj trajectories
        sfit = array_functions() #Spawn the object that holds all relevant arrays
        V_NO_traj = calculate_velocities(ALS1,V1_Peak,V1_min,V1_max,ntraj)
        V_Rg_traj = calculate_velocities(ALS2,V2_Peak,V2_min,V2_max,ntraj)
        V_Rg_traj *= -1.0 # Rg travels along -y direction, so is made negative

        ### Calculate X,Y,Z NO velocities
        #I FEEL THIS SHOULD BE STDDEV - polar angle from line of centre of mol beam
        NO_ang_t = np.random.randn(ntraj) * tfwhmNOangpidiv360
        # Azimuthal angle of NO trajectory. 0 is coincident with the TOF axis (z axis)
        NO_ang_p = np.random.rand(ntraj) * TPI
        # Velocity of NO beam along the propagation axis of NO (x axis)
        sfit.a['V_NO_trajx'] = V_NO_traj * np.cos(NO_ang_t)
        sfit.a['V_NO_trajy'] = V_NO_traj * np.sin(NO_ang_t) * np.sin(NO_ang_p)
        sfit.a['V_NO_trajz'] = V_NO_traj * np.sin(NO_ang_t) * np.cos(NO_ang_p)

        ### Calculate X,Y,Z Rg velocities
        # Polar Angle of Trajectory for Rg
        Rg_ang_t = np.random.randn(ntraj) * fwhm_Rg_ang * RPI

        # Azimuthal angle of Rg trajectory. 0 is coincident with the TOF axis (z axis)
        Rg_ang_p = np.random.rand(ntraj) * TPI
        sfit.a['V_Rg_trajx'] = V_Rg_traj * np.sin(Rg_ang_t) * np.sin(Rg_ang_p)
        sfit.a['V_Rg_trajy'] = V_Rg_traj * np.cos(Rg_ang_t)
        sfit.a['V_Rg_trajz'] = V_Rg_traj * np.sin(Rg_ang_t) * np.cos(Rg_ang_p)

        ### Initial Realtive Velocity (k) and Collision Energy
        sfit.a['vrelx'] = sfit.a['V_NO_trajx'] - sfit.a['V_Rg_trajx']
        sfit.a['vrely'] = sfit.a['V_NO_trajy'] - sfit.a['V_Rg_trajy']
        sfit.a['vrelz'] = sfit.a['V_NO_trajz'] - sfit.a['V_Rg_trajz']
        sfit.reshape_tall()
        sfit.a['vrel'] = np.concatenate(
            [sfit.a['vrelx'], sfit.a['vrely'], sfit.a['vrelz']],
            axis=1)
        #|k| = math.sqrt(sum(i**2 for i in x/y/z)) = linalg norm
        sfit.a['modvrel'] = np.linalg.norm(sfit.a['vrel'],axis=1)
        ecoll = half_mu * (sfit.a['modvrel'] * sfit.a['modvrel']) # Collision Energy
        ecoll = tranpose_1d(ecoll)
        sfit.a['ecollprime'] = ((ecoll - EROTd)*np.random.rand(ntraj, 1)).reshape(ntraj)
        passed_ecoll = np.greater(sfit.a['ecollprime'],0.0) #Get indexes that are > 0
        #Remove failed trajectories by taking true indexes from check
        sfit.shorten_arrays(passed_ecoll,ntraj)
        sp1 = sfit.percent

        ### Calculate collision arrival time for NO
        # Time from centre of NO beam
        dtno = np.random.randn(sfit.gtraj,1) * fwhm_NO_t
        # Distance from line of centre of secondary beam pulse in x direction
        sfit.a['dwRgx'] = np.random.randn(sfit.gtraj,1) * fwhm_Rg_w

        # Time taken for NO to travel from 'birth' location to the point at
        # which it meets a Rg atom
        sfit.a['tNOcoll'] = (dist_NO + sfit.a['dwRgx']) / sfit.a['V_NO_trajx'] + dtno
        #print(sfit.a['tNOcoll'][np.argmin(sfit.a['ecollprime'])], np.min(sfit.a['tNOcoll']), np.mean(sfit.a['tNOcoll']))
        # If the laser fires before collision NO will not be state selected
        # Therefore the collision time must be greater than the NO and laser timing

        successful_interaction = np.greater(t_NO + dtlaser,sfit.a['tNOcoll'])
        
        sfit.shorten_arrays(successful_interaction,ntraj)
        sp2 = sfit.percent
        
        ### LAB to COM Frame transformations (COM frame velocities given by letter w)

        vcomx = (NO_mass * sfit.a['V_NO_trajx'] + Rg_mass * sfit.a['V_Rg_trajx']) * inv_mNOmRg
        vcomy = (NO_mass * sfit.a['V_NO_trajy'] + Rg_mass * sfit.a['V_Rg_trajy']) * inv_mNOmRg
        vcomz = (NO_mass * sfit.a['V_NO_trajz'] + Rg_mass * sfit.a['V_Rg_trajz']) * inv_mNOmRg

        ### Initial COM Frame NO velocity
        sfit.a['wNOx'] = sfit.a['V_NO_trajx'] - vcomx
        sfit.a['wNOy'] = sfit.a['V_NO_trajy'] - vcomy
        sfit.a['wNOz'] = sfit.a['V_NO_trajz'] - vcomz
        sfit.a['wNO'] = np.concatenate(
            [sfit.a['wNOx'], sfit.a['wNOy'], sfit.a['wNOz']],
            axis=1)
        mag_wNO = np.linalg.norm(sfit.a['wNO'],axis=1) # Magnitude of wNO

        # Magnitude of Final COM frame velocity of NO - In this code is used interchangeably with kprime.
        # They are colinear, and some scale factor cancels out the difference.

        mag_wNOprime = np.sqrt(sfit.a['ecollprime'] * twoOmu) * mass_const

        # Angle between the x axis (NO propagation vector) and k
        theta0 = np.arctan(sfit.a['wNO'][::,[0]] / (-sfit.a['wNO'][::,[1]])) + HPI

        sfit.a['phicom'] = np.random.uniform(0.0, TPI, size=(sfit.gtraj, 1)) # Azimuthal Scattering Angle PhiT
        sfit.a['thetacom'] = np.random.uniform(0.0, 1.0, size=(sfit.gtraj, 1)) # Polar Scattering Angle
        sfit.a['thetacom']*=2.0
        sfit.a['thetacom']-=1.0
        sfit.a['thetacom'] = np.arccos(sfit.a['thetacom'])

        wNOprimexTOF = mag_wNOprime * np.cos(sfit.a['thetacom'])
        wNOprimeyTOF = mag_wNOprime * np.sin(sfit.a['thetacom']) * np.sin(sfit.a['phicom'])

        # Rotate this to be in the COM frame
        # x parallel to NO propagation, -y parallel to Rg propagation, z parallel to TOF axis
        sfit.a['wNOpx'] = np.cos(theta0) * wNOprimexTOF - np.sin(theta0) * wNOprimeyTOF
        sfit.a['wNOpy'] = np.cos(theta0) * wNOprimeyTOF + np.sin(theta0) * wNOprimexTOF
        sfit.a['wNOpz'] = mag_wNOprime * np.sin(sfit.a['thetacom']) * np.cos(sfit.a['phicom'])
        sfit.a['wNOprime'] = np.concatenate(
            [sfit.a['wNOpx'], sfit.a['wNOpy'], sfit.a['wNOpz']],
            axis=1)

        # Convert this to LAB frame (vlab)
        # x colinear with NO propagation, -y colinear with Rg propagation, z colinear with TOF axis - NB.
        # Note origin of COM frame shifts, hence the differnce (colinear vs parralel)
        sfit.a['vNOprime'] = np.concatenate(
            [sfit.a['wNOpx'] + vcomx, sfit.a['wNOpy'] + vcomy, sfit.a['wNOpz'] + vcomz],
            axis=1)
        ### Now we calculate the time that the Rg must originate from
        # relative to mean time of birth of Rg atoms, to be able to collide
        # with the NO molecule.
        # distance from line of centre of primary beam pulse in y direction
        dwNOy = np.random.randn(sfit.gtraj,1) * fwhm_NO_w
        # Return 0 instead of inf when dividing by zero
        nonzero = np.divide(
            (dist_Rg + dwNOy), #Numerator
            sfit.a['V_Rg_trajy'], #Denominator
            out = np.zeros_like((dist_Rg + dwNOy)), #Return 0 if cant divide
            where = sfit.a['V_Rg_trajy'] != 0 #Divide when array != 0
            )
        dtRg = sfit.a['tNOcoll'] - nonzero - dtNORg
        # alpha factor - used to calculate likelihood of this time
        part = np.exp(-s_Rg_t * (dtRg * dtRg))

        ### Laser fires at t_NO + dtlaser. Where is NO at this time?
        dz = np.sqrt(abs(np.random.randn(sfit.gtraj) * np.random.randn(sfit.gtraj) * gas_width))
        #Apply a random sign to dz
        dzs = np.random.choice([-1,1],sfit.gtraj)
        dz *= dzs
        dz = np.reshape(dz,(len(dz),1))

        # r_{det} See Chris Eyles' Thesis pages 91 and 104-106
        rdetx = (t_NO + dtlaser - sfit.a['tNOcoll']) * sfit.a['vNOprime'][:,[0]] + sfit.a['dwRgx']
        rdety = (t_NO + dtlaser - sfit.a['tNOcoll']) * sfit.a['vNOprime'][:,[1]] + dwNOy
        rdetz = (t_NO + dtlaser - sfit.a['tNOcoll']) * sfit.a['vNOprime'][:,[2]] + dz
        rdet = np.concatenate(
            [rdetx, rdety, rdetz],
            axis=1)

        def perp_dist_from_laser(rDET, Laser_Prop):
            aux1 = np.linalg.norm(np.cross(rDET, Laser_Prop), axis = 1)
            aux2 = np.linalg.norm(Laser_Prop)
            return aux1/aux2
            
        # We then calculate r226 and r266 and their respective gaussian weights
        rDET = rdet - rlaser
        r226 = perp_dist_from_laser(rDET, Vector_226_Prop)
        r266 = perp_dist_from_laser(rDET, Vector_266_Prop)

        # Calculate Probability of Detection at this location including
        # liklihood that Rg will collide. modvrel included to make up rate constant?
        sfit.a['det226'] = np.exp(-s_226 * (r226 * r226))
        sfit.a['det266'] = np.exp(-s_266 * (r266 * r266))
        sfit.reshape_tall()
        sfit.a['probdet'] = part * sfit.a['det226'] * sfit.a['det266'] * sfit.a['modvrel']

        passed_prob = np.greater(sfit.a['probdet'],1e-90)
        sfit.shorten_arrays(passed_prob,ntraj)
        sp3 = sfit.percent

        # Now we look to apply the ellipse stretching and squeezing caused
        # by the presence of the orientation rods. First we convert the final
        # velocity of NO into pixels (in float form for now)
        inv_pixtov = 1/PIX2V
        rx = sfit.a['wNOpx'] * inv_pixtov
        ry = sfit.a['wNOpy'] * inv_pixtov
        rplane = np.sqrt(rx**2 + ry**2)
        sfit.a['t'] = sfit.a['wNOpz'] * VZtoT
        sfit.a['PIMMSslice'] = np.around(sfit.a['t']/25).astype(int)
        if np.max(np.abs(sfit.a['PIMMSslice'])) > maxPIMMSslices:
            maxPIMMSslices = np.max(np.abs(sfit.a['PIMMSslice']))

        # Now we rotate the axes so that they align with the long and
        # short axes of the ellipse
        rxtemp = rx * np.cos(rtheta) - ry * np.sin(rtheta)
        rytemp = rx * np.sin(rtheta) + ry * np.cos(rtheta)

        # Now we stretch or squash along those axes
        rxtemp = rxtemp * rstretchx
        rytemp = rytemp * rstretchy

        # Now we rotate back
        rx = rxtemp * np.cos(-rtheta) - rytemp * np.sin(-rtheta)
        ry = rxtemp * np.sin(-rtheta) + rytemp * np.cos(-rtheta)

        # Now we make this into integer pixel numbers
        sfit.a['nx'] = np.round(rx[:, 0], 0).astype(int)
        sfit.a['ny'] = np.round(ry[:, 0], 0).astype(int)

        # Check that pixels are within bounds of image
        nxl = np.less_equal(sfit.a['nx'],img_size)
        sfit.shorten_arrays(nxl,ntraj)
        nxg = np.greater_equal(sfit.a['nx'],-img_size)
        sfit.shorten_arrays(nxg,ntraj)
        nyl = np.less_equal(sfit.a['ny'],img_size)
        sfit.shorten_arrays(nyl,ntraj)
        nyg = np.greater_equal(sfit.a['ny'],-img_size)
        sfit.shorten_arrays(nyg,ntraj)

        # For seeminly no reason wNOprime is multiplied by this factor, It's odd.
        sfit.a['wNOpRgn'] = Normalise(sfit.a['wNOprime'] * Rg_mass * inv_mNOmRg)
        sfit.a['vreln'] = Normalise(sfit.a['vrel'])
        oangle = 0
        # We now construct the scattering frame (See almost any thesis or paper)
        Y_Scatt = Normalise(np.cross(sfit.a['vreln'], sfit.a['wNOpRgn']))

        # We now want to rotate into the detector frame
        # This coordinates are the y & z axes of the scattering frame in the detector frame
        # (See J. Chem. Phys. 138, 104310 (2013))
        Y_Det = Normalise(np.array([np.matmul(Scat_To_Det_Frame, x) for x in Y_Scatt]))
        Z_Det = Normalise(np.array([np.matmul(Scat_To_Det_Frame, x) for x in sfit.a['vreln']]))

        # Now we calculate the angles - Again See J. Chem. Phys. 138, 104310 (2013)
        # First we calculate the normal Z zeta surface
        # the surface defined by the two z axis (detector and scattering frame)
        Z_zeta_norm = Normalise(np.cross(Z_Det, Vec_Z))

        ### Calculate various angles
        # Now laser propagation axis is aligned with z axis
        # thus angle between Z_Det (k in detector frame) and z axis is theta
        # (See Angular Momentum by Zare, page 78 for diagram explaining the three angles)
        theta_rot = np.arccos(np.dot(Z_Det, Vec_Z)) # Polar Angle


        theta_rot = tranpose_1d(theta_rot)
        sin_theta = np.sin(theta_rot)
        cos_theta = np.cos(theta_rot)
        sin_2theta = 2.0 * sin_theta * cos_theta

        ## Frobenius normalization of vectors
        nvec_y = np.linalg.norm(Vec_Y)
        # If theta != 0 or pi
        nZ_zeta_norm = np.linalg.norm(Z_zeta_norm,axis=1)
        nzeta_y = nZ_zeta_norm * nvec_y
        # If theta == 0 or pi
        nY_Det = np.linalg.norm(Y_Det,axis=1)
        nyd_y = nY_Det * nvec_y
        # if phi != 0
        nyd_zeta = nY_Det * nZ_zeta_norm

        phi_rot_arg = np.sum(Y_Det * Z_zeta_norm, axis=1)/nyd_zeta
        grt1 = np.greater(phi_rot_arg, 1)
        phi_rot_arg[grt1] = 1.0

        lt1 = np.less(phi_rot_arg, -1)
        phi_rot_arg[lt1] = -1.0

        phi_rot = np.arccos(phi_rot_arg) # Azimuthal Angle
        phi_rot = tranpose_1d(phi_rot)
        chi_rot = np.arccos(np.dot(Z_zeta_norm,Vec_Y)/nzeta_y) # Orientation Angle
        chi_rot = tranpose_1d(chi_rot)

        # When phi = 0.0, which it must be when theta = 0 or pi, then chi can equally
        # be measured from Z_zeta_Normal, or from Y, as they are coincident
        indices = np.argwhere((theta_rot == 0.0) | (theta_rot == np.pi))[:,0]
        if len(indices) != 0:
            t = np.arccos(np.dot(Y_Det[indices],Vec_Y)/nyd_y[indices])
            chi_rot[indices] = tranpose_1d(t)
            phi_rot[indices] = 0.0



        ## Check if angles should be +ve or -ve
        def check_sign(vec1,vec2,vec3):
            _cross = np.cross(vec1,vec2)
            _mul = _cross * vec3
            _sum = np.sum(_mul,axis=1)
            _norm = np.linalg.norm(_mul,axis=1)
            _prod = np.ones((len(_norm),1))
            indices = np.argwhere(_norm > 1e-8)
            if len(indices) != 0:
                _prod[indices, 0] = np.sign(_sum[indices]/_norm[indices])
            return _prod

        #theta
        sign = check_sign(Z_Det, Vec_Z, Z_zeta_norm)
        theta_rot *= sign

        #phi
        sign = check_sign(Y_Det, Z_zeta_norm, Z_Det)
        phi_rot *= sign

        #chi
        sign = check_sign(Vec_Y, Vec_Z, Z_zeta_norm)
        chi_rot *= sign

        # Orientation Angle Changes for Vertical Polarised Light Compared with above
        chi_ver = chi_rot - HPI
        indices = np.argwhere(chi_ver > TPI)
        if len(indices) != 0:
            t = chi_ver[indices] - TPI
            chi_ver[indices] = tranpose_1d(t)
        # Now we calculate the trigonometric functions once and then save them,
        # rather than getting the code to calculate trig over and over
        sin_phi = np.sin(phi_rot)
        cos_phi = np.cos(phi_rot)
        sin_2phi = 2.0 * cos_phi * sin_phi
        cos_2phi = 2.0 * (cos_phi * cos_phi) - 1.0

        # Convert Scattering angle to degrees
        thetacom_deg = sfit.a['thetacom']/ RPI

        # Calculation of c1 and c2 coefficient - See Balasz Hornung's Thesis section 7.3.1
        c1 = 1.0/np.sqrt(jprime*(jprime + 1.0))
        c2 = np.sqrt((2.0*jprime+3.0)*(2.0*jprime-1.0)/(jprime*(jprime+1.0)))

        invsqrt3c2 = (1.0/np.sqrt(3))*c2

        # Create interpolation object of isotopc DCS     
        if Final_state_designation == 1:
            sigma_iso = interp1d(PDDCS[:, 0], iso[:, 1])
        elif Final_state_designation == 3:
            sigma_iso = interp1d(PDDCS[:, 0], iso[:, 2])

        # Interpolate isotropic DCS at angles the trajectories are scattered into
        sigma_iso = sigma_iso(thetacom_deg)

        # Orientation and Alignment Parameters Dictionary
        rho_kq = {}

        # Same process as before
        rho_kq['rho_11'] = interp1d(PDDCS[:, 0], PDDCS[:, 4])
        rho_kq['rho_20'] = interp1d(PDDCS[:, 0], PDDCS[:, 5])
        rho_kq['rho_21'] = interp1d(PDDCS[:, 0], PDDCS[:, 6])
        rho_kq['rho_22'] = interp1d(PDDCS[:, 0], PDDCS[:, 7])

        # J. Chem. Phys. 138, 104310 (2013)
        # The moments loaded in from file are de Miranda real from table (doesn't include a11- for no reason)
        # Here they are converted into the Macek and Fano Real moments
        rho_kq['O1-'] = rho_kq['rho_11'](thetacom_deg) * c1
        rho_kq['A0'] = rho_kq['rho_20'](thetacom_deg) * c2
        rho_kq['A1+'] = rho_kq['rho_21'](thetacom_deg) * invsqrt3c2
        rho_kq['A2+'] = rho_kq['rho_22'](thetacom_deg) * invsqrt3c2

        # Dictionary containing r-PDDCSs
        R_kq = {}

        # Same as before
        R_kq['R00'] = interp1d(rPDDCS[:, 0], rPDDCS[:, 1])
        R_kq['R10'] = interp1d(rPDDCS[:, 0], rPDDCS[:, 2])
        R_kq['R11'] = interp1d(rPDDCS[:, 0], rPDDCS[:, 3])

        R_kq['R00'] = R_kq['R00'](thetacom_deg)
        R_kq['R10'] = R_kq['R10'](thetacom_deg)
        R_kq['R11'] = R_kq['R11'](thetacom_deg)

        def detector_frame(cos_2chi, sin_2chi, O1, A0, A1, A2):
            # dmomcal calculates Eq(18) in Rev. Mod. Phys. 45, 553
            # Converts the moments into the detector frame
            O0det = O1 * sin_theta * sin_phi
            A0det = 0.5*A0*(3*(cos_theta**2) - 1.0) + 1.5*A1*sin_2theta*cos_phi + 1.5*A2*(sin_theta**2)*cos_2phi
            A2det = 0.5*A0*(sin_theta**2)*cos_2chi + A1*(sin_theta*cos_phi*sin_2chi + sin_theta*cos_theta*sin_phi*cos_2chi) + A2*(0.5*((1.0+cos_theta**2)*cos_2phi*cos_2chi) - cos_theta*sin_2phi*sin_2chi)
            return O0det, A0det, A2det

        # See dmomcal
        cos_2chi = np.cos(2.0 * chi_ver)
        sin_2chi = np.sin(2.0 * chi_ver)

        O0det_V, A0det_V, A2det_V = detector_frame(cos_2chi, sin_2chi, rho_kq['O1-'], rho_kq['A0'], rho_kq['A1+'], rho_kq['A2+'])

        cos_2chi = np.cos(2.0*chi_rot)
        sin_2chi = np.sin(2.0*chi_rot)

        O0det_H, A0det_H, A2det_H = detector_frame(cos_2chi, sin_2chi, rho_kq['O1-'], rho_kq['A0'], rho_kq['A1+'], rho_kq['A2+'])

        # Fano Macek Intensity Formula for Intensity_H and Intensity_V - Eq(1) in J. Chem. Phys. 138, 104310 (2013)
        cos_2oangle = np.cos(2*oangle)
        sin_2oangle = np.sin(2*oangle)

        # Equation 14 Rev. Mod. Phys. 45, 553
        # The second term becomes negative for V for no reason. 
        # MESSAGE FOR FUTURE USERS
        #      IF USING CIRCULARLY POLARISED LIGHT CHECK THE ABOVE!!!!!
        # See Balasz Hornung's Thesis page 159 to see that depolarisation coefficients can be a part of h factors, thus can just be multiplied in here
        Intensity_H = 1.0 + 1.5*hcoe_sim1*O0det_H*Dep_fact1*sin_2oangle - 0.5*hcoe_sim2*A0det_V*Dep_fact2 + 1.5*A2det_H*hcoe_sim2*Dep_fact2*cos_2oangle
        Intensity_V = 1.0 - 1.5*hcoe_sim1*O0det_V*Dep_fact1*sin_2oangle - 0.5*hcoe_sim2*A0det_H*Dep_fact2 + 1.5*A2det_V*hcoe_sim2*Dep_fact2*cos_2oangle

        ### This section creates the instrument function
        # Converts nx and ny indices to positive or 0 integers so that it works with numpy arrays
        nx, ny = sfit.a['nx']+img_size, sfit.a['ny']+img_size
        PIMMSslice = sfit.a['PIMMSslice'] + 49
        # Instrument Function Only
        #Instr_Func = np.zeros((ias,ias))

        # tHIS SECTION CREATES SIMULATED IMAGES
        Intensity_1 = 1.0
        Intensity_2 = 1.0
        if BFlag:
            # LAB frame static electric field angles (see pages 139 and 142 of Victoria Walpole's Thesis)
            thetaE = HPI
            if Orientation_axis in ['z', 'Z']:
                phiE = 0.0
            elif Orientation_axis in ['x', 'X']:
                phiE = HPI
            else:
                print('Orientation Axis Not Recognised')
                print('Aborting')
                sys.exit()

            # Get renormalised rPDDCS values for each trajectory
            # Done like this to prevent division by 0
            R_kq['R10'] = np.divide(R_kq['R10'],\
                R_kq['R00'],\
                out = np.zeros_like(R_kq['R10']),\
                where = R_kq['R00'] != 0)
            R_kq['R11'] = np.divide(R_kq['R11'],\
                R_kq['R00'],\
                out = np.zeros_like(R_kq['R11']),\
                where = R_kq['R00'] != 0)
            
            # Equation 5.46 in Victoria Walpole's Thesis. Remember that we've already divided
            # through by R00 and A10 = -1/3|alphabeta| (Equation 5.16 in Victoria Walpole's Thesis)
            # 1 and 2 represent + or - x/z
            stheta, sphicom  = np.sin(thetaE), np.sin(sfit.a['phicom'])
            sphiE, sphiEpi = np.sin(phiE), np.sin(phiE + PI)
            ctheta, cphicom = np.cos(thetaE), np.cos(sfit.a['phicom'])
            cphiE, cphiEpi = np.cos(phiE), np.cos(phiE + PI)
            Intensity_1 = 1.0 - ((alphabeta * R_kq['R10'] * stheta * cphiE) \
                + (alphabeta * NPSQ2 * R_kq['R11'] * (sphicom * stheta * sphiE + cphicom * ctheta)))
            Intensity_2 = 1.0 - ((alphabeta * R_kq['R10'] * stheta * cphiEpi) \
                + (alphabeta * NPSQ2 * R_kq['R11'] * (sphicom * stheta * sphiEpi + cphicom * ctheta)))
        
        q_out.put([
            maxPIMMSslices,
            [(PIMMSslice,nx,ny),sfit.a['probdet']], #Instr_Func
            [(PIMMSslice,nx,ny),sfit.a['probdet'] * R_kq['R00'] * Intensity_H],
            [(PIMMSslice,nx,ny),sfit.a['probdet'] * R_kq['R00'] * Intensity_V],
            [(PIMMSslice,nx,ny),sfit.a['probdet'] * R_kq['R00'] * Intensity_H * Intensity_1],
            [(PIMMSslice,nx,ny),sfit.a['probdet'] * R_kq['R00'] * Intensity_H * Intensity_2],
            [(PIMMSslice,nx,ny),sfit.a['probdet'] * R_kq['R00'] * Intensity_V * Intensity_1],
            [(PIMMSslice,nx,ny),sfit.a['probdet'] * R_kq['R00'] * Intensity_V * Intensity_2],
            [sp1,sp2,sp3]
        ])

def setup_threads(Instr_Func,polimh,polimv,polimh1,polimh2,polimv1,polimv2):
    # Counter for Completion Output
    trajectories_run = 0
    maxPIMMSslices = 0
    percent_done = []
    show_traj_percent = True

    #Make queues to hold input and output data for threads
    q_in = mp.Queue()
    q_out = mp.Queue()

    #Initiate some threads and pass them the partial wave function and the queues
    main_threads = [mp.Process(target=run_traj, args=(q_in,q_out)) for i in range(nproc_main)]

    #Send each partial wave into the main queue
    send_to_main_queue = [q_in.put(i) for i in range(number_of_runs)]

    #Put None to end thread after all partial waves have finished
    [q_in.put(None) for i in range(nproc_main)]

    #Start main threads
    for t in main_threads:
        t.start()

    #Wait for data to be sent to the output queue
    for _ in send_to_main_queue:
        output = q_out.get()
        if show_traj_percent:
            print(output[8])
            print()
            show_traj_percent = False
        if output[0] > maxPIMMSslices: maxPIMMSslices = output[0]
        np.add.at(Instr_Func,output[1][0],output[1][1])
        np.add.at(polimh,output[2][0],output[2][1])
        np.add.at(polimv,output[3][0],output[3][1])
        np.add.at(polimh1,output[4][0],output[4][1])
        np.add.at(polimh2,output[5][0],output[5][1])
        np.add.at(polimv1,output[6][0],output[6][1])
        np.add.at(polimv2,output[7][0],output[7][1])
        
    
        
        trajectories_run+=ntraj
        percent = int(round(100*trajectories_run/tot_ntraj, 0))
        if percent not in percent_done:
            print ("\033[A                             \033[A")
            print(f'{percent}% of Total Trajectories Run')
            percent_done.append(percent)

    #Close threads
    [t.join() for t in main_threads]
    return maxPIMMSslices,Instr_Func,polimh,polimv,polimh1,polimh2,polimv1,polimv2

if __name__ == '__main__':
    maxPIMMSslices,Instr_Func,polimh,polimv,polimh1,polimh2,polimv1,polimv2 = setup_threads(Instr_Func,polimh,polimv,polimh1,polimh2,polimv1,polimv2)
    ###################################################################################
    # END OF A SINGLE RUN
    ###################################################################################

    # This is a weird way of dealing with slices so that we keep only those with intesity
    top_keep, bottom_keep = 49 + maxPIMMSslices + 1, 49 - maxPIMMSslices
    polimh1, polimh2, polimv1, polimv2, Instr_Func,= polimh1[bottom_keep:top_keep], polimh2[bottom_keep:top_keep], polimv1[bottom_keep:top_keep], polimv2[bottom_keep:top_keep], Instr_Func[bottom_keep:top_keep]
    polimh, polimv = polimh[bottom_keep:top_keep], polimv[bottom_keep:top_keep]

    print('Beginning gaussian blurring of functions')
    if BFlag:
        if not os.path.exists('VerticalSum'):
            os.mkdir('VerticalSum')
        if not os.path.exists('HorizontalSum'):
            os.mkdir('HorizontalSum')
        if not os.path.exists('VerticalDiff'):
            os.mkdir('VerticalDiff')
        if not os.path.exists('HorizontalDiff'):
            os.mkdir('HorizontalDiff')
    else:
        if not os.path.exists('Vertical'):
            os.mkdir('Vertical')
        if not os.path.exists('Horizontal'):
            os.mkdir('Horizontal')
    if not os.path.exists('Instr_Func'):
        os.mkdir('Instr_Func')

    # Calculate Blurring Parameters
    Radial_blur = np.sqrt(-np.log(Tolc_Blur)/np.log(2.0))*fwhmblur
    Blur_limit = int(Radial_blur)
    blur_const = np.log(2.0)/(fwhmblur**2)
    blur_sigma = 1/np.sqrt(2*blur_const)

    # Blur Images and Save Images
    for slices in range(top_keep-bottom_keep):
        if BFlag:
            polimh1_B[slices] = gf(polimh1[slices], sigma = blur_sigma, truncate = Blur_limit)
            polimh2_B[slices] = gf(polimh2[slices], sigma = blur_sigma, truncate = Blur_limit)
            polimv1_B[slices] = gf(polimv1[slices], sigma = blur_sigma, truncate = Blur_limit)
            polimv2_B[slices] = gf(polimv2[slices], sigma = blur_sigma, truncate = Blur_limit)
        else:
            polimh_B[slices] = gf(polimh, sigma = blur_sigma, truncate = Blur_limit)
            polimv_B[slices] = gf(polimv, sigma = blur_sigma, truncate = Blur_limit)

        Instr_Func_B[slices] = gf(Instr_Func[slices], sigma = blur_sigma, truncate = Blur_limit)

        if BFlag:
            simsumh = polimh1_B[slices] + polimh2_B[slices]
            simsumv = polimv1_B[slices] + polimv2_B[slices]
            simdiffh = polimh1_B[slices] - polimh2_B[slices]
            simdiffv = polimv1_B[slices] - polimv2_B[slices]

            np.savetxt('VerticalSum/Slice{}.dat'.format(slices), simsumv, delimiter = '\t',fmt='%.6f')
            np.savetxt('HorizontalSum/Slice{}.dat'.format(slices), simsumh, delimiter = '\t',fmt='%.6f')
            np.savetxt('VerticalDiff/Slice{}.dat'.format(slices), simdiffv, delimiter = '\t',fmt='%.6f')
            np.savetxt('HorizontalDiff/Slice{}.dat'.format(slices), simdiffh, delimiter = '\t',fmt='%.6f')
        else:
            np.savetxt('Horizontal/Slice{}.dat'.format(slices), polimh_B, delimiter = '\t',fmt='%.6f')
            np.savetxt('Vertical/Slice{}.dat'.format(slices), polimv_B, delimiter = '\t',fmt='%.6f')

        # np.savetxt('Instr_Func/Slice{}.dat'.format(slices), Instr_Func_B[slices],delimiter = '\t',fmt='%.6f')

    runtime = time()-st
    print(f'Total runtime: {round(runtime,5)}s')
    print(f'Runtime per step: {round(runtime/ntraj,5)}s')
    # plt.imshow(Instr_Func_B[37].T, origin = 'lower', cmap = 'gnuplot')
    # plt.show()