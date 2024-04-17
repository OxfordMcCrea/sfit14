# -*- coding: utf-8 -*-
"""
Program: Slice Intensity Evaluater

Description: Uses the PEISI slices (in an [x, y, z] format) and the instrument
function to create the VIP, VOOP, HIP, and HOOP intenisty profiles as seen in

Suits, A. G., et al. (2015). "Direct Extraction of Alignment Moments from
Inelastic Scattering Images." Journal of Physical Chemistry A 119(23):
5925-5931.

The input files must be named ND3Sim{}.dat with {} representing the slice, in
which low slice numbers represent negative z velocities (ie in PEISI, the
first slices recorded).

Author: Max McCrea

Date of Completion: 16/03/2022
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.special import sph_harm
from s_fitparameters import *
from functions import Vec_Ang, Vec_Ang_Sing, HUND_ENERGY
from scipy.interpolate import griddata

print('What is the polarisation of the laser used?')
print('V(v) = Vertical')
print('H(h) = Horizontal')
laser_polarisation = input().lower()
if not laser_polarisation in ['h', 'v']:
    raise SyntaxError('Invalid Laser Polarisation')
elif laser_polarisation in ['h']:
    H_Slice = 'HIP'
    V_Slice = 'H60'
    path = 'HorizontalDiff'
else:
    H_Slice = 'VIP'
    V_Slice = 'V60'
    path = 'VerticalDiff'

if not os.path.exists('Slice Collection Diff'):
    os.mkdir('Slice Collection Diff')

# Image size parameters
tot_img = 2*img_size + 1
final_tot_image = tot_img + 100

# Initilising Arrays to Read In
alleimage = np.zeros((sliceno, tot_img, tot_img))
Instr_Func = np.zeros((sliceno, tot_img, tot_img))


# Data is read in and adjusted to make it correct compared to gnuplot
# This include the swapaxes part at the end
for slices in range(sliceno):
    with open(f'{path}/ND3Sim{slices}.dat', 'r') as image:
        for nx in range(tot_img):
            for ny in range(tot_img):
                data = list(filter(None, image.readline().split('\t')))
                try:
                    data[-1] = data[-1].strip()
                except IndexError:
                    print(data, nx, ny)
                    sys.exit()
                alleimage[slices][nx][-ny] = float(data[2])
                '''if slices == 6 and nx == 201 and ny == 112:
                    print(alleimage[slices][nx][-ny])
                    sys.exit()'''
    with open('Instr_Func/hit_bas{}.dat'.format(slices), 'r') as image:
        for nx in range(tot_img):
            for ny in range(tot_img):
                data = list(filter(None, image.readline().split('\t')))
                try:
                    data[-1] = data[-1].strip()
                except IndexError:
                    print(data, nx, ny)
                    sys.exit()
                Instr_Func[slices][nx][-ny] = float(data[2])

alleimage = alleimage.swapaxes(1,2) # experimental image
Instr_Func = Instr_Func.swapaxes(1,2) # instrument function


# relative velocity vector defined (where k = V_NO - V_Rg for convenience) and
# the angle between it and the x axis calculated. The a rotation matrix is
# produced to convert coordinates from LAB frame defined by molecular
# beams to TOF frame.
relate = np.array([-V_Rg, V_NO, 0]) # relative velocity vector

Ecoll = 0.5*mu*(np.linalg.norm(relate)**2)/1.9865e-23
Ecollprimemax = Ecoll - HUND_ENERGY(SpinO, ROTCONST, jprime, Final_state_designation)/1.9865e-23
rel_ang = TPI - Vec_Ang_Sing(relate, np.array([1.0, 0.0, 0.0])) # Angle between k and x axis in LAB frame
rot_matrix = np.array([[np.cos(rel_ang), -np.sin(rel_ang)], [np.sin(rel_ang), np.cos(rel_ang)]]) # Rotation matrix to convert between frames (z not chnaged hence a 2D matrix)

# Divide the experimental image by the instrument function for horizontal
# central slice and copy central slice image for purposes later on
alleimage = np.divide(alleimage, Instr_Func, out = np.zeros_like(alleimage), where = Instr_Func != 0) # Newton_Sphere

if sliceno % 2 == 0:
    Central_Slice = np.copy((alleimage[int(sliceno/2)]+alleimage[int(sliceno/2)-1])/2)
else:
    Central_Slice = np.copy(alleimage[int(sliceno/2)])

# Define the arrays to contain the unstretched rotated x and y values
xrs = np.zeros((tot_img, tot_img))
yrs = np.zeros((tot_img, tot_img))

# Matrix multiplication of the rotation matrix and the x and y coordinates of any given image to rotate into TOF frame (k colinear with x)
for x in range(tot_img):
    for y in range(tot_img):
        xrs[x][y], yrs[x][y] = np.matmul(rot_matrix, np.array([x-img_size, y-img_size]))

# Stretching factors from ellipse program applied to unstretch

xrs, yrs = xrs/rstretchx, yrs/rstretchy

# Produce a 3D array of the Newton Sphere containing the x, y and z coordinates proportional to velocity space in the positions measured experimentally
x_cartesian_along_k = np.tile(xrs, (sliceno, 1, 1))
y_cartesian_along_k = np.tile(yrs, (sliceno, 1, 1))
z_cartesian_along_k = np.linspace(-int(sliceno/2), int(sliceno/2), sliceno)*25/VZtoT/PIX2V # Convert slice number into first a velocity and then into pixel space
z_cartesian_along_k = np.tile(z_cartesian_along_k, (tot_img, tot_img, 1)).T



# Flatten arrays for easier indexing
x_cartesian_along_k, y_cartesian_along_k, z_cartesian_along_k, alleimage = np.ndarray.flatten(x_cartesian_along_k), np.ndarray.flatten(y_cartesian_along_k), np.ndarray.flatten(z_cartesian_along_k), np.ndarray.flatten(alleimage)

##############################################################################
# TEST BLOCK FOR 60Degree Slice

# Set angle to Rotate, then define Rotation Matrix with that angle
angle = -60*PI/180
angle = 0.5*np.pi - angle
rot = np.array([[1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]])

position_vector = np.vstack((x_cartesian_along_k, y_cartesian_along_k, z_cartesian_along_k)).T

# Matrix Multiplication To convert into a coordinate system rotated by angle
x_rotated = np.sum(rot[0]*position_vector, axis = 1)
y_rotated = np.sum(rot[1]*position_vector, axis = 1)
z_rotated = np.sum(rot[2]*position_vector, axis = 1)

# Find edge of Newton Sphere by only looking at non-zero values, then get take same thickness slice as for time
non_zero = np.not_equal(alleimage, 0)
maxdistfromk = np.max(np.abs(y_rotated[non_zero]))
angled_slice = np.less(np.abs(y_rotated), maxdistfromk/(sliceno*2))

# Only Keep those coordinates in the desired slice
x_rotated, y_rotated, z_rotated, alleimage = x_rotated[angled_slice], y_rotated[angled_slice], z_rotated[angled_slice], alleimage[angled_slice]

# Create Image of slice
grid = np.zeros((501, 501))
np.add.at(grid, (np.around(x_rotated+250).astype(int), np.around(z_rotated+250).astype(int)), alleimage)
plt.figure(figsize = (10, 10))
plt.pcolormesh(grid.T, cmap = 'twilight', shading = 'gouraud')
plt.clim(-1, 1)
plt.show()

# Calculate sin(phiT)
sin_phiT_higher = np.sin(angle)
sin_phiT_lower = - sin_phiT_higher

# Calculate Scattering Angle and energy of every point
theta = np.abs(np.arctan2(z_rotated, x_rotated))*180/PI
energy = np.sqrt(x_rotated*x_rotated + z_rotated*z_rotated)*PIX2V
energy = Ecollprimemax - 0.5*mu*(energy**2)/(mass_const**2)/1.9865e-23

# Either side of k (along x axis) has a different value of phiT
lower_side, top_side = np.less_equal(z_rotated, 0), np.greater_equal(z_rotated, 0)

# Create Grid for interpolation
energy_grid, theta_grid = np.mgrid[0:int(np.max(energy)) + 1, 0:180.5:0.5]

# Interpolate to Find the intensity in each section of slice
# Notice division by sin(phiT) to normalise each theta intensity distribution
# See Equation 32 in Supplementary Information of Nat. Chem. Paper 2019
I_Top = griddata((energy.flatten()[top_side], theta.flatten()[top_side]), alleimage.flatten()[top_side]/sin_phiT_higher, (energy_grid, theta_grid), method = 'linear')
I_Lower = griddata((energy.flatten()[lower_side], theta.flatten()[lower_side]), alleimage.flatten()[lower_side]/sin_phiT_lower, (energy_grid, theta_grid), method = 'linear')

# Plot DCS as function of energy and angle
plt.pcolormesh(energy_grid, theta_grid, I_Lower, shading = 'auto', cmap = 'twilight')
plt.xlabel('Internal Energy of ND$_3$ / $cm^{-1}$')
plt.ylabel('Scattering Angle / Degrees')
plt.clim(-np.nanmax(np.abs(I_Top)), np.nanmax(np.abs(I_Top)))
plt.colorbar()
plt.show()

# Save Data to File
np.savetxt(f'Slice Collection Diff/{V_Slice}Top.dat'.format(V_Slice), I_Top)
np.savetxt(f'Slice Collection Diff/{V_Slice}Bottom.dat'.format(V_Slice), I_Lower)
###############################################################################

# Plot Horizontal Central Slice (HIP and VIP)
plt.figure(figsize = (7, 5))
plt.pcolormesh(Central_Slice.T, cmap = 'twilight')
plt.clim(-np.max(np.abs(Central_Slice)), np.max(np.abs(Central_Slice)))
plt.colorbar()
plt.show()

##############################################################################
# Now is time to convert the divided thorugh central slice into intensity as
# a function of theta and energy
#
# Similar procedure for the vertical image.
theta = np.abs(np.arctan2(yrs, xrs)*180/PI)
energy = np.sqrt(yrs*yrs + xrs*xrs)*PIX2V
energy = Ecollprimemax - 0.5*mu*(energy**2)/(mass_const**2)/1.9865e-23
fast_side, slow_side = np.less_equal(yrs.flatten(), 0), np.greater_equal(yrs.flatten(), 0)

# Interpolation to grid
energy_grid, theta_grid = np.mgrid[0:int(np.max(energy)) + 1, 0:180.5:0.5]
I_Slow = griddata((energy.flatten()[slow_side], theta.flatten()[slow_side]), Central_Slice.flatten()[slow_side], (energy_grid, theta_grid), method = 'linear')
I_Fast = griddata((energy.flatten()[fast_side], theta.flatten()[fast_side]), -Central_Slice.flatten()[fast_side], (energy_grid, theta_grid), method = 'linear')

# Plot it
plt.pcolormesh(energy_grid, theta_grid, I_Slow, shading = 'auto', cmap = 'twilight')
plt.xlabel('Internal Energy of ND$_3$ / $cm^{-1}$')
plt.ylabel('Scattering Angle / Degrees')
plt.clim(-np.nanmax(np.abs(I_Slow)), np.nanmax(np.abs(I_Slow)))
plt.colorbar()
plt.show()
np.savetxt('Slice Collection Diff/{}Slow.dat'.format(H_Slice), I_Slow)
np.savetxt('Slice Collection Diff/{}Fast.dat'.format(H_Slice), I_Fast)
##############################################################################
