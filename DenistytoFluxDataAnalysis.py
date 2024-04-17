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
    V_Slice = 'HOOP'
    path = 'HorizontalSum'
else:
    H_Slice = 'VIP'
    V_Slice = 'VOOP'
    path = 'VerticalSum'

if not os.path.exists('Slice Collection'):
    os.mkdir('Slice Collection')

# Image size parameters
tot_img = 2*img_size + 1
final_tot_image = tot_img + 200

# Initilising Arrays to Read In
alleimage = np.zeros((sliceno, tot_img, tot_img))
Instr_Func = np.zeros((sliceno, tot_img, tot_img))

# Data is read in and adjusted to make it correct compared to gnuplot
# This include the swapaxes part at the end
# for slices in range(sliceno):
#     with open(f'{path}/Slice{slices}.dat', 'r') as image:
#         for nx in range(tot_img):
#             for ny in range(tot_img):
#                 data = list(filter(None, image.readline().split('\t')))
#                 try:
#                     data[-1] = data[-1].strip()
#                 except IndexError:
#                     print(data, nx, ny)
#                     sys.exit()
#                 alleimage[slices][nx][-ny] = float(data[2])
#     with open('Instr_Func/Slice{}.dat'.format(slices), 'r') as image:
#         for nx in range(tot_img):
#             for ny in range(tot_img):
#                 try:
#                     data = list(filter(None, image.readline().split('\t')))
#                 except OSError:
#                     print(nx, ny, slices)
#                     sys.exit()
#                 try:
#                     data[-1] = data[-1].strip()
#                 except IndexError:
#                     print(data, nx, ny)
#                     sys.exit()
#                 Instr_Func[slices][nx][-ny] = float(data[2])

# alleimage = alleimage.swapaxes(1,2) # experimental image
# Instr_Func = Instr_Func.swapaxes(1,2) # instrument function

# Data is read in and adjusted to make it correct compared to gnuplot
# This include the swapaxes part at the end
for slices in range(sliceno):
    alleimage[slices] = np.loadtxt(f'{path}/Slice{slices}.dat')
    Instr_Func[slices] = np.loadtxt(f'Instr_Func/Slice{slices}.dat')

# relative velocity vector defined (where k = V_NO - V_Rg for convenience) and
# the angle between it and the x axis calculated. The a rotation matrix is
# produced to convert coordinates from LAB frame defined by molecular
# beams to TOF frame.
relate = np.array([V_NO, -V_Rg, 0]) # relative velocity vector

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

'''fast_side = np.greater(y_cartesian_along_k, 0)
x_cartesian_along_k = x_cartesian_along_k[fast_side]
y_cartesian_along_k = y_cartesian_along_k[fast_side]
z_cartesian_along_k = z_cartesian_along_k[fast_side]
alleimage = alleimage[fast_side]'''

# Find indices where the divided through image isn't equal to 0. Then using
# the maximum y value of those to calculate the maximum distance from k while
# still being within the Newton Sphere. We then only allow those within the
# the width of a central slice (so both vertical and horizontal central slices
# have the same width).
non_zero = np.not_equal(alleimage, 0)
maxdistfromk = np.max(np.abs(y_cartesian_along_k[non_zero]))
vert_slice = np.less(np.abs(y_cartesian_along_k), maxdistfromk/(sliceno*2))
x_cartesian_along_k, y_cartesian_along_k, z_cartesian_along_k, alleimage = x_cartesian_along_k[vert_slice], y_cartesian_along_k[vert_slice], z_cartesian_along_k[vert_slice], alleimage[vert_slice]

##############################################################################
# This section is all about creating a variable that represents the intensity
# of the vertical slice as a function of theta and energy. The purpose of this
# is to create an analogue of the I(theta) variable in Suits, A. G., et al.
# (2015). "Direct Extraction of Alignment Moments from Inelastic Scattering
# Images." Journal of Physical Chemistry A 119(23): 5925-5931
#
# First we calculate theta and the energy of every point in the array. For
# energy, velocity is calculated first, then converted into energy. The
# velocity is the COM velocity of NO, not the final relative velocity (they are
# colinear though)
theta = np.abs(np.arctan2(z_cartesian_along_k, x_cartesian_along_k))*180/PI
energy = np.sqrt(x_cartesian_along_k*x_cartesian_along_k + z_cartesian_along_k*z_cartesian_along_k)*PIX2V
energy = Ecollprimemax - 0.5*mu*(energy**2)/(mass_const**2)/1.9865e-23

top_side, bottom_side = np.greater_equal(z_cartesian_along_k.flatten(), 0), np.less_equal(z_cartesian_along_k.flatten(), 0)
# Interpolation to fill the grid
energy_grid, theta_grid = np.mgrid[0:int(np.max(energy)) + 1, 0:180.5:0.5]
I_Top = griddata((energy.flatten()[top_side], theta.flatten()[top_side]), alleimage.flatten()[top_side], (energy_grid, theta_grid), method = 'linear')
I_Bottom = griddata((energy.flatten()[bottom_side], theta.flatten()[bottom_side]), alleimage.flatten()[bottom_side], (energy_grid, theta_grid), method = 'linear')
# Plot it
plt.pcolormesh(energy_grid, theta_grid, I_Top, shading = 'auto', cmap = 'twilight')
plt.xlabel('Internal Energy of ND$_3$ / $cm^{-1}$')
plt.ylabel('Scattering Angle / Degrees')
plt.clim(-np.nanmax(np.abs(I_Top)), np.nanmax(np.abs(I_Top)))
plt.colorbar()
plt.show()

np.savetxt('Slice Collection/{}Top.dat'.format(V_Slice), I_Top)
np.savetxt('Slice Collection/{}Bottom.dat'.format(V_Slice), I_Bottom)
##############################################################################

# Convert into pixel numbers again, this time the image is made larger to fit
# points that were in the corner of the image, but no longer are.
x_final , y_final = np.around(x_cartesian_along_k).astype(int) + int((final_tot_image - 1)/2), np.around(z_cartesian_along_k).astype(int) + int((final_tot_image - 1)/2)

# Initialising arrays for vertical slice
vert_slice_point_count = np.zeros((final_tot_image, final_tot_image))
vert_slice_image = np.zeros((final_tot_image, final_tot_image))

# vert_slice_img_count marks the number of times a pixel has a value added to
# it, so that a mean can be taken. The other array is the vertical slice itself
np.add.at(vert_slice_point_count, (x_final, y_final), 1)
np.add.at(vert_slice_image, (x_final, y_final), alleimage)

# The mean is now taken for those points with multiple values
vert_slice_image = np.divide(vert_slice_image, vert_slice_point_count, out = np.zeros_like(vert_slice_image), where = vert_slice_point_count != 0)

# Plot the image without interpolation
plt.figure(figsize = (7, 5))
plt.pcolormesh(vert_slice_image.T, cmap = 'magma', shading = 'gouraud')
plt.clim(0, 10)
plt.colorbar()
plt.show()

# Prepare 1D arrays of image to be able to perform interpolation
img_x = (np.tile(np.linspace(0, final_tot_image-1, final_tot_image), (final_tot_image, 1)).T).flatten()
img_y = np.tile(np.linspace(0, final_tot_image-1, final_tot_image), (final_tot_image, 1)).flatten()
vert_slice_image = vert_slice_image.flatten()
vert_slice_point_count = vert_slice_point_count.flatten()

# Find where the point count was and wasn't 0. The points where it is zero are
# actually undefined points as we don't have data for that region. Thus we
# interpolate those data points from those left over.
non_zero = np.not_equal(vert_slice_point_count, 0)
img_x, img_y, vert_slice_image = img_x[non_zero], img_y[non_zero], vert_slice_image[non_zero]

# Define the final image grid onto which interpolated data is calculted. Then
# interpolaion occurs using the 'cubic' method
xgrid, ygrid = np.mgrid[0:final_tot_image, 0:final_tot_image]
interpdata = griddata((img_x, img_y), vert_slice_image, (xgrid, ygrid), method = 'linear', fill_value = 0)

# Plot interpolated data (vertical slice, HOOP and VOOP)
plt.figure(figsize = (7, 5))
plt.pcolormesh(interpdata.T, cmap = 'magma')
plt.colorbar()
plt.show()

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
I_Fast = griddata((energy.flatten()[fast_side], theta.flatten()[fast_side]), Central_Slice.flatten()[fast_side], (energy_grid, theta_grid), method = 'linear')

# Plot it
plt.pcolormesh(energy_grid, theta_grid, I_Slow, shading = 'auto', cmap = 'twilight')
plt.xlabel('Internal Energy of ND$_3$ / $cm^{-1}$')
plt.ylabel('Scattering Angle / Degrees')
plt.clim(-np.nanmax(np.abs(I_Slow)), np.nanmax(np.abs(I_Slow)))
plt.colorbar()
plt.show()
np.savetxt('Slice Collection/{}Slow.dat'.format(H_Slice), I_Slow)
np.savetxt('Slice Collection/{}Fast.dat'.format(H_Slice), I_Fast)
##############################################################################
