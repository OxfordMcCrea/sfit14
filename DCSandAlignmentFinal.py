# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:22:55 2022

@author: mb
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import simps
from s_fitparameters import *
from functions import hfact, Depolarisation_coeff, Vec_Ang_Sing, HUND_ENERGY

print('Which Parts of The Newton Sphere Should We Use?')
print('')
print('Fast/Slow of Horizontal Slice, Top/Bottom of Vertical Slice')
print('')
print('0: Whole')
print('1: Slow/Top')
print('2: Slow/Bottom')
print('3: Fast/Top')
print('4: Fast/Bottom')
print('5: Slow/Mean')
print('6: Fast/Mean')
print('7: Mean/Top')
print('8: Mean/Bottom')
Sphere_Part = int(input())

if Sphere_Part == 1:
    suffix1, suffix2 = 'Slow', 'Top'
elif Sphere_Part == 2:
    suffix1, suffix2 = 'Slow', 'Bottom'
elif Sphere_Part == 3:
    suffix1, suffix2 = 'Fast', 'Top'
elif Sphere_Part == 4:
    suffix1, suffix2 = 'Fast', 'Bottom'
elif Sphere_Part == 5:
    suffix1 = 'Slow'
elif Sphere_Part == 6:
    suffix1 = 'Fast'
elif Sphere_Part == 7:
    suffix2 = 'Top'
elif Sphere_Part == 8:
    suffix2 = 'Bottom'

oangle = 0

if Pure_Branch:
    branch_val = 1.0
else:
    branch_val = 0.0

Dep_fact1 = Depolarisation_coeff(jprime, 1.0, 1)
Dep_fact2 = Depolarisation_coeff(jprime, 1.0, 2)

if QFlag:
    hcoe1 = (1 + hfact(1, jprime, 1.0)) * Dep_fact1
    hcoe2 = (1 + hfact(2, jprime, 1.0)) * Dep_fact2
else:
    hcoe1 = hfact(1, jprime, branch_val) * Dep_fact1
    hcoe2 = hfact(2, jprime, branch_val) * Dep_fact2

c1 = 1/np.sqrt(jprime*(jprime+1))
c2 = np.sqrt((2.0*jprime+3.0)*(2.0*jprime-1.0)/(jprime*(jprime+1.0)))

# Index
# 0 = HIP
# 1 = HOOP
# 2 = VIP
# 3 = VOOP

#theta = np.array([1.409944, 1.409944, 1.409944,1.409944])
kp = np.array([-1/NPSQ2, 1/NPSQ2, 0])
theta = Vec_Ang_Sing(np.array([V_NO, V_Rg, 0]), kp)
print(theta*180/np.pi)
theta = np.tile(np.pi/2, 4)
phi = np.array([np.pi, 3*np.pi/2, np.pi, 3*np.pi/2])
chi = np.array([np.pi, np.pi, np.pi/2, np.pi/2])

sin_theta, cos_theta, sin_2theta, cos_2theta = np.sin(theta), np.cos(theta), np.sin(2*theta), np.cos(2*theta)
sin_phi, cos_phi, sin_2phi, cos_2phi = np.sin(phi), np.cos(phi), np.sin(2*phi), np.cos(2*phi)
sin_2chi, cos_2chi = np.sin(2*chi), np.cos(2*chi)

def dmomres(cos_2chi,sin_2chi,oangle):
    cos_2oang = np.cos(2.0 * oangle)
    sin_2oang = np.sin(2.0 * oangle)
    f11 = 1.5 * hcoe1 * c1 * sin_theta * sin_phi * sin_2oang
    f20 = 0.25 * hcoe2 * c2 * (3 * (sin_theta**2) * cos_2chi * cos_2oang - (3 * (cos_theta**2) - 1))
    f21 = np.sqrt(3)*c2*0.25*hcoe2*(2*(sin_theta*cos_phi*sin_2chi + sin_theta*cos_theta*sin_phi*cos_2chi)*cos_2oang - sin_2theta*cos_phi)
    f22 = 0.25*np.sqrt(3)*c2*hcoe2*((((cos_theta**2)+1)*cos_2phi*cos_2chi - 2*cos_theta*sin_2phi*sin_2chi)*cos_2oang - (sin_theta**2)*cos_2phi)
    return f11, f20, f21, f22


_, f20, f21, f22 = dmomres(cos_2chi,sin_2chi,oangle)
#f21 = np.round(f21, 0)

low_energy = 100
high_energy = 120
path = 'Slice Collection'

if Sphere_Part in [1,2,3,4]:
    VOOP = np.loadtxt(f'{path}/VOOP{suffix2}.dat')
    VIP = np.loadtxt(f'{path}/VIP{suffix1}.dat')
    HIP = np.loadtxt(f'{path}/HIP{suffix1}.dat')
    HOOP = np.loadtxt(f'{path}/HOOP{suffix2}.dat')
elif Sphere_Part in [5, 6]:
    VOOP = (np.loadtxt(f'{path}/VOOPTop.dat') + np.loadtxt(f'{path}/VOOPBottom.dat'))/2
    VIP = np.loadtxt(f'{path}/VIP{suffix1}.dat')
    HIP = np.loadtxt(f'{path}/HIP{suffix1}.dat')
    HOOP = (np.loadtxt(f'{path}/HOOPTop.dat') + np.loadtxt(f'{path}/HOOPBottom.dat'))/2
elif Sphere_Part in [7, 8]:
    VOOP = np.loadtxt(f'{path}/VOOP{suffix2}.dat')
    VIP = (np.loadtxt(f'{path}/VIPSlow.dat') + np.loadtxt(f'{path}/VIPFast.dat'))/2
    HIP = (np.loadtxt(f'{path}/HIPSlow.dat') + np.loadtxt(f'{path}/HIPFast.dat'))/2
    HOOP = np.loadtxt(f'{path}/HOOP{suffix2}.dat')
else:
    VOOP = (np.loadtxt(f'{path}/VOOPTop.dat') + np.loadtxt(f'{path}/VOOPBottom.dat'))/2
    VIP = (np.loadtxt(f'{path}/VIPSlow.dat') + np.loadtxt(f'{path}/VIPFast.dat'))/2
    HIP = (np.loadtxt(f'{path}/HIPSlow.dat') + np.loadtxt(f'{path}/HIPFast.dat'))/2
    HOOP = (np.loadtxt(f'{path}/HOOPTop.dat') + np.loadtxt(f'{path}/HOOPBottom.dat'))/2

Quantum = np.loadtxt('rpddcs_jp8.dat').T[:2]

alignment = np.loadtxt('ApseModelPDDCS8.dat').T

# H = np.nanmean(np.dstack((HIP, HOOP)), axis = 2)
# sumimage = np.nansum(np.dstack((VIP, VOOP, H)), axis = 2)
# energy_grid, theta_grid = np.mgrid[0:426, 0:180.5:0.5]

# DCS = np.nanmean(np.dstack((VIP, VOOP, H)), axis = 2)
# A22plus = np.sqrt(3)*(VOOP-VIP)/sumimage
# A20 = (3*H-sumimage)/sumimage

'''Epart = HUND_ENERGY(SpinO, ROTCONST, jprime, Final_state_designation)/1.9865e-23
k = np.sqrt(V_NO**2 + V_Rg**2)
Ecoll = half_mu*(k**2)/1.9865e-23
Etop = int(np.floor((Ecoll-Epart)/25)*25 + 1)'''
Etop = (VIP.shape)[0]

energy_grid, theta_grid = np.mgrid[0:Etop, 0:180.5:0.5]

D_HIVI = f20[0]/f20[2]
D_VIVO = f20[2]/f20[3]
D_HIHO = f20[0]/f20[1]
D_HOVO = f20[1]/f20[3]

C_HV = (f21[0] - D_HIVI*f21[2])/(f21[1] - D_HOVO*f21[3])
C_IOP = (f21[0] - D_HIHO*f21[1])/(f21[2] - D_VIVO*f21[3])

B_HV = f22[0] - D_HIVI*f22[2] - C_HV*(f22[1] - D_HOVO*f22[3])
B_IOP = f22[0] - D_HIHO*f22[1] - C_IOP*(f22[2] - D_VIVO*f22[3])

A_HV = 1 - D_HIVI - C_HV*(1 - D_HOVO)
A_IOP = 1 - D_HIHO - C_IOP*(1 - D_VIVO)

DCS = ((1 - B_HV/B_IOP)*HIP - (C_HV - D_HIHO*B_HV/B_IOP)*HOOP - (D_HIVI - C_IOP*B_HV/B_IOP)*VIP + (C_HV*D_HOVO - C_IOP*D_VIVO*B_HV/B_IOP)*VOOP)/(A_HV - A_IOP*B_HV/B_IOP)

A22plus = (HIP - D_HIVI*VIP - C_HV*(HOOP - D_HOVO*VOOP))/(DCS*B_HV) - A_HV/B_HV

D_HIVI01 = f21[0]/f21[2]
D_HOVO01 = f21[1]/f21[3]

C_HV01 = (f22[0] - D_HIVI01*f22[2])/(f22[1] - D_HOVO01*f22[3])
A_HV01 = 1 - D_HIVI01 - C_HV01*(1 - D_HOVO01)
B_HV01 = f20[0] - D_HIVI01*f20[2] - C_HV01*(f20[1] - D_HOVO01*f20[3])

A20 = (HIP - D_HIVI01*VIP - C_HV01*(HOOP - D_HOVO01*VOOP))/(DCS*B_HV01) - A_HV01/B_HV01

C_HV12 = (f22[0] - D_HIVI*f22[2])/(f22[1] - D_HIVI*f22[3])
A_HV12 = 1 - D_HIVI - C_HV12*(1 - D_HOVO)
B_HV12 = f21[0] - D_HIVI*f21[2] - C_HV12*(f21[1] - D_HOVO*f21[3])

p21plus = (HIP - D_HIVI*VIP - C_HV12*(HOOP - D_HOVO*VOOP))/(DCS*B_HV12) - A_HV12/B_HV12

if not os.path.exists('Analysis Maps'):
    os.mkdir('Analysis Maps')
plt.pcolormesh(energy_grid, theta_grid, DCS[:Etop], cmap = 'magma', shading = 'auto')
plt.clim(0, np.nanmax(np.abs(DCS)))
cbar = plt.colorbar()
plt.xlabel(r'Internal Energy of ND$_{3}$')
plt.title('DCS')
cbar.set_label(r'$\frac{d\sigma}{d\omega}$ (Arbitrary Units)')
plt.ylabel(r'Scattering Angle ($\degree$)')
plt.savefig('Analysis Maps/DCS.png')
plt.clf()

plt.pcolormesh(energy_grid, theta_grid, A22plus[:Etop], cmap = 'twilight', shading = 'auto')
plt.clim(-10, 10)
plt.colorbar()
plt.savefig('Analysis Maps/rho22.png')
plt.clf()

plt.pcolormesh(energy_grid, theta_grid, A20[:Etop], cmap = 'twilight', shading = 'auto')
plt.clim(-np.nanmax(np.abs(A20)), np.nanmax(np.abs(A20)))
plt.colorbar()
plt.savefig('Analysis Maps/rho20.png')
plt.clf()

plt.pcolormesh(energy_grid, theta_grid, p21plus[:Etop], cmap = 'twilight', shading = 'auto')
plt.clim(-np.nanmax(np.abs(p21plus)), np.nanmax(np.abs(p21plus)))
plt.colorbar()
plt.savefig('Analysis Maps/rho21.png')
plt.clf()

plot1 = np.nanmean(DCS[low_energy:high_energy], axis = 0)
plot2 = np.nanmean(A22plus[low_energy:high_energy], axis = 0)
plot3 = np.nanmean(A20[low_energy:high_energy], axis = 0)
plot4 = np.nanmean(p21plus[low_energy:high_energy], axis = 0)

x = np.linspace(0, 180, 361)
x1, plot1 = x[np.invert(np.isnan(plot1))], plot1[np.invert(np.isnan(plot1))]
x2, plot2 = x[np.invert(np.isnan(plot2))], plot2[np.invert(np.isnan(plot2))]
x3, plot3 = x[np.invert(np.isnan(plot3))], plot3[np.invert(np.isnan(plot3))]
x4, plot4 = x[np.invert(np.isnan(plot4))], plot4[np.invert(np.isnan(plot4))]

baseint1 = simps(Quantum[1], Quantum[0])
baseint2 = simps(alignment[7], alignment[0])
baseint3 = simps(alignment[5], alignment[0])
baseint4 = simps(alignment[6], alignment[0])
baseint = np.array([baseint2, baseint3])

expint1 = simps(plot1, x1)
expint2 = simps(plot2, x2)
expint3 = simps(plot3, x3)
expint4 = simps(plot4, x4)
expint = np.array([expint2, expint3])

normalign = np.mean(baseint/expint)

# plot1 *= baseint1/expint1
plot2 *= normalign
plot3 *= normalign
plot4 *= normalign
if not os.path.exists('DataForPlots'):
    os.mkdir('DataForPlots')
np.savetxt('DCS.dat', np.vstack((x1, plot1)).T)
np.savetxt('rho22.dat', np.vstack((x2, plot2)).T)
np.savetxt('rho20.dat', np.vstack((x3, plot3)).T)
np.savetxt('rho21.dat', np.vstack((x4, plot4)).T)

print(f'####### {normalign} #######')
if not os.path.exists('Plotsv2'):
    os.mkdir('Plotsv2')
print(baseint1/expint1)
Quantumr00 = np.loadtxt('BackScattR00.dat').T[1]
plt.plot(x1, plot1/2, Quantum[0], Quantumr00)
plt.title(r'$R^{(0)}_{0}$')
plt.ylabel(r'$\frac{d\sigma}{d\omega}$ / Arbitrary Units')
plt.xlabel(r'Scattering Angle $(\theta)$ / $\degree$')
plt.xlim(0, 180)
plt.ylim(0, 1.2)
plt.savefig('Plotsv2/DCS.png')
plt.clf()
plt.plot(x2, plot2, alignment[0], alignment[7])
plt.title(r'$\rho^{\{{2}\}}_{2+}$')
plt.xlabel(r'Scattering Angle $(\theta)$ / $\degree$')
plt.ylabel(r'$\rho^{\{{2}\}}_{2+}$ / Arbitrary Units')
plt.xlim(0, 180)
#plt.ylim(-0.5, 0.5)
plt.savefig('Plotsv2/rho22.png')
plt.clf()
plt.plot(x3, plot3, alignment[0], alignment[5])
plt.title(r'$\rho^{\{2\}}_{0}$')
plt.xlabel(r'Scattering Angle $(\theta)$ / $\degree$')
plt.ylabel(r'$\rho^{\{{2}\}}_{0}$ / Arbitrary Units')
plt.xlim(0, 180)
#plt.ylim(-0.5, 0.5)
plt.savefig('Plotsv2/rho20.png')
plt.clf()
plt.plot(x4, plot4, alignment[0], alignment[6])
plt.title(r'$\rho^{\{2\}}_{1+}$')
plt.xlabel(r'Scattering Angle $(\theta)$ / $\degree$')
plt.ylabel(r'$\rho^{\{{2}\}}_{1+}$ / Arbitrary Units')
plt.xlim(0, 180)
#plt.ylim(-0.5, 0.5)
plt.savefig('Plotsv2/rho21.png')
plt.clf()


Energy_Dist = np.nanmean(DCS, axis = 1)
Energy_Dist_x = np.linspace(0, len(Energy_Dist)-1, len(Energy_Dist))
Energy_Dist_x, Energy_Dist = Energy_Dist_x[np.invert(np.isnan(Energy_Dist))], Energy_Dist[np.invert(np.isnan(Energy_Dist))]
Energy_Dist_Int = simps(Energy_Dist, Energy_Dist_x)

# Comp_Energ = np.loadtxt('GaussianPop.dat').T
# Energy_Levels = np.loadtxt('ND3_Levels.dat', delimiter = '\t', dtype = str).T[0].astype(float)
# Comp_Energ_Int = simps(Comp_Energ, Energy_Levels)

plt.plot(Energy_Dist_x, Energy_Dist/Energy_Dist_Int, label = 'Output')
# plt.plot(Energy_Levels, Comp_Energ/Comp_Energ_Int, label = 'Input')
plt.title('Energy Distribution')
plt.xlabel('$E_{int}$ of ND$_3$ / $cm^{-1}$')
plt.ylabel('P($E_{int}$)')
#plt.ylim(0, 0.002)
plt.legend()
plt.savefig('Plotsv2/EnergyDist.png')
plt.clf()


VIP =  np.nanmean(VIP[low_energy:high_energy], axis = 0)
HIP =  np.nanmean(HIP[low_energy:high_energy], axis = 0)
VOOP =  np.nanmean(VOOP[low_energy:high_energy], axis = 0)
HOOP =  np.nanmean(HOOP[low_energy:high_energy], axis = 0)

plt.plot(np.linspace(0, 180, 361), VIP, label = 'VIP')
plt.plot(np.linspace(0, 180, 361), HIP, label = 'HIP')
plt.plot(np.linspace(0, 180, 361), VOOP, label = 'VOOP')
plt.plot(np.linspace(0, 180, 361), HOOP, label = 'HOOP')
# plt.ylim(0, 0.5)


#plt.plot(np.linspace(0, 180, 361), VIP_should, label = 'Theory VIP')


plt.legend()
plt.savefig('Plotsv2/Slices.png')
# plt.savefig('SlicePlotter.png')
#plt.plot(np.linspace(0, 180, 361), Quantum[1])

