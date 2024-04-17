# Functions File for Data Analysis Codes

import numpy as np
from scipy.special import factorial
from sympy.physics.wigner import wigner_6j, wigner_3j
from scipy.special import sph_harm
import sys

def tranpose_1d(array):
    array = np.reshape(array,(len(array),1))
    return array

# Calculates Rotational Energy of NO 
# See page 303 of Zare's "Angular Momentum"
def HUND_ENERGY(Spin_orbit_constant, Rotational_constant, j, LambdaSO_label):
    Y = Spin_orbit_constant/ Rotational_constant
    X = np.sqrt(4*((j+0.5)**2)+Y*(Y-4))
    #Energy = Rotational_constant*((j-0.5)*(j+1.5)+0.5*X*(-1)**(LambdaSO_label))
    Energy = Rotational_constant*((j-0.5)*(j+1.5)+(X*(-1)**(LambdaSO_label)+Y-2)/2)
    return Energy

# Rist, C., et al. (1993). "Scattering of Nh3 by Ortho-H2 and Para-H2 - Expansion of the Potential and Collisional Propensity Rules." Journal of Chemical Physics 98(6): 4662-467
# Data From : Urban, S., et al. (1984). "Simultaneous Analysis of the Microwave and Infrared-Spectra of (Nd3)-N-14 and (Nd3)-N-15 for the Nu-2 Excited-State." Journal of Molecular Spectroscopy 106(1): 29-3
def ND3Hund(jprime, kprime, epsilon):
    A = 3.1142
    B = 5.1428
    DJ = 1.978e-4
    DJK = -3.49e-4
    DK = 2.001e-4
    E = B*jprime*(jprime+1) + (A-B)*kprime*kprime -DJ*(jprime**2)*((jprime+1))**2 -DJK*jprime*(jprime+1)*(kprime**2) - DK*(kprime**4)
    if epsilon == 1:
        return (E - 0.053/2)*0.12397849462365593
    else:
        return (E + 0.053/2)*0.
    
def HLFactor(j, Fin_LambdaSO_label, QFlag, SOconst, Rotconst):
    Y = SOconst/Rotconst
    X = np.sqrt(4*((j+0.5)**2)+Y*(Y-4))
    if Fin_LambdaSO_label in [1,2]:
        if QFlag == 'T':
            Q11 = (2*j+1)*((4*(j**2) + 4*j - 1) + (1/X)*(8*(j**3) + 12*(j**2) - 2*j -7 + 2*Y))/(32*j*(j+1))
            P21 = ((2*j+1)**2 - (2*j+1)*(1/X)*(4*(j**2) + 4*j -7 + 2*Y))/(32*j)
            return Q11*2, P21*2
        else:
            R21 = ((2*j+1)**2 - (2*j+1)*(1/X)*(4*(j**2) + 4*j + 1 - 2*Y))/(32*(j+1))
            return R21*2
    else:
        if QFlag == 'T':
            Q21 = (2*j+1)*((4*(j**2) + 4*j - 1) - (1/X)*(8*(j**3) + 12*(j**2) - 2*j + 1 - 2*Y))/(32*j*(j+1))
            R11 = ((2*j+1)**2 + (2*j+1)*(1/X)*(4*(j**2) + 4*j -7 + 2*Y))/(32*(j+1))
            return Q21*2, R11*2
        else:
            P11 = ((2*j+1)**2 + (2*j+1)*(1/X)*(4*(j**2) + 4*j + 1 - 2*Y))/(32*j)
            return P11*2

def threej(j1, j2, j3, m1, m2, m3):
    W = wigner_3j(j1, j2, j3, m1, m2, m3)
    return W



def PCIA(jprime, Fin_LambdaSO_label, QFlag, SOconst, Rotconst, kprime, Polarisation):
    if Polarisation == 'H':
        evector = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
    else:
        evector = np.array([0.0, 0.0, 1.0])
    Theta = Vec_Ang(kprime, evector)
    redwig = reduced_wigner(jprime, np.arange(-jprime, jprime+1, 1), 0.5, Theta).T
    threejs = np.loadtxt('ThreeJ.dat')[np.array([int(jprime-1.5), int(jprime - 1.5 + 16), int(jprime - 1.5 +32)])]
    threejsP = threejs[0]
    threejsQ = threejs[1]
    threejsR = threejs[2]
    if Fin_LambdaSO_label in [1,2]:
        if QFlag == 'T':
            HLQ11, HLP21 = HLFactor(jprime, Fin_LambdaSO_label, QFlag, SOconst, Rotconst)     
            summedQ11 = np.sum(((redwig.T)*threejsQ[threejsQ != 0])**2, axis = 1)
            summedP21 = np.sum(((redwig.T)*threejsP[threejsP != 0])**2, axis = 1)
            return HLQ11*summedQ11 + HLP21*summedP21
        else:
            HLR21 = HLFactor(jprime, Fin_LambdaSO_label, QFlag, SOconst, Rotconst)
            summedR21 = np.sum(((redwig.T)*threejsR[threejsR != 0])**2, axis = 1)
            return HLR21*summedR21
    else:
        if QFlag == 'T':
            HLQ21, HLR11 = HLFactor(jprime, Fin_LambdaSO_label, QFlag, SOconst, Rotconst)
            summedQ21 = np.sum(((redwig.T)*threejsQ[threejsQ != 0])**2, axis = 1)
            summedR11 = np.sum(((redwig.T)*threejsR[threejsR != 0])**2, axis = 1)
            return HLQ21*summedQ21 + HLR11*summedR11
        else:
            HLP11 = HLFactor(jprime, Fin_LambdaSO_label, QFlag, SOconst, Rotconst)
            summedP11 = np.sum(((redwig.T)*threejsP[threejsP != 0])**2, axis = 1)
            return HLP11*summedP11

def thetaminandmax(chifactor):
    x = np.zeros(2)
    x[0] = (np.cos(np.pi/2 - chifactor) + 1)/2
    x[1] = (np.cos(np.pi/2 + chifactor) + 1)/2
    return max(x), min(x)

# Calculates reduced Wigner coefficient
# Taken from eq. 3.57 on page 86 of Zare's Angular Momentum Book
def reduced_wigner(j, m, mprime, theta):
    Term_1 = np.sqrt(factorial(j+m)*factorial(j-m)*factorial(j+mprime)*factorial(j-mprime))
    # Term 2 is a sum over all values of nu for which the arguments of the 
    # factorials performed are nonnegative. 
    if isinstance(m, int):
        nu_min = int(max((0.0, (m-mprime))))
        nu_max = int(min((j-mprime),(j+m))) 
        Term_2 = 0
        for nu in range(nu_min, nu_max+1):
            Term_2 = Term_2 + (-1)**nu/(factorial(j-mprime-nu)*factorial(j+m-nu)*factorial(nu+mprime-m)*factorial(nu)) *((np.cos(theta/2))**(2*j+m-mprime-2*nu))*((np.sin(theta/2))**(mprime-m+2*nu))
    else:
        comp = np.vstack((np.zeros(len(m)), m-mprime))
        comp2 = np.vstack((np.tile((j-mprime), reps = (len(m),)),(j+m)))
        nu_min = np.max(comp, axis = 0)
        nu_max = np.min(comp2, axis = 0)
        ranges = nu_max - nu_min + 1
        Term_2 = np.zeros((len(m), len(theta)))
        for x in range(len(m)):
            nus = np.arange(nu_min[x], nu_max[x]+1, 1, dtype = int)
            for nus in nus:
                Term_2[x] += (-1)**nus/(factorial(j-mprime-nus)*factorial(j+m[x]-nus)*factorial(nus+mprime-m[x])*factorial(nus)) *((np.cos(theta/2))**(2*j+m[x]-mprime-2*nus))*((np.sin(theta/2))**(mprime-m[x]+2*nus))
    return Term_1*Term_2.T
#print(reduced_wigner(8.5, 0.5, 0.5, np.pi/2), reduced_wigner(8.5, 5.5, 0.5, np.pi/2),)
# Calculates proportionality constant for Clebsch_Gordon and 3j-symbols
# See pg 50 of Zare's Angular Momentum Book
# This is for a 3j-symbol with bottom right number = -m3 
# ie (j1  j2  j3) for <j1 m1, j2 m2|j3 m3>
#    (m1  m2 -m3)
def three_j_to_CG_Prop_Const(j1, j2, j3, m3):
    return ((-1)**(j1-j2+m3))*((2*j3+1)**(1/2))          

# Calculates wigner 6j symbols
def sixj(j1, j2, j3, j4, j5, j6):
    return float(wigner_6j(j1, j2, j3, j4, j5, j6))

# Calculates h factors (seemingly a butchery of eq 7.2 in Balasz Hornung's Thesis)
# BRING UP WITH MARK
def hfact(k, jinit, branch):
    jfinl = jinit + branch
    six1 = sixj(jinit, jinit, k, 1, 1, jfinl)
    six2 = sixj(jinit, jinit, k, 1, 1, jinit)
    return (six1/six2)*(-1.0)**(jinit-jfinl)

# Calculates Depolarisation Coefficients (See Page 159 of Balasz Hornung's Thesis) - Note 6j symbols are invariant to any permutation of columns (ie this gives same answer as layout in Balasz's thesis)
# ftot2 is total rotational quantum number (includes electron spin), inucl is nuclear spin (1/2 for 14N, 0 for 16O), order is rank of hfactor 
def Depolarisation_coeff(jprime, inucl, order):
    DEPFAC = 0.0
    for ftot in range(int(abs(jprime - inucl)), int(abs(jprime + inucl))+1):
        ftot2 = (ftot+0.5)*2
        DEPFAC = DEPFAC + ((2*ftot + 2.0)**2.0)*((sixj(ftot2/2, ftot2/2, order, jprime, jprime, inucl))**2.0)
    DEPFAC = DEPFAC / (2.0*inucl + 1.0)
    return DEPFAC

#Normalises Vectors
def Normalise(vector):
    norm_vec = (vector.T/np.linalg.norm(vector, axis = 1)).T
    return norm_vec

def Vec_Mag(Vec):
    sums = 0.0
    for i in range(len(Vec)):
        sums = sums + Vec[i]**2
    return np.sqrt(sums)

# Calculates the angle between two vectors        
def Vec_Ang(vec1, vec2):
    dot = np.divide(np.dot(vec1, vec2), (np.linalg.norm(vec1, axis = 1)*np.linalg.norm(vec2)), out = np.zeros(len(vec1)), where = (np.linalg.norm(vec1, axis = 1)*np.linalg.norm(vec2)) != 0)
    return np.arccos(dot) 

def Vec_Ang_Sing(vec1, vec2):
    vec1 = np.divide(vec1, np.linalg.norm(vec1), out = np.zeros_like(vec1), where = vec1 != 0)
    vec2 = np.divide(vec2, np.linalg.norm(vec2), out = np.zeros_like(vec2), where = vec2 != 0)
    dot = np.dot(vec1, vec2)
    return np.arccos(dot)

# This function claims to be (vec1 x vec2).vec3, but isn't. It in fact returns the sign of this projection.
def Cross_Then_Dot(vec1, vec2, vec3):
    vec4 = np.cross(vec1, vec2)
    vec5 = vec4*vec3
    print(vec5)
    norm = Vec_Mag(vec5)
    print(norm)
    if norm < 1e-8:
        prod = 1
    else:
        prod = (vec5[0]+vec5[1]+vec5[2])/norm
    return 1.0*(prod/abs(prod))

#Transposes a 3x3 matrix
def Transpose(Dict):
    OuterDict = {}
    for i in range(1,4):
        OuterDict[i] = []
        for j in range(0,3):
            OuterDict[i].append(Dict[j+1][i-1])
    return OuterDict

# Calculates the r226 and r308 factors by a sensible method that i think is right (only gives same answer when rlaser = 0)
def laser_projection(rdet, rlaser, proplaser):
    proplaser = proplaser/Vec_Mag(proplaser)
    rDET = rdet- rlaser
    magrDET = Vec_Mag(rDET)
    theta = np.arccos(np.dot(rDET, proplaser)/magrDET)
    return magrDET*np.sin(theta)

# Calculates the r226 and r308 factors by the method used in the old fortran program
def POLNDS(rdet, rlaser, proplaser):
    diff12 = np.array([])
    diff13 = np.array([])
    diff23 = np.array([])
    for i in range(0,3):
        diff12 = np.append(diff12, rdet[i] - rlaser[i])
        diff13 = np.append(diff13, rdet[i] - proplaser[i])
        diff23 = np.append(diff23, rlaser[i] - proplaser[i])
    aux1 = np.cross(diff12, diff23)       
    aux2 = np.dot(aux1, aux1)
    aux3 = np.dot(diff23, diff23)
    return np.sqrt(aux2/aux3)

# Multiplies a matrix in the format of a dictionary, by a vector in the form of a numpy array or list
def matrix_multiply(matrix, vector):
    output_vector = np.array([])
    for row in range(1,4):
        new_row = 0.0
        for column in range(0,3):
            new_row = new_row + matrix[row][column]*vector[column]
        output_vector = np.append(output_vector, new_row)
    return output_vector      

# Calculates the F functions (Balasz Hornung's Thesis page 151), with c1 removed for f11, c2 removed for f20, and 1/sqrt3*c2 removed for f21 and f22
def dmomres(cos_theta, sin_theta, sin_2theta, cos_phi, sin_phi, cos_2phi, sin_2chi, cos_2chi, sin_2phi, hcoe1, hcoe2, oangle):
    cos_2oang = np.cos(2.0*oangle)
    f11 = 1.5*hcoe1*sin_theta*sin_phi*np.sin(2.0*oangle)
    f20 = 0.25*hcoe2*(3*(sin_theta**2)*cos_2chi*cos_2oang - (3*(cos_theta**2)-1))
    f21 = 0.75*hcoe2*((2*sin_theta*cos_phi*sin_2chi + 2*sin_theta*cos_theta*sin_phi*cos_2chi)*cos_2oang - sin_2theta*cos_phi) #I THINK THIS ONE IS WRONG
    f22 = 0.75*hcoe2*((((cos_theta**2)+1)*cos_2phi*cos_2chi - 2*cos_theta*sin_2phi*sin_2chi)*cos_2oang - (sin_theta**2)*cos_2phi)
    return f11, f20, f21, f22

# Calculates Eq(4) in J. Chem. Phys. 138, 104310 (2013) I THINK A22 IS WRONG(MISSING A FACTOR OF SQRT(3) IN SECOND AND THIRD TERMS)
def dmomcal(cos_theta, sin_theta, sin_2theta, cos_phi, sin_phi, cos_2phi, sin_2chi, cos_2chi, sin_2phi, rho11, rho20, rho21, rho22):
    A10det = rho11*sin_theta*sin_phi
    A20det = 0.5*rho20*(3*(cos_theta**2) - 1.0) + 1.5*rho21*sin_2theta*cos_phi + 1.5*rho22*(sin_theta**2)*cos_2phi
    A22det = 0.5*rho20*(sin_theta**2)*cos_2chi + rho21*(sin_theta*cos_theta*sin_2chi + sin_theta*cos_theta*sin_phi*cos_2chi) + rho22*(0.5*((1.0+cos_theta**2)*cos_2phi*cos_2chi) - cos_theta*sin_2phi*sin_2chi)
    return A10det, A20det, A22det

def mod_sph_harm(k, q, theta):
    return np.sqrt(4*np.pi/(2*k+1))*sph_harm(q, k, 0, theta).real