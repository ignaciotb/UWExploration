#!/usr/bin/env python3

import sympy as sym
import numpy as np
import math
from scipy.spatial.transform import Rotation as rot

def plot_cov(mu, C, k):
    Pxy = C[0:2,0:2]
    eig_val, eig_vec = np.linalg.eig(Pxy)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # eig_val[big_ind] or eiq_val[small_ind] were occasionally negative
    # numbers extremely close to 0 (~10^-20), catch these cases and set
    # the respective variable to 0
    try:
        a = k * math.sqrt(eig_val[big_ind])
    except ValueError:
        a = 0

    try:
        b = k * math.sqrt(eig_val[small_ind])
    except ValueError:
        b = 0

    beta = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    rot_t = rot.from_euler('z', beta).as_dcm()[0:2, 0:2]
    e = np.sqrt(1 - (b/a)**2)

    px = []
    py = []
    for i in range(len(t)):
        theta = 2*np.pi*(i-1)/(len(t) -1)
        r = a*(1-e**2)/(1+e*np.cos(theta))
        z = np.matmul(rot_t, np.array([r*np.cos(theta) - (1-e)*a + a,
                                     r*np.sin(theta)]).reshape(2,1)) + np.array(mu[0:2]).astype(np.float64).reshape(2,1)

        px.append(z[0])
        py.append(z[1])

    return px, py


# Create sympy rotation matrix as Rz*Ry*Rx
def create_rot(Xrot):
    rot_mat = rot.from_euler("xyz", Xrot, degrees=False).as_dcm()
    return rot_mat

# Build a homogenous transformation matrix the standard way
def vec2homMat(p):
    rho = np.array(p[0:3])
    phi = np.array(p[3:6])
    
    # Rotation matrices
    Rxyz = create_rot(phi)

    T = np.eye(4)
    T[0:3, 0:3] = Rxyz 
    T[0:3, 3] = rho
    
    return T

def vec2homVec(p):
    p_hom = np.ones(4,1)
    p_hom[0:3, 0] = np.array(p)
    return p_hom

def transInv(T):

    T_inv = np.eye(4)
    T_inv[0:3,0:3] = T[0:3,0:3].transpose()
    T_inv[0:3,3] = -np.matmul(T_inv[0:3,0:3], T[0:3,3])

    return T_inv

# EXPT Build a transformation matrix using the exponential map, closed form
def vec2tran(p):
    rho = np.array(p[0:3])
    phi = np.array(p[3:6])
    
    C = vec2rot(phi.T)
    J = vec2jac(phi.T)

    T = np.eye(4)
    T[0:3, 0:3] = C
    T[0:3, 3] = np.matmul(J, rho.T)

    return T

# VEC2ROT Build a rotation matrix using the exponential map
# phi: 3x1 sym.matrix
def vec2rot(phi):
    tolerance = 1e-12
    angle = np.linalg.norm(phi)
    
    # If angle is too small, series representation
    if angle < tolerance:
        C = vec2rotSeries(phi, 10)

    else:
        axis = phi * (1./angle)
        cp = np.cos(angle)
        sp = np.sin(angle)

        C = np.eye(3) * cp + np.matmul(axis.reshape(3,1), axis.reshape(1,3)) * (1-cp) + hat(axis) * sp

    return C

# HAT builds the 3x3 skew symmetric matrix from the 3x1 input or 4x4 from 6x1 input
def hat(vec):
    if len(vec) == 3:
        vechat = np.array([[0., -vec[2], vec[1]],
                           [vec[2], 0., -vec[0]],
                           [-vec[1], vec[0], 0.]])

    elif len(vec) == 6:
        vechat = np.array([[hat(vec[3:6, 0]), vec[0:3, 0]],
                           [np.zeros((1,4))]])
    
    return vechat

# VEC2ROTSERIES Build a rotation matrix using the exponential map series with N elements in the series
def vec2rotSeries(phi, N):
    C = np.eye(3)
    xM = np.eye(3)
    cmPhi = hat(phi)
    for n in range(1, N):
        xM = np.matmul(xM, cmPhi) * (1./n)
        C += xM

    C = np.matmul(C, np.linalg.pinv((np.matmul(C.T, C))**0.5))
    rotValidate(C)

    return C

# TODO: this one has to work with numerical values
# VALIDATEROTATION causes an error if the rotation matrix is not orthonormal
def rotValidate(C):
    CtC = np.matmul(C.T, C)
    E = CtC - np.eye(3)
    err = np.max(abs(E))

    if err > np.e**-10:
        print("Rotation matrix not valid")

# VEC2JAC Construction of the 3x3 J matrix or 6x6 J matrix
def vec2jac(vec):
    tolerance = 1.e-12
    
    if len(vec) == 3:
        phi = vec
        ph = np.linalg.norm(phi)
        if ph < tolerance:
            J = vec2jacSeries(phi, 10)
        else:
            axis = phi * (1./ph)
            cph = (1 - np.cos(ph)) / ph
            sph = np.sin(ph)/ph
            J = np.eye(3) * sph + np.matmul(axis.reshape(3,1), axis.reshape(1,3)) * (1-sph) + hat(axis) * cph
    elif len(vec) == 6:
        rho = vec[0:3]
        phi = vec[3:6]
        ph = np.linalg.norm(phi)

        if ph < tolerance:
            J = vec2jacSeries(phi, 10)

        else:
            Jsmall = vec2jac(phi)
            Q = vec2Q(vec)
            J = np.block([[Jsmall, Q],
                          [sym.zeros(3), Jsmall]])

    return J

# VEC2JACSERIES Construction of the J matrix from Taylor series
def vec2jacSeries(vec, N):
    if len(vec) == 3:
        J = np.eye(3)
        pxn = np.eye(3)
        px = hat(vec)
        for n in range(N):
            pxn = np.matmul(pxn, px) * (1./(n+1))
            J = J + pxn

    elif len(vec) == 6:
        J = np.eye(6)
        pxn = np.eye(6)
        px = curlyhat(vec)
        for n in range(N):
            pxn = np.matmul(pxn, px) * (1./(n+1))
            J = J + pxn
    
    return J

# VEC2Q Construction of the 3x3 Q matrix
def vec2Q(vec):
    rho = vec[0:3]
    phi = vec[3:6]

    ph = np.linalg.norm(phi)
    ph2 = ph * ph
    ph3 = ph2 * ph
    ph4 = ph3 * ph
    ph5 = ph4 * ph

    cph = np.cos(ph)
    sph = np.sin(ph)

    rx = hat(rho)
    px = hat(phi)

    t1 = rx * 0.5
    prx = np.matmul(px, rx)
    rpx = np.matmul(rx, px)
    t2 = (prx + rpx + np.matmul(prx, px)) * ((ph - sph)/ph3)
    m3 = (1 - 0.5 * ph2 - cph)/ph4
    t3 = np.matmul(px, prx) + np.matmul(rpx, px) + np.matmul(prx, px) * (-3) * (-m3)
    m4 = 0.5 * (m3 - 3*(ph - sph - ph3/6)/ph5)
    t4 = (np.matmul(np.matmul(prx, px), px) + 
          np.matmul(np.matmul(px, px), rpx)) * (-m4)

    Q = t1 + t2 + t3 + t4

    return Q

# CURLYHAT builds the 6x6 curly hat matrix from the 6x1 input
def curlyhat(vec):
    phihat = hat(vec[3:6])
    veccurlyhat = np.block([[phihat, hat(vec[0:3])],
                            [sym.zeros(3), phihat]])

    return veccurlyhat


























































