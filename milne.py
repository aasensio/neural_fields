# Author: cdiazbas@iac.es
# Code: Zeeman pattern

# ADDED: velocidad LOS
# ADDED: signo a eta_blue y eta_red
# ADDED: normalization factor in p,b,r eta + rho profiles

import torch
import numpy as np
from utils import *
import matplotlib.pyplot as pl

# ============================================
# FUNCTIONS
# ============================================


def grad(x):
    return x * 180. / np.pi


def rad(x):
    return x * np.pi / 180.


def stokesSyn(param, x, B1, B2, B3, vlos, eta0, a, ddop, S_0, S_1, cartesian=False):

    # PARAMETROS:
    # param			Parametros de la linea
    # x 			Array Longitud de onda
    # 0 = B			Campo magnetico
    # 1 = gamma		Inclinacion
    # 2 = xi		Angulo azimutal
    # 3 = vlos      Velocidad en la linea de vision [km/s]
    # 4 = eta0		Cociente de abs linea-continuo
    # 5 = a 		Parametro de amortiguamiento
    # 6 = ddop		Anchura Doppler
    # 7 = S_0		Ordenada de la funcion fuente
    # 8 = S_1		Gradiente de la funcion fuente

    # Pi square factor:
    sqrtpi = 1. / np.sqrt(np.pi)

    if cartesian:
        B = torch.sqrt(B1**2. + B2**2. + B3**2.)
        gamma = torch.acos(B3 / B)
        xi = torch.atan2(B2, B1)
    else:
        B = B1
        gamma = B2
        xi = B3

    # Magnetic field direction:
    singamma = torch.sin(gamma[:, None])
    cosgamma = torch.cos(gamma[:, None])
    sin2xi = torch.sin(2 * xi[:, None])
    cos2xi = torch.cos(2 * xi[:, None])

    # Calculating the Zeeman Pattern
    [[dpi, dsr, dsb], [spi, ssr, ssb], [ju, lu, su, jl,ll, sl, elem, l0, gu, gl, gef]] = param
        
    # Lorentz factor
    lB = 4.67E-13 * l0**2. * B[:, None]

    
    # VLOS: km2A
    cc = 3.0E+5        # veloc luz [km/s]
    vlosA = l0 * vlos[:, None] / cc

    # ============================================
    # Perfiles de absorcion y dispersion

    # COMPONENTE PI
    # --------------------------------------------
    eta_p = 0.
    rho_p = 0.
    for i in range(0, len(spi)):
        xx = (x[None, :] - lB * dpi[i] - vlosA) / ddop[:, None]        
        [H, F] = fvoigt(a[:, None], xx)
        eta_p = eta_p + H * spi[i] * sqrtpi / ddop[:, None]
        rho_p = rho_p + 2. * F * spi[i] * sqrtpi / ddop[:, None]

    # COMPONENTE SIGMA BLUE
    # --------------------------------------------
    eta_b = 0.
    rho_b = 0.
    for i in range(0, len(ssb)):
        xx = (x[None, :] - lB * dsb[i] - vlosA) / ddop[:, None]
        [H, F] = fvoigt(a[:, None], xx)
        eta_b = eta_b + H * ssb[i] * sqrtpi / ddop[:, None]
        rho_b = rho_b + 2. * F * ssb[i] * sqrtpi / ddop[:, None]

    # COMPONENTE SIGMA RED
    # --------------------------------------------
    eta_r = 0.
    rho_r = 0.
    for i in range(0, len(ssr)):
        xx = (x[None, :] - lB * dsr[i] - vlosA) / ddop[:, None]
        [H, F] = fvoigt(a[:, None], xx)
        eta_r = eta_r + H * ssr[i] * sqrtpi / ddop[:, None]
        rho_r = rho_r + 2. * F * ssr[i] * sqrtpi / ddop[:, None]

    # ============================================
    # Elementos la matriz de propagacion

    # 1.- Elemento de absorcion
    eta_I = 1.0 + 0.5 * eta0[:, None] * \
        (eta_p * (singamma**2.) + 0.5 * (eta_b + eta_r) * (1. + cosgamma**2.))
    # 2.- Elementos de dicroismo (pol dif en dif direcc)
    eta_Q = 0.5 * eta0[:, None] * (eta_p - 0.5 * (eta_b + eta_r)
                          ) * (singamma**2.) * cos2xi
    eta_U = 0.5 * eta0[:, None] * (eta_p - 0.5 * (eta_b + eta_r)
                          ) * (singamma**2.) * sin2xi
    eta_V = 0.5 * eta0[:, None] * (eta_r - eta_b) * cosgamma
    # 3.- Elementos de dispersion
    rho_Q = 0.5 * eta0[:, None] * (rho_p - 0.5 * (rho_b + rho_r)
                          ) * (singamma**2.) * cos2xi
    rho_U = 0.5 * eta0[:, None] * (rho_p - 0.5 * (rho_b + rho_r)
                          ) * (singamma**2.) * sin2xi
    rho_V = 0.5 * eta0[:, None] * (rho_r - rho_b) * cosgamma

    # ============================================
    # Perfiles de Stokes normalizados al continuo// Sc = S1/S0

    # ScDown = 1.+Sc
    # Det=eta_I**2.*(eta_I**2.-eta_Q**2.-eta_U**2.-eta_V**2.+rho_Q**2.+rho_U**2.+rho_V**2.)-(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V)**2.
    # IDet = 1./Det
    # I=(1.+IDet*eta_I*(eta_I**2.+rho_Q**2.+rho_U**2.+rho_V**2.)*Sc)/ScDown
    # Q=-IDet*(eta_I**2.*eta_Q+eta_I*(eta_V*rho_U-eta_U*rho_V)+rho_Q*(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V))*Sc/ScDown
    # U=-IDet*(eta_I**2.*eta_U+eta_I*(eta_Q*rho_V-eta_V*rho_Q)+rho_U*(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V))*Sc/ScDown
    # V=-IDet*(eta_I**2.*eta_V+eta_I*(eta_U*rho_Q-eta_Q*rho_U)+rho_V*(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V))*Sc/ScDown

    Det = eta_I**2. * (eta_I**2. - eta_Q**2. - eta_U**2. - eta_V**2. + rho_Q**2. +
                       rho_U**2. + rho_V**2.) - (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V)**2.
    IDet = 1. / Det
    I = S_0[:, None] + IDet * eta_I * (eta_I**2. + rho_Q**2. + rho_U**2. + rho_V**2.) * S_1[:, None]
    Q = -IDet * (eta_I**2. * eta_Q + eta_I * (eta_V * rho_U - eta_U * rho_V) + rho_Q * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V)) * S_1[:, None]
    U = -IDet * (eta_I**2. * eta_U + eta_I * (eta_Q * rho_V - eta_V * rho_Q) + rho_U * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V)) * S_1[:, None]
    V = -IDet * (eta_I**2. * eta_V + eta_I * (eta_U * rho_Q - eta_Q * rho_U) + rho_V * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V)) * S_1[:, None]

    out = torch.cat([I[:, None, :], Q[:, None, :], U[:, None, :], V[:, None, :]], dim=1)
    
    return out
    

if __name__ == "__main__":

    # PARAMETROS:
    nlinea = 3						# Numero linea en fichero
    # class paramlib(object):
    param = paramLine(nlinea)



    x = torch.arange(-2.8, 2.8, 20e-3)				# Array Longitud de onda
    B = torch.tensor([992., 992.], requires_grad=True)						# Campo magnetico
    gamma = torch.tensor([rad(134.), rad(134.)])  					# Inclinacion
    xi = torch.tensor([rad(145.), rad(145.)]) 						# Angulo azimutal
    vlos = torch.tensor([0.0, 4.0])                     				 # velocidad km/s
    eta0 = torch.tensor([73., 73.]) 						# Cociente de abs linea-continuo
    a = torch.tensor([0.2, 0.1]) 						# Parametro de amortiguamiento
    ddop = torch.tensor([0.02, 0.02]) 						# Anchura Doppler
    # Sc = 4.0						# Cociente Gradiente y Ordenada de la funcion fuente
    S_0 = torch.tensor([0.5, 0.5])							# Ordenada de la funcion fuente
    S_1 = torch.tensor([0.5, 0.5])							# Gradiente de la funcion fuente


    stokes = stokesSyn(param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
    
    for i in range(4):        
        pl.subplot(2, 2, i + 1)
        if i == 0:
            pl.ylim(0, 1.1)
        for j in range(2):
            pl.plot(x, stokes[j, i, :].detach().numpy())        
        # if i != 0: plt.ylim(-0.4,0.4)
    # plt.tight_layout()
    pl.show()
    # plt.savefig('stokes.pdf')

    # np.save('stokes.npy',stokes)