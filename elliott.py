# -*- coding: utf-8 -*-
"""
Fit the FACs and FAMACs absorption coefficient with Elliott Model
For Jongchul's TPC paper
Using the approach of Chris Davies in DOI: 10.1038/s41467-017-02670-2
Modified by multiplying by E**1/2 in the prefactor
Using LMFIT

@update: 9/5/2020
@author: wenger
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import lognorm
from lmfit import Model


" Functions "

#  BROADENING FUNCTIONS
        
# Gauss function
def gauss(x, sig, mu, norm=1):
	if norm == 1: # Normalised by area
		return 1/(np.sqrt(2*np.pi*sig**2))*np.exp(-((x-mu)/sig)**2/2)
	else:         # Normalised by peak
		return np.exp(-((x-mu)/sig)**2/2)

# Cosh function
def fb_cosh(x, sig, mu):
	return 1 / np.cosh((x-mu)/sig)

# Lognorm broadening function (same as Davies)
def fbroad(x, sig, mu, sig0):
    x0 = np.linspace(-2, 2, x.size)
    # https://stackoverflow.com/questions/47136595/scipy-convolve-depends-on-x?rq=1
    def g(x, x0, sig, mu, sig0):
        return signal.convolve(lognorm.pdf(x0, sig0, loc=mu), gauss(x, sig, 0, norm=0), mode='same')
    g_x = g(x, x0, sig, mu, sig0)
    argE = x[np.argmax(g_x)]
    a = g(x+argE, x0, sig, mu, sig0)/np.trapz(g_x, x)
    return a


# ELLIOTT MODEL (with Pcv)

# Excitonic part
# --------------

def alpha_x(E, p, fb=0):
    [b0, Ex, Eg, sigma_exc, mu_lognorm, sigma_cont,Ex_split, Ex_broad, Dielectric_Factor] = p

    alpha = np.zeros(len(E))


    mu = Eg-Ex
    mua = mu - 0.5*Ex_split
    mub = mu + 0.5*Ex_split
    sigma_exc2 = Ex_broad*sigma_exc

    if fb == 0:  # Gaussian broadening
        bf = gauss(E, sigma_exc, mu, norm=1) #+ np.where(E > mub, gauss(E, sigma_exc2, mub, norm=1), gauss(E, sigma_exc, mub, norm=1))
    elif fb == 1:  # lognorm broadening
        bf = fbroad(E - mua, sigma_exc, mu_lognorm, sigma_cont) / np.trapz(
            fbroad(E - mua, sigma_exc, mu_lognorm, sigma_cont), E - mua) + fbroad(E - mub, sigma_exc, mu_lognorm, sigma_cont) / np.trapz(
            fbroad(E - mub, sigma_exc, mu_lognorm, sigma_cont), E - mub)
    elif fb == 2:  # cosh broadening
        bf = fb_cosh(E, sigma_exc, mua) + np.where(E > mub, fb_cosh(E, sigma_exc2, mub,), fb_cosh(E, sigma_exc, mub))

    alpha += (4 * np.pi * Ex**(3/2)) * bf

    for n in np.arange(2,11):
        mu = Eg-Ex/n**2

        # Select broadening function
        if fb == 0: # Gaussian broadening
            bf = gauss(E, sigma_exc, mu, norm=1)
        elif fb == 1: # lognorm broadening
            bf = fbroad(E-mu, sigma_exc, mu_lognorm, sigma_cont)/np.trapz(fbroad(E-mu, sigma_exc, mu_lognorm, sigma_cont), E-mu)
        elif fb == 2: # cosh broadening
            bf = fb_cosh(E, sigma_exc, mu)

        alpha += (Dielectric_Factor**4 * np.pi * Ex**(3/2) / n**3) * bf # the Delta function is the identity in a convolution
    return alpha
 

# Continuum part
# # --------------

def alpha_c(E, p):
    [b0, Ex, Eg, sigma_exc, mu_lognorm, sigma_cont,Ex_split, Ex_broad, Dielectric_Factor] = p
    
    x = np.where(E > Eg, np.sqrt(Ex / (E - Eg), where=E>Eg), 0)
    xi = 2 * np.pi * x / (1 - np.exp(-2 * np.pi * x, where=x!=0)) # Sommerfeld factor
    free = np.where(E > Eg, np.sqrt(E-Eg, where=E>Eg), 0) # free continuum without exciton
    return np.where(E > Eg, xi * free, 0)

def alpha_c_conv(E, Erange, p, fb=0): # Broadening of the continuum part
    [b0, Ex, Eg, sigma_exc, mu_lognorm, sigma_cont,Ex_split, Ex_broad, Dielectric_Factor] = p
    
    # build kernel
    dx = E[1] - E[0] # works with evenly distributed abscissa
    kernel = np.arange(-8*sigma_exc, 8*sigma_exc, dx)
    E_pad = np.pad(E, (E.size, E.size), 'edge')
    
    # Select broadening function
    if fb == 0: # Gaussian broadening
        res = np.convolve(alpha_c(E_pad, p), gauss(kernel, sigma_cont, 0, 1), mode='same')/np.sum(gauss(kernel, sigma_cont, 0, norm=1))
    elif fb == 1: # lognorm broadening
        res = signal.convolve(alpha_c(E_pad, p), fbroad(kernel, sigma_cont, mu_lognorm, sigma_cont), mode='same')/np.sum(fbroad(kernel, sigma_cont, mu_lognorm, sigma_cont))
    elif fb == 2: # cosh broadening
        res = signal.convolve(alpha_c(E_pad, p), fb_cosh(kernel, sigma_cont, 0), mode='same')/np.sum(fb_cosh(kernel, sigma_cont, 0))
    
    res = res[np.max(np.where(E_pad <= Erange[0])): np.min(np.where(E_pad >= Erange[1]))+1]
    # division by the sum is required to keep the scale
    return res

def alpha_sum(E, Erange, b0, Ex, Eg, sigma_exc, mu_lognorm, sigma_cont,Ex_split,Ex_broad, Dielectric_Factor, fb=0):
    p = [b0, Ex, Eg, sigma_exc, mu_lognorm, sigma_cont,Ex_split, Ex_broad, Dielectric_Factor]

    Esub = E[np.max(np.where(E <= Erange[0])): np.min(np.where(E >= Erange[1]))+1]

    return b0 * Ex**0.5 / Esub * ((alpha_x(Esub, p, fb) + alpha_c_conv(Esub, Erange, p, fb)))


def dofit(eV, abscoef, guess, eVrange, fb=0):
    # need to have evenly spaced data for the convolution
    eV_interp = np.linspace(eV[0], eV[-1], 600)
    data_interp = np.interp(eV_interp, eV, absCoef)
    print(eV_interp)
    # Fit range
    #eVsub = eV_interp[np.max(np.where(eV_interp < eVrange[0])): np.min(np.where(eV_interp > eVrange[1]))+1]
    data = data_interp[np.max(np.where(eV_interp < eVrange[0])): np.min(np.where(eV_interp > eVrange[1]))+1]
    print(fb)
    # Using LMfit
    myMod = Model(alpha_sum, independent_vars=['E', 'Erange'])
    print(myMod.param_names)
    myMod.set_param_hint('b0',value=guess[0], min = 0.01, max = 100, vary = True)
    myMod.set_param_hint('Ex', value=guess[1], min=0., max=0.08, vary=False)
    myMod.set_param_hint('Eg', value=guess[2], min=eVrange[0], max=eVrange[1], vary=False)
    myMod.set_param_hint('sigma_exc', value=guess[3], min=0.0001, max=0.1, vary=True)#, min=0)
    myMod.set_param_hint('sigma_cont', value=guess[4], min=0.001, max=0.2,vary=True)
    myMod.set_param_hint('mu_lognorm', value=guess[5], vary=True)
    myMod.set_param_hint('Ex_split', value=guess[6],min = 0, max = 80, vary=False)
    myMod.set_param_hint('Ex_broad', value=guess[7], min = 1, max = 10, vary=False)
    myMod.set_param_hint('Dielectric_Factor', value=guess[8],min = 1, max = 10, vary=False)
    myMod.set_param_hint('fb', value=int(fb), vary=False)

    result = myMod.fit(data, E=eV_interp, Erange=eVrange)
    print(result.fit_report())

    # Plotting results
    # ----------------
    best_params = result.best_values
    bp = [best_params['b0'], best_params['Ex'], best_params['Eg'], best_params['sigma_exc'], best_params['mu_lognorm'], best_params['sigma_cont'],best_params['Ex_split'],best_params['Ex_broad'],best_params['Dielectric_Factor']]

    fit_exc = bp[0] * bp[1]**0.5 / eV_interp * alpha_x(eV_interp, bp, fb)
    fit_cont = bp[0]  * bp[1]**0.5 / eV_interp * alpha_c_conv(eV_interp, [eV_interp[0], eV_interp[-1]], bp, fb)
    fit_y = bp[0] * bp[1]**0.5 / eV_interp * ( alpha_x(eV_interp, bp, fb) + alpha_c_conv(eV_interp, [eV_interp[0], eV_interp[-1]], bp, fb))

    plt.plot(eV, absCoef, label='data', ls=':')

    plt.plot(eV_interp, fit_exc, label='exc')
    plt.plot(eV_interp, fit_cont, label='cont')
    plt.plot(eV_interp, fit_y, ls='-', label='fit full')

    plt.xlim(fit_range[0]-0.1, fit_range[1]+0.1)
    plt.ylim(bottom=0)
    plt.xlabel('eV'); plt.ylabel('Abs. coef [1e4 * cm-1]')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,3))

    plt.legend()
    plt.show()

    np.savetxt(f'{File_name[:-4]}_fits.dat', np.c_[eV_interp, data_interp*scale, fit_y*scale, fit_exc*scale, fit_cont*scale])
    save_report = open(f'{File_name[:-4]}_fit_results.txt', "w")
    save_report.write(result.fit_report())
    save_report.close()


" Data processing "
# load and convert data
#loaded = np.loadtxt('Mat1.txt', skiprows=0, delimiter='\t', unpack=False)

File_name = r"nk_0br_1.txt"
Raw_File = pd.read_csv(File_name, sep="\t")
eV = Raw_File["eV"].values
absCoef = 2*np.pi*Raw_File["k-Bottom"].values/(1240*eV*1e-7)

scale =1e4
absCoef /= scale # rescale

# Guess parameters
b0 = 10
Ex = 21 # meV
Eg = 1.58# eV
sigma_exc = 37 # meV
sigma_cont = 20# meV
mu_lognorm = -0.2 # eV

fit_range = [1.45, 1.65]

#########For 2D Materials %%%%%%
Ex_split = 0#meV  (only used, if >0
Ex_broad =1 #Factor that simluates an asymmetric broadening of the 1s excitonic peak
Dielectric_Factor = 1#enhances excitonic peak (set to 1 for 3D)
#############
print(fit_range)
guess = [b0 / scale, Ex*1e-3, Eg, sigma_exc*1e-3, sigma_cont*1e-3, mu_lognorm,Ex_split*1e-3,Ex_broad,Dielectric_Factor]
dofit(eV, absCoef, guess, fit_range, fb=0)