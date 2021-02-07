#! /usr/bin/python
# """ Least-squares fit example of point-spread-function (PSF) using the zernike methods """
# __author__ = [pacosalces, ]

import numpy as np
import matplotlib.pyplot as plt

from physunits import *
from lmfit import minimize, Parameters, report_fit

from zernike import zernike_expansion

#####################################################################
#                                                                   #
#                                                                   #
#                            CONSTANTS                              #
#                                                                   #
#                                                                   #
#####################################################################

# Numerical aperture
NA = 0.12

# Wavelength
wavelength = 780*nm

# Rayleigh's criterion for diffraction limited resolution
NA_diff_lim = 0.61 * wavelength / NA

# Diffraction limited spatial frequency 
kNA = np.pi / NA_diff_lim

# Square pixel size at the image plane (ccd)
pixel = 5.6*um

# Microscope magnification
magnification = 6

# Pixel size at the object plane (sample)
px, py = pixel/magnification, pixel/magnification

#####################################################################
#                                                                   #
#                                                                   #
#                            DATA INPUT                             #
#                                                                   #
#                                                                   #
#####################################################################

# Read image data
raw_image = plt.imread('examples/psf.png')

# Center of region of interest (ROI)
y0, x0 = 348, 704

# ROI sizes (pick an FFT friendly number if possible)
dy, dx = 128, 128

# ROI
pinhole = raw_image[y0 - dy:y0 + dy, x0 - dx:x0 + dx]

# Select a corner as technical noise baseline
background = raw_image[0:dy, 0:dx]

# Make subgrids centered around the ROI
Ny, Nx = raw_image.shape

# One-dimensional object space cartesian grids
x, y = px * np.linspace(-dx//2, dx//2, dx), py * np.linspace(-dy//2, dy//2, dy)

# Two-dimensional object space cartesian grids
X, Y = np.meshgrid(x, y)

# One-dimensional object space cartesian Fourier grids
kx, ky = np.fft.fftfreq(2*dy, d=px), np.fft.fftfreq(2*dx, d=py)

# Two-dimensional object space cartesian Fourier grids
kX, kY = np.meshgrid(kx, ky)

# Two-dimensional object space polar Fourier grids
kR, kT = np.sqrt(kX**2 + kY**2), np.arctan2(kY, kX)

# Subtract background level from data
signal = pinhole - background.mean()

# Coarsely estimate uncertainty assuming shot noise limited image
u_signal = np.sqrt(np.abs(signal))

#####################################################################
#                                                                   #
#                                                                   #
#                            FITTING                                #
#                                                                   #
#                                                                   #
#####################################################################

# Guess; each value below is the unnormalized contribution of each
# aberration to the total rms wavefront error 
first_guess_terms = {
                            'Z00':0.,
                      'Z1m1':0.01, 'Z11':0.01,
                'Z2m2':0.1, 'Z20':1, 'Z22':0.1,
          'Z3m3':0.01, 'Z3m1':0.1, 'Z31':0.1, 'Z33':0.01,
    'Z4m4':0.01, 'Z4m2':0.1, 'Z40':3, 'Z42':0.1, 'Z44':0.01,
}

# The zernike methods evaluate the pupil (k-space), so we need a Zernike
# real-space transformation into the point-spread-function
def pupil_to_psf(amp, kc, zernike_coefficients, offset=0.):
    """ Take 2d k-grids, a cutoff frequency, and coefficients,
    and evaluate the real space psf """
    
    # Unit aperture pupil coordinates
    ur, ut = kR / kc, kT
    
    # Pupil phase Zernike expansion
    pupil = np.exp(1j*zernike_expansion(ur, ut, zernike_coefficients))
    
    # Aperture
    pupil[ur > 1] = 0.
    
    return amp*np.abs(np.fft.fftshift(np.fft.ifftn(pupil, norm='ortho')))**2 + offset

first_guess_psf = pupil_to_psf(amp=1., kc=kNA, zernike_coefficients=first_guess_terms)

# Fitting function
def fit_psf(psf_data, max_Zernike_order=4, u_data=None, init_terms=None, verbose=True):
    """ LMfit routine to fit constrained Zernike PSF"""

    # lmift Parameters() instance
    pars = Parameters()
    
    # Unpack the pupil parameters dict. Only values initialized to 
    # different than zero vary, otherwise taken as fixed parameters.
    for key, val in init_terms.items():
        if str(max_Zernike_order + 1) in key:
            break
        else:
            if bool(val):
                pars.add(key, value=val, vary=True, min=val-2*np.abs(val), max=val+2*np.abs(val))
            else:
                pars.add(key, value=val, vary=False)

    # (Optional) add scale, kNA (aperture), and offset parameters
    pars.add('scale', value=psf_data.max(), min=1e-3, max=1e2, vary=True)
    pars.add('k_apt', value=1., min=0.8, max=1.2, vary=True)
    pars.add('offset', value=0, min=-1e1, max=1e1, vary=False)
    
    def psf_residuals(pars):
        # Separate truncated zernike coefficients from other pars
        all_pars = pars.valuesdict()
        pars_dict = dict(all_pars)
        scale = pars_dict.pop('scale')
        k_apt = pars_dict.pop('k_apt')
        offset = pars_dict.pop('offset')
        zernike_dict = pars_dict
    
        # Evaluate model psf (optional; scale and offset)
        psf_model = pupil_to_psf(amp=scale, 
                                kc=k_apt*kNA/(2*np.pi), 
                                zernike_coefficients=zernike_dict, 
                                offset=offset)
        if u_data is None:
            return (psf_model - psf_data).flatten()
        return ((psf_model - psf_data)/u_data).flatten()
    
    # Minimizer
    result = minimize(psf_residuals, pars, method='leastsq', iter_cb=None, nan_policy='omit')
    
    # Pull optimum parameters
    optimum_pars = np.array([result.params[key].value for key in result.params.keys()])

    # Try to pull covariance for estimated statistical uncertainties
    try:
        covariance_matrix = np.array(result.covar)
    except:
        covariance_matrix = 1.
    
    if verbose:    
        print(report_fit(result))
    
    return optimum_pars, covariance_matrix

# Attempt a fit
fit_parameters, fit_covariance = fit_psf(signal, 
                                        max_Zernike_order=4, 
                                        u_data=u_signal, 
                                        init_terms=first_guess_terms, 
                                        verbose=True)

best_fit_zernike_dict = {k:v for k, v in zip(first_guess_terms.keys(), fit_parameters[:-3])}

# Evaluate best fit
best_fit = pupil_to_psf(amp=fit_parameters[-3], 
                        kc=fit_parameters[-2]*kNA/(2*np.pi), 
                        zernike_coefficients=best_fit_zernike_dict, 
                        offset=fit_parameters[-1])

# Evaluate residuals
residuals = best_fit - signal

#####################################################################
#                                                                   #
#                                                                   #
#                            PLOTTING                               #
#                                                                   #
#                                                                   #
#####################################################################


plt.figure(1, figsize=(12, 4))
ax = plt.subplot(131)
ax.set_title(fR'Data PSF')
ax.imshow(signal, cmap='Blues', vmin=0, vmax=0.3)
ax = plt.subplot(132)
ax.set_title(fR'Best fit')
ax.imshow(best_fit, cmap='Reds', vmin=0, vmax=0.3)
ax = plt.subplot(133)
ax.set_title(fR'Residuals')
ax.imshow(residuals, cmap='seismic', vmin=-0.1, vmax=0.1)

plt.figure(2, figsize=(12, 4))
ax = plt.subplot(111)
ax.set_title(fR'Zernike aberration budget')
ax.bar(x=list(range(np.size(fit_parameters[:-3]))), 
       height=fit_parameters[:-3], 
       width=0.4,
       color='salmon',
       edgecolor='crimson')
ax.set_xticks(ticks=list(range(np.size(fit_parameters[:-3]))),)
ax.set_xticklabels([k for k in first_guess_terms.keys()])
ax.set_ylabel(fR'Unnormalized rms error')
ax.grid(True, c='gray', ls='--', lw=0.5)
plt.show()