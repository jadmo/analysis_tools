# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
# from skopt import dummy_minimize
from qutip import *
import re
import time
from scipy import optimize
from scipy.optimize import minimize, least_squares, curve_fit
from scipy.signal import get_window
from scipy.stats import norm


def error_bar(popt, pcov):
    ub = popt + np.sqrt(np.diag(pcov))
    lb = popt - np.sqrt(np.diag(pcov))

    return ub, lb

def lin_func(x, a, b):
    return a * x + b

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def exp_sin_func(x, amp, decay, omega, phi, offset):
    return amp*np.exp(-x/decay)*np.sin(2.*np.pi*omega*x+phi)+offset

def lsq_fit(func, x, y):
    try:
        popt, pcov = curve_fit(func, x, y, maxfev=40000)

    except:
        popt, pcov = None, None

    return {'func':func, 'popt':popt, 'pcov':pcov}

def Fit_3D_Rabi(data, dt, freq_est=None, use_freq_bound=True):

    #find plane of system trajectory
    def cost_func(x):
        nx = np.cos(x[0])*np.sin(x[1])
        ny = np.sin(x[0])*np.sin(x[1])
        nz = np.cos(x[1])
        return np.var((data[0]*nx+data[1]*ny+data[2]*nz))

    result = dummy_minimize(cost_func, ((-np.pi,np.pi),(0.,np.pi)), n_calls=10000)

    phi = result.x[0]
    theta = result.x[1]

    # axis of rotation
    nx = np.cos(result.x[0])*np.sin(result.x[1])
    ny = np.sin(result.x[0])*np.sin(result.x[1])
    nz = np.cos(result.x[1])

    #get direction of axis correct
    tx,ty,tz = 0.,0.,0.
    for i in range(len(data[0])-1):
        ax = data[0][i]
        ay = data[1][i]
        az = data[2][i]

        bx = data[0][i+1]
        by = data[1][i+1]
        bz = data[2][i+1]

        vx,vy,vz = ay*bz-az*by,az*bx-ax*bz,ax*by-ay*bx

        tx+=vx
        ty+=vy
        tz+=vz

    norm = np.sqrt(tx*tx+ty*ty+tz*tz)
    tx = tx/norm
    ty = ty/norm
    tz = tz/norm

    if tx*nx+ty*ny+tz*nz < 0:
        nx,ny,nz = -nx,-ny,-nz

    # vector in plane of rotation
    ox,oy,oz = ny,-nx,0.
    norm = np.sqrt(ox*ox+oy*oy+oz*oz)
    ox = ox/norm
    oy = oy/norm
    oz = oz/norm
    #another vector in plane of rotation
    o2x,o2y,o2z = ny*oz-nz*oy,nz*ox-nx*oz,nx*oy-ny*ox
    norm = np.sqrt(o2x*o2x+o2y*o2y+o2z*o2z)
    o2x = o2x/norm
    o2y = o2y/norm
    o2z = o2z/norm

    data2d = np.empty(shape=data[0].shape, dtype='cfloat')
    data2d.real = ox*data[0]+oy*data[1]+oz*data[2]
    data2d.imag = o2x*data[0]+o2y*data[1]+o2z*data[2]
    datan = nx*data[0]+ny*data[1]+nz*data[2]

    t = np.linspace(0.,dt*(len(data2d)-1), len(data2d))

    fft = np.abs(np.fft.fft(data2d))
    f = np.fft.fftfreq(len(data2d),dt)

    imax = np.argmax(fft[1:])+1
    fmax = f[imax]
    omega = fmax

    df = f[1]-f[0]
    min_omega = fmax - df
    max_omega = fmax + df

    if freq_est is not None:
        omega = freq_est

    icos = data2d.real*np.cos(2.0*np.pi*omega*t)
    isin = data2d.real*np.sin(2.0*np.pi*omega*t)
    qcos = data2d.real*np.cos(2.0*np.pi*omega*t)
    qsin = data2d.real*np.sin(2.0*np.pi*omega*t)
    iphi = np.arctan2(np.mean(icos),np.mean(isin))
    qphi = np.arctan2(np.mean(qcos),np.mean(qsin))
    ioffset = np.mean(data2d.real)
    qoffset = np.mean(data2d.imag)
    iamp = 2**0.5*np.mean((data2d.real-ioffset)**2)**0.5
    qamp = 2**0.5*np.mean((data2d.imag-qoffset)**2)**0.5

    def leastsq(x):
        omega, iamp, iphi, ioffset, qamp, qphi, qoffset=x

        ifit = iamp*np.sin(2.0*np.pi*omega*t+iphi)
        qfit = qamp*np.sin(2.0*np.pi*omega*t+qphi)

        return np.sum((ifit-data2d.real)**2) + np.sum((qfit-data2d.imag)**2)


    args = dict()
    if use_freq_bound:
        args['bounds'] = ((min_omega,max_omega),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None))

    result=minimize(leastsq, np.array([omega,iamp,iphi,ioffset,qamp,qphi,qoffset]), **args)

    omega,iamp,iphi,ioffset,qamp,qphi,qoffset=result.x

    if omega < 0.:
        iamp = -iamp
        qamp = -qamp
        omega = -omega

    normalaxis = np.array((nx,ny,nz))
    iaxis = np.array((ox, oy, oz))
    qaxis = np.array((o2x,o2y,o2z))
    centre = np.mean(datan)

    return {'CompFrequency':omega*normalaxis, 'Frequency':omega, 'RotationAxis':normalaxis, 'IAxis':iaxis, 'QAxis':qaxis, 'IAmp':iamp,'IPhase':iphi,'IOffset':ioffset, 'QAmp':qamp, 'QPhase':qphi, 'QOffset':qoffset, 'Centre':centre, 'Time':t}

def Fit_1D_Freq_exp(z, dt, use_freq_bound=True):

    z = np.real(z)

    rfft = np.abs(np.fft.rfft(z))
    f = np.fft.rfftfreq(len(z),dt)

    imax = np.argmax(rfft[1:])+1
    # print('shape of z', np.shape(z))
    # print('rfft', np.shape(rfft))
    # print('rfft[1:]', np.shape(rfft[1:]))
    # print('len(z)', len(z))
    # print('f', np.shape(f))
    # print('imax', imax)

    fmax = f[imax]

    t = np.linspace(0.,dt*(len(z)-1), len(z))

    omega = fmax

    df = f[1]-f[0]
    min_omega = fmax - df
    max_omega = fmax + df

    cosz = z*np.cos(2.0*np.pi*omega*t)
    sinz = z*np.sin(2.0*np.pi*omega*t)
    phi = np.arctan2(np.mean(cosz),np.mean(sinz))
    offset = np.mean(z)
    amp = 2**0.5*np.mean((z-offset)**2)**0.5
    decay = t[-1]*10

    def lsq_exp_sin(x,t,z):
        omega,amp,phi,offset,decay=x
        fit = amp*np.exp(-t/decay)*np.sin(2.*np.pi*omega*t+phi)+offset
        return np.sum((fit-z)**2)*1e3

    def lsq_exp(x,t,z):
        amp,offset,decay=x
        fit = amp*np.exp(-t/decay)+offset
        return np.sum((fit-z)**2)*1e3

    args = dict()
    if use_freq_bound:
        args['bounds'] = ((min_omega,max_omega),(-1.2,1.2),(None,None),(-1.2,1.2),(0,None))

    if True:
        result=minimize(lsq_exp_sin, np.array([omega,amp,phi,offset,decay]),args=(t,z),**args)
    else:
        result=minimize(lsq_exp, np.array([amp,offset,decay]),args=(t,z))

    omega,amp,phi,offset,decay=result.x

    popt = amp, decay, omega, phi, offset

    return {'Frequency':omega,'Amplitude':amp,'Phase':phi, 'Offset':offset, 'Decay':decay, 'func':exp_sin_func, 'popt':popt}

def Fit_1D_Freq_Gaussian_Estimation(z, dt, gaussian_window_std = 0.2):

    amp = Fit_1D_Freq_exp(z, dt)['Amplitude']
    phi = Fit_1D_Freq_exp(z, dt)['Phase']
    offset = Fit_1D_Freq_exp(z, dt)['Offset']
    decay = Fit_1D_Freq_exp(z, dt)['Decay']

    #  fourier transform with gaussian window applied to data
    gaussian_std = round(gaussian_window_std*len(z))
    windowed_signal = np.abs(np.fft.rfft(z*get_window(('gaussian', gaussian_std), len(z))))
    f = np.fft.rfftfreq(len(z),dt)

    df = f[1]-f[0]

    # select peak and neighbour peaks

    imax = np.argmax(windowed_signal[2:])+2 # DC signal removed
    fmax = f[imax]
    if imax > len(windowed_signal)-2:
        imax = imax - 1

    signal_max = windowed_signal[imax]
    signal_left_lobe =  windowed_signal[imax-1]
    signal_right_lobe = windowed_signal[imax+1]

    # apply gaussian estimation function to more accurately find frequency
    def gaussian_estimation(peak_sig, left_lobe_sig, right_lobe_sig, peak_index, df, f_min):
        # https://mgasior.web.cern.ch/pap/FFT_resol_note.pdf
        p_sig = peak_sig
        l_sig = left_lobe_sig
        r_sig = right_lobe_sig

        delta = np.log(r_sig/l_sig)/(2*np.log(p_sig**2/(l_sig*r_sig)))

        estimated_frequency = f_min+df*(peak_index+delta)

        return(estimated_frequency)

    omega = gaussian_estimation(signal_max, signal_left_lobe, signal_right_lobe, imax, df, 0)

    popt = amp, decay, omega, phi, offset

    return {'Frequency':omega,'Amplitude':amp,'Phase':phi, 'Offset':offset, 'Decay':decay, 'func':exp_sin_func, 'popt':popt}
