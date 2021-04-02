import numpy as np
from matplotlib import pylab as plt
import scipy as s

def exp(x, amp, rate, offset):
    return amp * np.exp(rate * x) + offset

def exp_sin(x, amp, rate, freq, phase, offset):
    return amp * np.exp(- x * rate) * np.sin(2 * np.pi * freq * x + phase) + offset

def sin(x, amp, freq, phase, offset):
    return amp * np.sin(2 * np.pi * freq * x + phase) + offset

def linear(x, slope, offset):
    return slope * x + offset
