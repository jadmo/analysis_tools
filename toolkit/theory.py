# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import re
import time
from scipy import optimize
from scipy.stats import norm

def pauli(N):

    pI = np.array([[1.,0.],[0.,1.]],dtype='cfloat')
    pX = np.array([[0, 1],[ 1, 0]],dtype='cfloat')
    pY = np.array([[0, -1j],[1j, 0]],dtype='cfloat')
    pZ = np.array([[1, 0],[0, -1]],dtype='cfloat')

    if N == 1:
        paulis = np.empty(shape=(4**N,2**N,2**N),dtype=np.kron(pI).dtype)
        for i in range(4**N):
            paulis[0] = pI; paulis[1] = pX; paulis[2] = pY; paulis[3] = pZ

    elif N == 2:
        paulis = np.empty(shape=(4**N,2**N,2**N),dtype=np.kron(pI,pI).dtype)
        paulis[0] = np.kron(pI,pI); paulis[1] = np.kron(pI,pX); paulis[2] = np.kron(pI,pY); paulis[3] = np.kron(pI,pZ)
        paulis[4] = np.kron(pX,pI); paulis[5] = np.kron(pX,pX); paulis[6] = np.kron(pX,pY); paulis[7] = np.kron(pX,pZ)
        paulis[8] = np.kron(pY,pI); paulis[9] = np.kron(pY,pX); paulis[10] = np.kron(pY,pY); paulis[11] = np.kron(pY,pZ)
        paulis[12] = np.kron(pZ,pI); paulis[13] = np.kron(pZ,pX); paulis[14] = np.kron(pZ,pY); paulis[15] = np.kron(pZ,pZ)

    else: ### Generalise in the future
        paulis = np.empty(shape=(4**N,2**N,2**N),dtype=np.kron(pI,pI,pI).dtype)
        for i in range(4**N):
            paulis[i] = np.kron(pI,pI,pI)

    return paulis

import numpy as np

def bare_to_dressed(freq_b, anharm, J):

    detuning = freq_b[0] - freq_b[1]

    zeta = - 2 * J**2 * (anharm[0] + anharm[1]) / (detuning + anharm[0]) / (anharm[1] - detuning)
    freq_d = []
    for i in range(2):
        freq_d.append(freq_b[i] + (-1)**i * J**2 / detuning + zeta / 2)

    return freq_d, zeta

def dressed_to_bare(freq_d, anharm, zeta):

    detuning_d = freq_d[0] - freq_d[1]

    a = anharm[0] + anharm[1] + zeta
    b = - zeta * (anharm[1] - anharm[0]) - detuning_d * (anharm[0] + anharm[1])
    c = - zeta * anharm[0] * anharm[1]

    detuning_b = (- b - np.sqrt(b**2 - 4 * a * c)) / 2 / a

    J = np.sqrt(-zeta * (detuning_b + anharm[0]) * (anharm[1] - detuning_b) / 2 / (anharm[0] + anharm[1]))
    # zeta = - 2 * J**2 * (anharm[0] + anharm[1]) / (detuning_b + anharm[0]) / (anharm[1] - detuning_b)

    freq_b = []
    for i in range(2):
        freq_b.append(freq_d[i] - (-1)**i * J**2 / detuning_b - zeta / 2)

    return freq_b, J
def bloch_vector(result):
    sum = 0
    for i in range(4):
        for j in range(4):
            sum += np.absolute(result[i,j])**2

    return sum