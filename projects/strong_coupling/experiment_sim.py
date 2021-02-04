# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import re
import time
from scipy import optimize
from scipy.stats import norm
from sim_functions import *
from analysis_tools import *

# test

def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])

class time_evolution_2q(Experiment):

  def __init__(self, freqs, anharms, T1s, T2s, J, files, Hamiltonian=None):

    if Hamiltonian is 'dispersive' or Hamiltonian is 'dipole':
      if anharms[0] == 0.0 and anharms[1] == 0.0:
        level = 2

      else:
        level = 3

      super(time_evolution_2q, self).__init__(levels=level, total_time=1.0, time_resolution=0.0005)

      self.add_qubit(freq=freqs[0], anharmonicity=anharms[0], T1=T1s[0], T2=T2s[0])
      self.add_qubit(freq=freqs[1], anharmonicity=anharms[1], T1=T1s[1], T2=T2s[1])

      if Hamiltonian is 'dispersive':
        self.set_connectivity(0,1,J,"ZZ")

      elif Hamiltonian is 'dipole':
        self.set_connectivity(0,1,J,"XX")

      self.set_hamiltonian()
      self.set_rot_frame(index=0,freq=freqs[0])
      self.set_rot_frame(index=1,freq=freqs[1])

    else:
      super(time_evolution_2q, self).__init__(levels=2, total_time=1.0, time_resolution=0.0005)
      for term in Hamiltonian:
        self.add_terms(omega=term[0], pauli_2q=term[1]) # [[4, 'ZZ'], [30, 'XX']]
      for i, t1 in enumerate(T1s):
        self.qubits['T1'].append(T1s[i])
        self.qubits['T2'].append(T2s[i])
    self.J = J
    self.freqs = freqs
    self.anharms = anharms
    self.files = files

    self.initial = [0, 0]
    self.num_pulses = [0, 0]

    self.meas_time = []
    self.qobj = Qobj()

  def update_duration(self, widths, truncs):

    self.widths = widths
    self.truncs = truncs
    self.durations = []

    for i in range(len(self.widths)):
      self.durations.append(self.widths[i]*self.truncs[i])

  def normalise_gaussian(self, trunc):

    x = np.arange(-trunc, trunc, 0.001) # range of x in spec
    x_all = np.arange(-10, 10, 0.001) # entire range of x, both in and out of spec

    return np.sum(norm.pdf(x,0,1))/np.sum(norm.pdf(x_all,0,1))

  def total_time(self, durations, delays):

    self.duration = 0.0005
    for n, t in enumerate(durations):
      self.duration += self.num_pulses[n] * t

    return self.duration + np.sum(delays)

  def run_test(self, delays, amps, phis, drags, widths, truncs):
    # print('changed')
    self.update_duration(widths, truncs)
    self.total_duration = self.total_time(self.durations, delays)
    self.tlist = np.arange(0., self.total_duration, 0.0005)
    self.set_clock(total_time=self.total_duration, time_resolution=0.0005)
    self.clear_pulses()
    self.meas_time = self.sequence(delays, amps, phis, drags, widths, truncs)
    self.initialization(initial=self.initial)
    self.q_obj = self.run()
    self.result = self.tomography_2q_data(self.q_obj.states, [0,1], self.meas_time)

  def plot(self, full_trace=False, sequence=False, pauli=False, tomography=False, bloch=False, corr=False, error=False, save=None):

    self.save = save

    if full_trace:
      time = self.tlist
    else:
      time = self.meas_time

    if sequence:
      self.show_pulse_sequence()

    if pauli:
      self.show_pauli(self.q_obj.states, ['ZI','IZ','XX','YY','ZZ'], time) # [0::10])

    if tomography:
      self.tomography_2q(self.q_obj.states, [0,1], time)

    if bloch:
      self.show_bloch(self.q_obj.states, 0, time) # [0::10])
      self.show_bloch(self.q_obj.states, 1, time) # [0::10])

    if corr:
      self.show_corr(self.q_obj.states, [0,1], time, show_qst=True)

    if error:
      pass

class time_evolution_2q_lab(Experiment):

  def __init__(self, freq_d, detunings, anharms, T1s, T2s, zeta, files, Hamiltonian=None, quick=False):

    if Hamiltonian is 'dispersive':

      if anharms[0] == 0.0 and anharms[1] == 0.0:
        level = 2

      else:
        level = 3

      super(time_evolution_2q_lab, self).__init__(levels=level, total_time=1.0, time_resolution=0.0005)

      if quick is True:
        rot = freq_d[0]
      else:
        rot = 0

      self.freqs = [freq_d[0] - rot, freq_d[1] - rot]

      self.add_qubit(freq=freq_d[0] - rot, anharmonicity=anharms[0], T1=T1s[0], T2=T2s[0])
      self.add_qubit(freq=freq_d[1] - rot, anharmonicity=anharms[1], T1=T1s[1], T2=T2s[1])

      self.set_connectivity(0,1,zeta,"ZZ")

      self.set_hamiltonian()
      self.set_rot_frame(index=0,freq=freq_d[0] - rot + detunings[0])
      self.set_rot_frame(index=1,freq=freq_d[1] - rot + detunings[1])

    elif Hamiltonian is 'dipole':

      if anharms[0] == 0.0 and anharms[1] == 0.0:
        level = 2
        freq_b, J = dressed_to_bare(freq_d, anharms, zeta) # TODO: make dressed_to_bare function for 2-level system

      else:
        level = 3
        freq_b, J = dressed_to_bare(freq_d, anharms, zeta)
        # print('J', J)
        # print('freq_b', freq_b)
        # print('freq_d', freq_d)

      super(time_evolution_2q_lab, self).__init__(levels=level, total_time=1.0, time_resolution=0.0005)

      # self.add_qubit(freq=freq_b[0], anharmonicity=anharms[0], T1=T1s[0], T2=T2s[0])
      # self.add_qubit(freq=freq_b[1], anharmonicity=anharms[1], T1=T1s[1], T2=T2s[1])

      if quick is True:
        rot = freq_b[0]
      else:
        rot = 0

      self.freqs = [freq_b[0] - rot, freq_b[1] - rot]

      self.add_qubit(freq=freq_b[0] - rot, anharmonicity=anharms[0], T1=T1s[0], T2=T2s[0])
      self.add_qubit(freq=freq_b[1] - rot, anharmonicity=anharms[1], T1=T1s[1], T2=T2s[1])

      if quick is True:
        self.set_connectivity(0,1,J,"XX_RWA")
      else:
        self.set_connectivity(0,1,J,"XX")

      self.J = J

      self.set_hamiltonian()
      # print('qubit 1 shift', freq_d[0]-freq_b[0])
      # print('qubit 2 shift', freq_d[1]-freq_b[1])
      # self.set_rot_frame(index=0,freq=freq_d[0]+detuning)
      # self.set_rot_frame(index=1,freq=freq_d[1]+detuning)
      self.set_rot_frame(index=0,freq=freq_d[0] - rot + detunings[0])
      self.set_rot_frame(index=1,freq=freq_d[1] - rot + detunings[1])

    else:
      super(time_evolution_2q_lab, self).__init__(levels=2, total_time=1.0, time_resolution=0.0005)
      for term in Hamiltonian:
        self.add_terms(omega=term[0], pauli_2q=term[1]) # [[4, 'ZZ'], [30, 'XX']]
      for i, t1 in enumerate(T1s):
        self.qubits['T1'].append(T1s[i])
        self.qubits['T2'].append(T2s[i])


    self.zeta = zeta
    # self.freqs = freq_d
    self.anharms = anharms
    self.files = files

    self.initial = [0, 0]
    self.num_pulses = [0, 0]

    self.meas_time = []
    self.qobj = Qobj()

  def update_duration(self, widths, truncs):

    self.widths = widths
    self.truncs = truncs
    self.durations = []

    for i in range(len(self.widths)):
      self.durations.append(self.widths[i]*self.truncs[i])

  def normalise_gaussian(self, trunc):

    x = np.arange(-trunc, trunc, 0.001) # range of x in spec
    x_all = np.arange(-10, 10, 0.001) # entire range of x, both in and out of spec

    return np.sum(norm.pdf(x,0,1))/np.sum(norm.pdf(x_all,0,1))

  def total_time(self, durations, delays):

    self.duration = 0.0005
    for n, t in enumerate(durations):
      self.duration += self.num_pulses[n] * t

    return self.duration + np.sum(delays)

  def run_test(self, delays, amps, phis, drags, widths, truncs):
    # print('changed')
    self.update_duration(widths, truncs)
    self.total_duration = self.total_time(self.durations, delays)
    self.tlist = np.arange(0., self.total_duration, 0.0005)
    self.set_clock(total_time=self.total_duration, time_resolution=0.0005)
    self.clear_pulses()
    self.meas_time = self.sequence(delays, amps, phis, drags, widths, truncs)
    self.initialization(initial=self.initial)
    self.q_obj = self.run()
    self.result = self.tomography_2q_data(self.q_obj.states, [0,1], self.meas_time)

  def plot(self, full_trace=False, sequence=False, pauli=False, tomography=False, bloch=False, corr=False, error=False, save=None):

    self.save = save

    if full_trace:
      time = self.tlist
    else:
      time = self.meas_time

    if sequence:
      self.show_pulse_sequence()

    if pauli:
      self.show_pauli(self.q_obj.states, ['ZI','IZ','XX','YY','ZZ'], time) # [0::10])

    if tomography:
      self.tomography_2q(self.q_obj.states, [0,1], time)

    if bloch:
      self.show_bloch(self.q_obj.states, 0, time) # [0::10])
      self.show_bloch(self.q_obj.states, 1, time) # [0::10])

    if corr:
      self.show_corr(self.q_obj.states, [0,1], time, show_qst=True)

    if error:
      pass

class vqe(time_evolution_2q):

  def __init__(self, freqs, detunings, anharms, T1s, T2s, J, files, echo, initial=None, Hamiltonian=None, quick=False):
    super(vqe, self).__init__(freqs, detunings, anharms, T1s, T2s, J, files, Hamiltonian, quick)
    # super(vqe, self).__init__(freqs, anharms, T1s, T2s, J, files, Hamiltonian)

    self.initial = initial

    if echo == 'single':
      self.num_pulses = [1, 2]

    elif echo == 'double':
      self.num_pulses = [3, 4]

  def sequence(self, delays, amps, phis, drags, widths, truncs):

    self.update_duration(widths, truncs)
    width = widths[0]; trunc = truncs[0]
    norm = self.normalise_gaussian(trunc)

    amps = [1/width/8 /norm * amps[0], 1/width/8/norm * amps[1], 1/width/4/norm * amps[2], 1/width/4/norm * amps[2]]
    drags = [self.anharms[0] * drags[0], self.anharms[1] * drags[1], self.anharms[1] * drags[2], self.anharms[0] * drags[2]]
    # print(amps[2])


    if self.num_pulses[0] == 1:

      if self.initial[0] is not '0' or self.initial[1] is not '0':
        t_2pulses = 0

      else:
        t1 = trunc * width / 2
        self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss_DRAG',amp=amps[0],centre=t1,width=width,trunc=trunc,alpha=drags[0])

        t2 = trunc * width * 3 / 2
        self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss_DRAG',amp=amps[1],centre=t2,width=width,trunc=trunc,alpha=drags[1])

        t_2pulses = trunc * width * 2

      t3 = t_2pulses + trunc * width / 2 + delays[0]
      self.add_pulse(index=1,f=self.freqs[1],phi=phis[2],shape='gauss_DRAG',amp=amps[2],centre=t3,width=width,trunc=trunc,alpha=self.anharms[1])

      # t4 = trunc * width * 7 / 2 + delays[0] + delays[1]
      # self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss_DRAG',amp=1/width/8/norm,centre=t4,width=width,trunc=trunc,alpha=self.anharms[0])
      # t5 = trunc * width * 9 / 2 + delays[0] + delays[1]
      # self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss_DRAG',amp=1/width/8/norm,centre=t5,width=width,trunc=trunc,alpha=self.anharms[1])

    if self.num_pulses[0] == 3:
      pass
    final_time = self.total_duration - 0.0005# trunc * width * np.sum(self.num_pulses) + np.sum(delays)

    meas_time = [final_time]

    return meas_time

class ham_tomo(time_evolution_2q_lab):

  def __init__(self, freqs, detunings, anharms, T1s, T2s, J, files, echo, initial=None, Hamiltonian=None, quick=False):
    super(ham_tomo, self).__init__(freqs, detunings, anharms, T1s, T2s, J, files, Hamiltonian, quick)

    self.initial = initial

    if self.initial[0] is not '0' or self.initial[1] is not '0':

      if echo == 'single':
        self.echo = 'single'
        self.num_pulses = [0, 1]

      elif echo == 'double':
        self.echo = 'double'
        self.num_pulses = [1, 1]

      else:
        self.echo = 'no_echo'
        self.num_pulses = [0, 0]

    else:

      if echo == 'single':
        self.echo = 'single'
        self.num_pulses = [1, 2]

      elif echo == 'double':
        self.echo = 'double'
        self.num_pulses = [2, 2]

      else:
        self.echo = 'no_echo'
        self.num_pulses = [1, 1]


  def sequence(self, delays, amps, phis, drags, widths, truncs):

    self.update_duration(widths, truncs)
    width = widths[0]; trunc = truncs[0]
    norm = self.normalise_gaussian(trunc)

    amps = [1/width/4/norm * amps[0], 1/width/8/norm * amps[1], 1/width/4/norm * amps[2], 1/width/4/norm * amps[2]]
    drags = [self.anharms[0] * drags[0], self.anharms[1] * drags[1], self.anharms[1] * drags[2], self.anharms[0] * drags[2]]

    if self.initial[0] is not '0' or self.initial[1] is not '0':
      t_2pulses = 0

    else:
      t1 = trunc * width / 2
      self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss_DRAG',amp=amps[0],centre=t1,width=width,trunc=trunc,alpha=drags[0])

      t2 = trunc * width * 3 / 2
      self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss_DRAG',amp=amps[1],centre=t2,width=width,trunc=trunc,alpha=drags[1])

      t_2pulses = trunc * width * 2

    if self.echo == 'single':
      t3 = t_2pulses + trunc * width / 2 + delays[0] / 2
      self.add_pulse(index=0,f=self.freqs[1],phi=phis[2],shape='gauss_DRAG',amp=amps[2],centre=t3,width=width,trunc=trunc,alpha=drags[2])

    elif self.echo == 'double':
      t3 = trunc * width * 5 / 2 + delays[0] / 2
      self.add_pulse(index=0,f=self.freqs[1],phi=phis[2],shape='gauss_DRAG',amp=amps[2],centre=t3,width=width,trunc=trunc,alpha=drags[2])

      t4 = trunc * width * 7 / 2 + delays[0] / 2
      self.add_pulse(index=1,f=self.freqs[0],phi=phis[3],shape='gauss_DRAG',amp=amps[3],centre=t4,width=width,trunc=trunc,alpha=drags[3])

    else:
      pass

    # t4 = trunc * width * 7 / 2 + delays[0] + delays[1]
    # self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss_DRAG',amp=1/width/8/norm,centre=t4,width=width,trunc=trunc,alpha=self.anharms[0])
    # t5 = trunc * width * 9 / 2 + delays[0] + delays[1]
    # self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss_DRAG',amp=1/width/8/norm,centre=t5,width=width,trunc=trunc,alpha=self.anharms[1])

    final_time = self.total_duration - 0.0005# trunc * width * np.sum(self.num_pulses) + np.sum(delays)

    meas_time = [final_time]

    return meas_time

class pulse_train_2q(Experiment):

  def __init__(self, freqs, anharms, T1s, T2s, J, files, Hamiltonian=None):

    if Hamiltonian is 'dispersive' or 'dipole':
      if anharms[0] == 0.0 and anharms[1] == 0.0:
        level = 2

      else:
        level = 3

      super(pulse_train_2q, self).__init__(levels=level, total_time=1.0, time_resolution=0.0005)

      self.add_qubit(freq=freqs[0], anharmonicity=anharms[0], T1=T1s[0], T2=T2s[0])
      self.add_qubit(freq=freqs[1], anharmonicity=anharms[1], T1=T1s[1], T2=T2s[1])

      if Hamiltonian is 'dispersive':
        self.set_connectivity(0,1,J,"ZZ")

      elif Hamiltonian is 'dipole':
        self.set_connectivity(0,1,J,"XX")

      self.set_hamiltonian()
      self.set_rot_frame(index=0,freq=freqs[0])
      self.set_rot_frame(index=1,freq=freqs[1])

    else:
      super(pulse_train_2q, self).__init__(levels=2, total_time=1.0, time_resolution=0.0005)
      for term in Hamiltonian:
        self.add_terms(omega=term[0], pauli_2q=term[1])
      for i, t1 in enumerate(T1s):
        self.qubits['T1'].append(T1s[i])
        self.qubits['T2'].append(T2s[i])


    self.N_pulses = 0
    self.train = 0

    self.freqs = freqs
    self.anharms = anharms
    self.files = files

    self.initial = [0,0]

    self.meas_time = []
    self.qobj = Qobj()

  def update_duration(self, widths, truncs):

    self.widths = widths
    self.truncs = truncs
    self.durations = []

    for i in range(len(self.widths)):
      self.durations.append(self.widths[i]*self.truncs[i])

  def normalise_gaussian(self, trunc):

    x = np.arange(-trunc, trunc, 0.001) # range of x in spec
    x_all = np.arange(-10, 10, 0.001) # entire range of x, both in and out of spec

    return np.sum(norm.pdf(x,0,1))/np.sum(norm.pdf(x_all,0,1))

  def total_time(self, durations, n_pulses, buffer):

    self.duration = (sum(n_pulses) + 1) * buffer
    for n, t in enumerate(durations):
      self.duration += n_pulses[n] * t

    return self.duration

  def run_test(self, amps, phis, widths, truncs, buffer, no_pulse=False):

    self.update_duration(widths, truncs)
    self.total_duration = self.total_time(self.durations, self.N_pulse_list, buffer)
    self.tlist = np.arange(0., self.total_duration, 0.0005)
    self.set_clock(total_time=self.total_duration, time_resolution=0.0005)
    self.clear_pulses()
    self.meas_time = self.sequence(self.N_pulses, amps, phis, widths, truncs, buffer, self.files)
    if no_pulse: self.clear_pulses()
    self.initialization(initial=self.initial)
    self.q_obj = self.run()

  def fidelity_list(self):

    ideal_den = self.psi * self.psi.dag()

    fidelities = []
    for mt in self.meas_time:
      index = self.time_to_index(mt)
      real = self.q_obj.states[index]
      fidelities.append(fidelity(ideal_den, real))

    return fidelities

  def fidelity_fit(self, x, a):
    return np.cos(2*np.pi * a * x)

  def plot(self, full_trace=False, sequence=False, pauli=False, tomography=False, bloch=False, corr=False, error=False, save=None):

    self.save = save

    if full_trace:
      time = self.tlist
    else:
      time = self.meas_time

    if sequence:
      self.show_pulse_sequence()

    if pauli:
      self.show_pauli(self.q_obj.states, ['ZI','IZ','XX','YY','ZZ'], time) # [0::10])

    if tomography:
      self.tomography_2q(self.q_obj.states, [0,1], time)

    if bloch:
      self.show_bloch(self.q_obj.states, 0, time) # [0::10])
      self.show_bloch(self.q_obj.states, 1, time) # [0::10])

    if corr:
      self.show_corr(self.q_obj.states, [0,1], time, show_qst=True)

    if error:

      N = self.N_pulses
      k = self.train
      x_data = np.linspace(0,N*k,N+1)
      y_data = self.fidelity_list()
      print(x_data, y_data)
      popt, pcov = optimize.curve_fit(self.fidelity_fit, x_data, y_data, p0=[0.01])

      print('Fidelity:', 1 - 360*popt[0]/(360/k)) # rotation error: 360*popt[0], total rotation angle: 360/k
      x_fit = np.linspace(0,N*k+1,500)
      y_fit = self.fidelity_fit(x_fit, popt[0])
      plt.plot(x_data, y_data)
      plt.plot(x_fit, y_fit)
      plt.show()

class x90_train(pulse_train_2q):

  def __init__(self, freqs, anharms, T1s, T2s, J, files, N_pulses, initial=None, Hamiltonian=None):
    super(x90_train, self).__init__(freqs, anharms, T1s, T2s, J, files, Hamiltonian)

    self.N_pulses = N_pulses
    self.train = 4
    self.N_pulse_list = [1,N_pulses*4]
    self.initial = initial

  def sequence(self, N, amps, phis, widths, truncs, buffer, files):
      
    self.update_duration(widths, truncs)
    width = widths[0]; trunc = truncs[0]
    norm = self.normalise_gaussian(trunc)
    
    # self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss_DRAG',amp=1/width/8/norm,centre=trunc*width/2,width=width,trunc=trunc,alpha=self.anharms[0])
    # self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss_DRAG',amp=1/width/8/norm,centre=trunc*width/2,width=width,trunc=trunc,alpha=self.anharms[1])
    
    t = 0# trunc*width
    meas_time = [t]

    for n in range(N):
        
      t1 = t + buffer + trunc*width/2 
      self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss_DRAG',amp=1/width/8/norm,centre=t1,width=width,trunc=trunc,alpha=self.anharms[0])
      self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss_DRAG',amp=1/width/8/norm,centre=t1,width=width,trunc=trunc,alpha=self.anharms[1])
    
      t2 = t1 + trunc*width/2 + buffer + trunc*width/2
      self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss_DRAG',amp=1/width/8/norm,centre=t2,width=width,trunc=trunc,alpha=self.anharms[0])
      self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss_DRAG',amp=1/width/8/norm,centre=t2,width=width,trunc=trunc,alpha=self.anharms[1])
      
      t3 = t2 + trunc*width/2 + buffer + trunc*width/2 
      self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss_DRAG',amp=1/width/8/norm,centre=t3,width=width,trunc=trunc,alpha=self.anharms[0])
      self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss_DRAG',amp=1/width/8/norm,centre=t3,width=width,trunc=trunc,alpha=self.anharms[1])
    
      t4 = t3 + trunc*width/2 + buffer + trunc*width/2
      self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss_DRAG',amp=1/width/8/norm,centre=t4,width=width,trunc=trunc,alpha=self.anharms[0])
      self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss_DRAG',amp=1/width/8/norm,centre=t4,width=width,trunc=trunc,alpha=self.anharms[1])
      
      t = t4 + trunc*width/2
      meas_time.append(t)
    
    return meas_time

class x180_train(pulse_train_2q):

  def __init__(self, freqs, anharms, T1s, T2s, J, files, N_pulses, initial=None, Hamiltonian=None):
    super(x180_train, self).__init__(freqs, anharms, T1s, T2s, J, files, Hamiltonian)

    self.width = 0.01
    self.trunc = 3.0
    self.update_duration(widths=[self.width, self.width], truncs=[self.trunc, self.trunc])

    self.N_pulses = N_pulses
    self.train = 2
    self.N_pulse_list = [1,N_pulses*2]
    self.initial = initial

  def sequence(self, N, amps, phis, widths, truncs, buffer, files):
    self.update_duration(widths, truncs)
    width = widths[0]; trunc = truncs[0]
    norm = self.normalise_gaussian(trunc)
    
    # self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss',amp=1/width/4/norm,centre=trunc*width/2,width=width,trunc=trunc,alpha=self.anharms[0])
    # self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss',amp=1/width/4/norm,centre=trunc*width/2,width=width,trunc=trunc,alpha=self.anharms[1])

    t = 0 # trunc*width
    meas_time = [t]

    for n in range(N):
        
      t1 = t + buffer + trunc*width/2 
      self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss',amp=1/width/4/norm,centre=t1,width=width,trunc=trunc,alpha=self.anharms[0])
      self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss',amp=1/width/4/norm,centre=t1,width=width,trunc=trunc,alpha=self.anharms[1])
    
      t2 = t1 + trunc*width/2 + buffer + trunc*width/2
      self.add_pulse(index=0,f=self.freqs[0],phi=phis[0],shape='gauss',amp=1/width/4/norm,centre=t2,width=width,trunc=trunc,alpha=self.anharms[0])
      self.add_pulse(index=1,f=self.freqs[1],phi=phis[1],shape='gauss',amp=1/width/4/norm,centre=t2,width=width,trunc=trunc,alpha=self.anharms[1])
      t = t2 + trunc*width/2
      meas_time.append(t)
    
    return meas_time

class GRAPE_x90_train(pulse_train_2q):

  def __init__(self, freqs, anharms, T1s, T2s, J, files, N_pulses, initial=None, Hamiltonian=None):
    super(GRAPE_x90_train, self).__init__(freqs, anharms, T1s, T2s, J, files, Hamiltonian)

    self.update_duration(widths=[0.03,0.03], truncs=[1.0,1.0])

    self.N_pulses = N_pulses
    self.train = 4
    self.N_pulse_list = [1, N_pulses*4]
    self.initial = initial

  def sequence(self, N, amps, phis, widths, truncs, buffer, files):

    # self.add_grape(index_1=0,index_2=1,amp=amps,freq=self.freqs,phi=phis,centre=widths[0]/2,width=widths[0], file=files[0])

    t = 0 # widths[0]
    meas_time = [t]

    for n in range(N):

      t1 = t + buffer + widths[0]/2 
      self.add_grape(index_1=0,index_2=1,amp=amps,freq=self.freqs,phi=phis,centre=t1,width=widths[1], file=files[0])
      t2 = t1 + widths[0]/2 + buffer + widths[1]/2
      self.add_grape(index_1=0,index_2=1,amp=amps,freq=self.freqs,phi=phis,centre=t2,width=widths[1], file=files[0])
      t3 = t2 + widths[0]/2 + buffer + widths[1]/2
      self.add_grape(index_1=0,index_2=1,amp=amps,freq=self.freqs,phi=phis,centre=t3,width=widths[1], file=files[0])
      t4 = t3 + widths[0]/2 + buffer + widths[1]/2
      self.add_grape(index_1=0,index_2=1,amp=amps,freq=self.freqs,phi=phis,centre=t4,width=widths[1], file=files[0])
      t = t4 + widths[0]/2
      # print('time (pulse):', t)
      meas_time.append(t)

    return meas_time

class GRAPE_x180_train(pulse_train_2q):

  def __init__(self, freqs, anharms, T1s, T2s, J, files, N_pulses, initial=None, Hamiltonian=None):
    super(GRAPE_x180_train, self).__init__(freqs, anharms, T1s, T2s, J, files, Hamiltonian)

    self.update_duration(widths=[0.03,0.05], truncs=[1.0,1.0])

    self.N_pulses = N_pulses
    self.train = 2
    self.N_pulse_list = [1,N_pulses*2]
    self.initial = initial

  def sequence(self, N, amps, phis, widths, truncs, buffer, files):

    # self.add_grape(index_1=0,index_2=1,amp=amps,freq=self.freqs,phi=phis,centre=widths[1]/2,width=widths[1], file=files[1])

    t = 0 # widths[1]
    meas_time = [t]

    for n in range(N):
      t1 = t + buffer + widths[1]/2 
      self.add_grape(index_1=0,index_2=1,amp=amps,freq=self.freqs,phi=phis,centre=t1,width=widths[1], file=files[1])
      t2 = t1 + widths[1]/2 + buffer + widths[1]/2
      self.add_grape(index_1=0,index_2=1,amp=amps,freq=self.freqs,phi=phis,centre=t2,width=widths[1], file=files[1])
      t = t2 + widths[1]/2
      meas_time.append(t)
    
    return meas_time

