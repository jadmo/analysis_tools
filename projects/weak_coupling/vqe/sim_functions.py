# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import re
import time

# test

def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])
        
class Experiment(object):

    def __init__(self, levels, total_time, time_resolution):
        self.levels = levels
        self.a = destroy(levels)
        self.n = destroy(levels).dag()*destroy(levels)
        self.I = qeye(levels)
        self.g = basis(self.levels, 0)  # ground state
        self.e = basis(self.levels, 1)  # excited state

        self.qubits = {'freq':[],'frame':[],'anharmonicity':[],'T1':[],'T2':[],'num':0,'levels':levels}

        self.connectivity = []
        self.tlist = np.arange(0., total_time, time_resolution)
        self.H = None
        self.H0 = []
        self.states = None
        self.Cos = [np.cos(2. * np.pi * (self.qubits['frame'][i]) * self.tlist) for i in range(self.qubits['num'])]
        self.Sin = [np.sin(2. * np.pi * (self.qubits['frame'][i]) * self.tlist) for i in range(self.qubits['num'])]

        self.X = basis(levels, 0) * basis(levels, 1).dag() + basis(levels, 1) * basis(levels, 0).dag()
        self.Y = -1.0j * (basis(levels, 0) * basis(levels, 1).dag() - basis(levels, 1) * basis(levels, 0).dag())
        self.Z = -     (basis(levels, 1) * basis(levels, 1).dag() - basis(levels, 0) * basis(levels, 0).dag())

    def add_qubit(self,freq,anharmonicity,T1,T2):
        self.qubits['freq'].append(freq)
        self.qubits['frame'].append(freq)
        self.qubits['anharmonicity'].append(anharmonicity)
        self.qubits['T1'].append(T1)
        self.qubits['T2'].append(T2)
        self.qubits['num'] += 1

    def set_clock(self, total_time, time_resolution):
        self.tlist = np.arange(0., total_time, time_resolution)

        self.Cos = [np.cos(2. * np.pi * (self.qubits['frame'][i]) * self.tlist) for i in range(self.qubits['num'])]
        self.Sin = [np.sin(2. * np.pi * (self.qubits['frame'][i]) * self.tlist) for i in range(self.qubits['num'])]
        
    def time_to_index(self, time):
        dt = self.tlist[1]
        return int(round(time / dt))

    def set_rot_frame(self,index,freq):
        self.qubits['frame'][index] = freq

        self.Cos = [np.cos(2. * np.pi * (self.qubits['frame'][i]) * self.tlist) for i in range(self.qubits['num'])]
        self.Sin = [np.sin(2. * np.pi * (self.qubits['frame'][i]) * self.tlist) for i in range(self.qubits['num'])]

    def set_connectivity(self,index1,index2,coupling,type):
        self.connectivity.append([index1,index2,coupling,type])

    def refresh_connectivity(self):
        self.connectivity = []
    
    def set_hamiltonian(self):

        self.H0 = []

        # Qubits
        for i in range(self.qubits['num']):
            one_body_terms= [self.I for j in range(self.qubits['num'])]
            if self.levels == 2:
                one_body_terms[i] = (self.qubits['freq'][i]) * (self.n - self.I/2) #Z / 2 #self.Z
                # print(self.qubits['freq'])
            else:
                one_body_terms[i] = (self.qubits['freq'][i]+self.qubits['anharmonicity'][i]/2.0*(self.n-1)) * self.n - self.qubits['freq'][i] * self.I/2

            self.H0 += [2.*np.pi*tensor(one_body_terms),]

        # Couplings
        for i in self.connectivity:
            if i[3] is "XX":
                two_body_terms = [self.I for j in range(self.qubits['num'])]
                two_body_terms[i[0]] = self.a + self.a.dag()
                two_body_terms[i[1]] = self.a + self.a.dag()
                self.H0 += [2.*np.pi*i[2]*tensor(two_body_terms),]

                # two_body_terms = tensor([self.a.dag(), self.a])  + tensor([self.a, self.a.dag()])
                # self.H0 += [2.*np.pi*i[2]*two_body_terms,]

            elif i[3] is "XX_RWA":
                two_body_terms = tensor([self.a.dag(), self.a])  + tensor([self.a, self.a.dag()])
                self.H0 += [2.*np.pi*i[2]*two_body_terms,]

            elif i[3] is "ZZ":
                two_body_terms = [self.I for j in range(self.qubits['num'])]
                two_body_terms[i[0]] = self.n - self.I/2
                two_body_terms[i[1]] = self.n - self.I/2
                self.H0 += [2.*np.pi*i[2]*tensor(two_body_terms),]

    def add_terms(self, omega, pauli_2q):

        split_pauli = pauli_2q.split()
        self.qubits['num'] = len(split_pauli)
        self.qubits['freq'] = [0.0] * self.qubits['num']
        self.qubits['frame'] = [0.0] * self.qubits['num']
        two_body_terms = [self.I for j in range(self.qubits['num'])]

        for i, pauli in enumerate(split_pauli):

            if pauli == 'I':
                two_body_terms[i] = self.I
            elif pauli == 'X':
                two_body_terms[i] = self.a + self.a.dag()
            elif pauli == 'Y':
                two_body_terms[i] = 1.0j * self.a.dag() - 1.0j * self.a
            elif pauli == 'Z':
                two_body_terms[i] = self.n - self.I/2

        self.H0 += [2.*np.pi*omega*tensor(two_body_terms),]

    # def add_terms(self, omega, pauli_2q):
    #
    #     split_pauli = pauli_2q.split()
    #     self.qubits['num'] = len(split_pauli)
    #     self.qubits['freq'] = [0.0] * self.qubits['num']
    #     self.qubits['frame'] = [0.0] * self.qubits['num']
    #     two_body_terms = [self.I for j in range(self.qubits['num'])]
    #
    #     for i, pauli in enumerate(split_pauli):
    #
    #         if pauli == 'I':
    #             two_body_terms[i] = self.I
    #         elif pauli == 'X':
    #             two_body_terms[i] = self.X
    #         elif pauli == 'Y':
    #             two_body_terms[i] = self.Y
    #         elif pauli == 'Z':
    #             two_body_terms[i] = self.Z
    #
    #     self.H0 += [2.*np.pi*omega*tensor(two_body_terms),]

    def set_lindblad(self):

        # T1
        self.lindblad_T1s = 0#None
        for i in range(self.qubits['num']):
            T1_terms = [self.I for j in range(self.qubits['num'])]
            T1_terms[i] = self.a
            self.lindblad_T1s += np.sqrt(1/self.qubits['T1'][i])*tensor(T1_terms)
        
        # T2
        self.lindblad_T2s = 0#None
        for i in range(self.qubits['num']):
            T2_terms = [self.I for j in range(self.qubits['num'])]
            T2_terms[i] = self.n
            T2_star = 1. / (1./self.qubits['T2'][i]-1./self.qubits['T1'][i]/2.)
            self.lindblad_T2s += np.sqrt(1/T2_star)*tensor(T2_terms)

    def clear_pulses(self):
        # self.set_hamiltonian()
        self.H = self.H0[:]
        self.pulse_sequence = np.zeros((self.qubits['num'], 2, len(self.tlist)))

    def initialization(self, initial):
        psi = []
        for i in range(self.qubits['num']):
            if initial[i]=='+':
                psi.append((basis(self.qubits['levels'],0) + basis(self.qubits['levels'],1)) / np.sqrt(2))
            elif initial[i]=='-':
                psi.append((basis(self.qubits['levels'],0) - basis(self.qubits['levels'],1)) / np.sqrt(2))
            elif initial[i]=='+i':
                psi.append((basis(self.qubits['levels'],0) + 1.0j * basis(self.qubits['levels'],1)) / np.sqrt(2))
            elif initial[i]=='-i':
                psi.append((basis(self.qubits['levels'],0) - 1.0j * basis(self.qubits['levels'],1)) / np.sqrt(2))
            elif initial[i]=='0':
                psi.append(basis(self.qubits['levels'],0))
            elif initial[i]=='1':
                psi.append(basis(self.qubits['levels'],1))

        self.psi = tensor(psi)
        self.rho0 = tensor(psi)*tensor(psi).dag()

    def pulse_for_c(self,f=5000.,phi=0.,shape='gauss',amp=1.0,centre=0.0,width=0.0,trunc=1.0,rise=0.0,alpha=0.0):

        in_phase = "%lf*cos(%lf*t+%lf)*" % (2. * np.pi * amp, 2. * np.pi * f, phi / 180. * np.pi)
        
        qu_phase = "%lf*sin(%lf*t+%lf)*" % (2. * np.pi * amp, 2. * np.pi * f, phi / 180. * np.pi)

        hwidth = width / 2.

        if shape == 'gauss':
            adjusted_omega = 2. / np.pi ** 0.5
            in_phase += "%lf*exp(-((t-%lf)/%lf)**2)*0.5*(tanh((t-%lf)/0.001)-tanh((t-%lf)/0.001))"%(adjusted_omega,centre,hwidth,centre-hwidth*trunc,centre+hwidth*trunc)
            qu_phase += "0"
            
        if shape == 'gauss_DRAG':
            adjusted_omega = 2. / np.pi ** 0.5
            in_phase += "%lf*exp(-((t-%lf)/%lf)**2)*0.5*(tanh((t-%lf)/0.001)-tanh((t-%lf)/0.001))"%(adjusted_omega,centre,hwidth,centre-hwidth*trunc,centre+hwidth*trunc)
            qu_phase += "2.0*(t-%lf)/(%lf*%lf)*%lf*exp(-((t-%lf)/%lf)**2)*0.5*(tanh((t-%lf)/0.001)-tanh((t-%lf)/0.001))"%(centre,2.0*np.pi*2.0*alpha,hwidth**2,adjusted_omega,centre,hwidth,centre-hwidth*trunc,centre+hwidth*trunc)

        if shape == 'soft':
            in_phase += "0.5*(tanh((t-%lf)/%lf)-tanh((t-%lf)/%lf))" % (centre - hwidth, rise, centre + hwidth, rise)
            qu_phase += "0"

        # in_phase += ")"
        
        # qu_phase += ")"

        return in_phase + "+" + qu_phase

    def pulse_for_plt(self,index=0,f=5000.,phi=0.,shape='gauss',amp=1.0,centre=0.0,width=0.0,trunc=1.0,rise=0.0,alpha=0.0):
        pulse_sequence = np.zeros((self.qubits['num'],2,len(self.tlist)))

        hwidth = width / 2.

        if shape == 'gauss':
            adjusted_omega = 2. / np.pi ** 0.5
            start = centre - hwidth * trunc
            end = centre + hwidth * trunc
            envelope_i = adjusted_omega * np.exp(-((self.tlist-centre)/hwidth)**2)*0.5*(np.tanh((self.tlist-start)/0.001)-np.tanh((self.tlist-end)/0.001))
            envelope_q = adjusted_omega * np.exp(-((self.tlist-centre)/hwidth)**2)*0.5*(np.tanh((self.tlist-start)/0.001)-np.tanh((self.tlist-end)/0.001))
            
        if shape == 'gauss_DRAG':
            adjusted_omega = 2. / np.pi ** 0.5
            start = centre - hwidth * trunc
            end = centre + hwidth * trunc
            envelope_i = adjusted_omega * np.exp(-((self.tlist-centre)/hwidth)**2)*0.5*(np.tanh((self.tlist-start)/0.001)-np.tanh((self.tlist-end)/0.001))
            envelope_q = 2.0 * (self.tlist - centre) / (2.0*np.pi*2.0*alpha*hwidth**2) * adjusted_omega * np.exp(-((self.tlist-centre)/hwidth)**2)*0.5*(np.tanh((self.tlist-start)/0.001)-np.tanh((self.tlist-end)/0.001))

        if shape == 'soft':
            start = centre - hwidth
            end = centre + hwidth
            envelope_i = 0.5*(np.tanh((self.tlist-start)/rise)-np.tanh((self.tlist-end)/rise))
            envelope_q = 0.5*(np.tanh((self.tlist-start)/rise)-np.tanh((self.tlist-end)/rise))

        pulse_sequence[index][0] = 2. * np.pi * amp * envelope_i # * np.pi * amp * np.cos(2. * np.pi * f +  phi / 180. * np.pi)
        pulse_sequence[index][1] = 2. * np.pi * amp * envelope_q # * np.pi * amp * np.sin(2. * np.pi * f +  phi / 180. * np.pi)
        # print(pulse_sequence)
        return pulse_sequence

    def add_pulse(self,index,f=5000.,phi=0.,shape='gauss',amp=1.0,centre=0.0,width=0.0,trunc=1.0,rise=0.0,alpha=0.0,file=None):
        
        if shape=='grape_2q':
            self.add_grape(index[0], index[1], amp, freq, phi, centre, width, file=None)
        
        else:
            Hd = [self.I for j in range(self.qubits['num'])]
            Hd[index] = self.a + self.a.dag()
    
            self.H += [[tensor(Hd), self.pulse_for_c(f,phi,shape,amp,centre,width,trunc,rise,alpha)],]
    
            self.pulse_sequence += self.pulse_for_plt(index,f,phi,shape,amp,centre,width,trunc,rise,alpha)

    def add_grape(self,index_1=0,index_2=1,amp=None,freq=None,phi=None,centre=0.0,width=0.0,file=None):
        #global self.H

        Hd1x = [self.I for j in range(self.qubits['num'])]
        Hd1x[index_1] = self.a + self.a.dag()

        Hd2x = [self.I for j in range(self.qubits['num'])]
        Hd2x[index_2] = self.a + self.a.dag()

        Hd1y = [self.I for j in range(self.qubits['num'])]
        Hd1y[index_1] = - 1.0j * self.a + 1.0j * self.a.dag()

        Hd2y = [self.I for j in range(self.qubits['num'])]
        Hd2y[index_2] = - 1.0j * self.a + 1.0j * self.a.dag()

        file = open(file, 'r')
        grape_data = file.read().split('\n')

        time_resolution = self.tlist[1] - self.tlist[0]
        centre_index = int(centre / time_resolution)
        half_width = int(width / time_resolution / 2.)
        start = centre_index - half_width
        end = centre_index + half_width

        pre_grape = []
        for i in grape_data:
            pre_grape.append([float(j) for j in i.split('\t')])

        grape = [np.zeros(len(self.tlist)) for i in range(2 * self.qubits['num'])]

        for i in range(2 * self.qubits['num']):
            grape[i][start:end] = np.array(pre_grape).T[i]

        S_1x = Cubic_Spline(self.tlist[0], self.tlist[-1], grape[0])
        S_2x = Cubic_Spline(self.tlist[0], self.tlist[-1], grape[1])
        S_1y = Cubic_Spline(self.tlist[0], self.tlist[-1], grape[2])
        S_2y = Cubic_Spline(self.tlist[0], self.tlist[-1], grape[3])

        S_1x = S_1x(self.tlist) * amp[0] / 1000000.
        S_2x = S_2x(self.tlist) * amp[1] / 1000000.
        S_1y = S_1y(self.tlist) * amp[2] / 1000000.
        S_2y = S_2y(self.tlist) * amp[3] / 1000000.
        
        in_phase_1 = 2. * np.pi * np.cos(2. * np.pi * freq[0] * self.tlist + phi[0] / 180. * np.pi)
        in_phase_2 = 2. * np.pi * np.cos(2. * np.pi * freq[1] * self.tlist + phi[1] / 180. * np.pi)
        qu_phase_1 = 2. * np.pi * np.sin(2. * np.pi * freq[0] * self.tlist + phi[2] / 180. * np.pi)
        qu_phase_2 = 2. * np.pi * np.sin(2. * np.pi * freq[1] * self.tlist + phi[3] / 180. * np.pi)

        self.H += [[tensor(Hd1x), S_1x/2.], ] # in_phase_1 * S_1x/2.], ]
        self.H += [[tensor(Hd2x), S_2x/2.], ] # in_phase_2 * S_2x/2.], ]
        self.H += [[tensor(Hd1y), S_1y/2.], ] # qu_phase_1 * S_1y/2.], ]
        self.H += [[tensor(Hd2y), S_2y/2.], ] # qu_phase_2 * S_2y/2.], ]

        pulse_sequence = np.zeros((self.qubits['num'], 2, len(self.tlist)))
        pulse_sequence[index_1][0] = S_1x
        pulse_sequence[index_1][1] = S_1y
        pulse_sequence[index_2][0] = S_2x
        pulse_sequence[index_2][1] = S_2y

        self.pulse_sequence += pulse_sequence

    def run(self):

        self.set_lindblad()
        result = mesolve(H=self.H,rho0=self.rho0,tlist=self.tlist, c_ops=[self.lindblad_T1s,self.lindblad_T2s] ,e_ops=[], options=Options(gui=False))

        return result
        
    def heisenberg(self, pauli_ind, freq, time_points):
        # Annihilation/creation operator of the first 2-levels in the rotating frame for measuring in the computational basis
        a_list = [np.exp(+1.0j * 2 * np.pi * freq * t) * self.g * self.e.dag() for t in time_points]
        c_list = [np.exp(-1.0j * 2 * np.pi * freq * t) * self.e * self.g.dag() for t in time_points]
        # print(freq)

        if pauli_ind == 'I':
            time_pauli = [qeye(self.levels) for i in range(len(time_points))]
        elif pauli_ind == 'X':
            time_pauli = [a + c for a, c in zip(a_list, c_list)]
        elif pauli_ind == 'Y':
            time_pauli = [1.0j * c - 1.0j * a for a, c in zip(a_list, c_list)]
        elif pauli_ind == 'Z':
            time_pauli = [self.g * self.g.dag() - self.e * self.e.dag() for i in range(len(time_points))] # (self.g * self.g.dag() - self.e * self.e.dag()) / 2

        return time_pauli

    def exp_value(self, states, Pauli, time_points):
        
        time_indices = [self.time_to_index(t) for t in np.array(time_points)]

        pauli_time = []
        for i, pauli_ind in enumerate(list(Pauli)):
            pauli_time.append(self.heisenberg(pauli_ind, self.qubits['frame'][i], time_points))    # pauli_time: (800,2)
            
        # print(len(pauli_time))
        # print(np.shape((pauli_time[0])))
        # print(time_indices)
        # print(time_points)
        # print(len(states))

        tensor_pauli_time = [expect(states[t_i],tensor([pauli_time[j][i] for j in range(self.qubits['num'])])) for i, t_i in enumerate(time_indices)]

        return tensor_pauli_time

    def show_pauli(self, states, Paulis, time_points):
        plt.figure()
        for Pauli in Paulis:
            exp_v_list = self.exp_value(states, Pauli, time_points)
            plt.plot(time_points, exp_v_list, label=Pauli)

        plt.title("Time dependence of Pauli operators", fontsize=20)
        plt.legend()
        if self.save is None:
            pass
        else:
            file_name = self.save + 'pauli'
            plt.savefig(file_name)
        plt.show()
        
    def show_pulse_sequence(self):
        
        
        for i in range(self.qubits['num']):
            # print(self.pulse_sequence[i])
            plt.figure(figsize=(6,4))
            plt.plot(self.tlist, self.pulse_sequence[i][0],'b')
            plt.plot(self.tlist, self.pulse_sequence[i][1],'r')
            plt.ylabel('Amplitude', fontsize=18) # Hopefully in MHz
            plt.xlabel('Time in microseconds', fontsize=18)
            plt.tight_layout()
            if self.save is None:
                pass
            else:
                file_name = self.save + 'pulse_sequence_%s'% i
                plt.savefig(file_name)
            plt.show()

    def show_bloch(self, states, q_index, time_points):

        pnts = []
        for pauli in ['X', 'Y', 'Z']:

            Pauli_list = list('I' * self.qubits['num'])
            Pauli_list[q_index] = pauli  # specifies X, Y or Z as a measurement operator
            exp = self.exp_value(states, ''.join(Pauli_list), time_points)
            pnts.append(exp)

        b = Bloch()
        b.add_points(pnts)
        # b.title("Bloch sphere representation of qubit %s"% q_index)
        if self.save is None:
            pass
        else:
            file_name = self.save + 'bloch_%s' % q_index
            b.save(file_name)
        b.show()
        
    def show_corr(self, states, q_indices, time_points, show_qst=False):
        
        time_indices = [self.time_to_index(t) for t in np.array(time_points)]
        
        label = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
        
        Pauli = ['I' for i in range(self.qubits['num'])]
        
        e_values = []
        for l in label:
            separate = list(l)
            Pauli[q_indices[0]] = list(l)[0]
            Pauli[q_indices[1]] = list(l)[1]
            e_value = self.exp_value(states, ''.join(Pauli), time_points)
            e_values.append(e_value)
            
        
        rii = e_values[0]; rix = e_values[1]; riy = e_values[2]; riz = e_values[3]
        rxi = e_values[4]; rxx = e_values[5]; rxy = e_values[6]; rxz = e_values[7]
        ryi = e_values[8]; ryx = e_values[9]; ryy = e_values[10]; ryz = e_values[11]
        rzi = e_values[12]; rzx = e_values[13]; rzy = e_values[14]; rzz = e_values[15]
        
        gxx = [rxx[i] - rix[i] * rxi[i] for i in range(len(time_points))]
        gyx = [ryx[i] - rix[i] * ryi[i] for i in range(len(time_points))]
        gzx = [rzx[i] - rix[i] * rzi[i] for i in range(len(time_points))]
        gxy = [rxy[i] - riy[i] * rxi[i] for i in range(len(time_points))]
        gyy = [ryy[i] - riy[i] * ryi[i] for i in range(len(time_points))]
        gzy = [rzy[i] - riy[i] * rzi[i] for i in range(len(time_points))]
        gxz = [rxz[i] - riz[i] * rxi[i] for i in range(len(time_points))]
        gyz = [ryz[i] - riz[i] * ryi[i] for i in range(len(time_points))]
        gzz = [rzz[i] - riz[i] * rzi[i] for i in range(len(time_points))]
        
        gx = [gxx, gyx, gzx]
        gy = [gxy, gyy, gzy]
        gz = [gxz, gyz, gzz]
        
        b = Bloch()
        b.xlabel = ['$xx,xy,xz$', '']
        b.ylabel = ['$yx,yy,yz$', '']
        b.zlabel = ['$zx,zy,zz$', '']
        b.xlpos = [1.1, -1.1]
        b.ylpos = [1.1, -1.1]
        b.zlpos = [1.1, -1.1]
        b.figsize = [6,6]
        b.add_points(gx)
        b.add_points(gy)
        b.add_points(gz)
        if self.save is None:
            pass
        else:
            file_name = self.save + 'corr_bloch'
            b.save(file_name)
        b.show()
        
        if show_qst:
            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12,8))
            label = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
            gs = [rii, rix, riy, riz, rxi, gxx, gxy, gxz, ryi, gyx, gyy, gyz, rzi, gzx, gzy, gzz]
            for i, g in enumerate(gs):
                pc = len(g)
                pulse_count = np.linspace(0,pc-1,pc)
                axes[int(i / 4), i % 4].set_title(r'$\langle{%s}\rangle$' % label[i])
                axes[int(i / 4), i % 4].scatter(pulse_count, g, s=10, marker=".")
                axes[int(i / 4), i % 4].set_ylim(-1.1, 1.1)
            fig.tight_layout()
            if self.save is None:
                pass
            else:
                file_name = self.save + 'corr_tomography'
                plt.savefig(file_name)
            plt.show()

    def tomography_2q_data(self, states, q_indices, time_points):

        label = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]

        Pauli = ['I' for i in range(self.qubits['num'])]

        Paulis = []
        for l in label:
            Pauli[q_indices[0]] = list(l)[0]
            Pauli[q_indices[1]] = list(l)[1]
            Paulis.append(''.join(Pauli))

        e_value = []
        for i, Pauli in enumerate(Paulis):
            e_value.append(self.exp_value(states, Pauli, time_points))

        return np.array(e_value)

    def tomography_2q(self, states, q_indices, time_points):

        label = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
        
        Pauli = ['I' for i in range(self.qubits['num'])]
        
        Paulis = []
        for l in label:
            separate = list(l)
            Pauli[q_indices[0]] = list(l)[0]
            Pauli[q_indices[1]] = list(l)[1]
            Paulis.append(''.join(Pauli))
        
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12,8))

        for i, Pauli in enumerate(Paulis):
            
            e_value = self.exp_value(states, Pauli, time_points)
            pc = len(e_value)
            pulse_count = np.linspace(0,pc-1,pc)
            axes[int(i / 4), i % 4].set_title(r'$\langle{%s}\rangle$' % label[i])
            axes[int(i / 4), i % 4].scatter(pulse_count, e_value, s=10, marker=".")
            axes[int(i / 4), i % 4].set_ylim(-1.1, 1.1)
        fig.tight_layout()
        if self.save is None:
            pass
        else:
            file_name = self.save + 'tomography'
            plt.savefig(file_name)
            
        plt.show()
        


